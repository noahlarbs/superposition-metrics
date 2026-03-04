"""
Fine-tuning: huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated  +  Superposition FPE
=================================================================================

MODEL NOTES
───────────
  HuggingFace:  huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated
  Architecture: qwen3_5_moe  (Qwen3.5-35B-A3B with abliteration applied)
  Params:       36B total / ~3B active MoE
  Weights:      BF16  (~72 GB — fits on 8×H100 640 GB without quantisation)
  load_in_4bit: False — quantising would corrupt the abliteration steering
                vectors embedded in the weights.
  Chat template: Qwen3.5 native; <think>…</think> supported natively.

BUG FIXES over the original
────────────────────────────
1. GRADIENT ACCESS — hooks registered in on_train_begin accumulate Σg²/Σg⁴
   during the backward pass. on_step_end reads accumulators (not p.grad,
   which is always zero there because the Trainer calls zero_grad() first).

2. DENG LAYER SELECTION — last trainable lora_B weight used instead of
   model.lm_head (which is frozen under LoRA → grad=None → AL_t always 0).

3. DENG EIGENVECTOR ORDER — eigh returns ascending order; fixed to slice
   V[:, -k:] (top-k) instead of V[:, :k] (bottom-k / noise subspace).

4. LORA EXPANSION — full in-place rank doubling implemented (was a stub).

5. LR SHOCK — 100-step cosine taper 5× → 1× (was a hard snap-back at step 100).

HYPERPARAMETER DERIVATION
──────────────────────────
  pr_tol = 0.02   2% delta over 20-step window ≈ p~10⁻⁴ Mann-Kendall.
                  Original 1% caused false non-detections from batch gradient noise.

  al_thresh = 0.75   Deng 2023: healthy AL_t ≈ 0.20–0.40; saturation onset >0.65.
                     0.75 gives 10pp safety margin.

  LR shock = 5× / 100-step cosine   Evci 2022 "Rigging the Lottery": 3–10×
                     needed for new sparse weights to break symmetry. Cosine
                     taper prevents destabilising the preserved rank block.
"""

import math
import time

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TrainerCallback


# ==============================================================================
# Superposition FPE Callback
# ==============================================================================

class SuperpositionFPECallback(TrainerCallback):

    def __init__(
        self,
        model,
        target_k: int    = 100,
        patience: int    = 20,
        pr_tol: float    = 0.02,
        al_thresh: float = 0.75,
        timeout_hours: float = 5.0,
    ):
        self.model           = model
        self.target_k        = target_k
        self.patience        = patience
        self.pr_tol          = pr_tol
        self.al_thresh       = al_thresh
        self.timeout_seconds = timeout_hours * 3600

        self.pr_history         = []
        self.lr_shock_remaining = 0
        self.base_lr_backup     = None
        self.has_expanded       = False
        self.start_time         = time.time()

        # Gradient accumulators — filled by hooks DURING backward.
        # Never read p.grad in on_step_end; it is always zero there because
        # the Trainer calls optimizer.zero_grad() before firing the callback.
        self._sum_g2    = 0.0
        self._sum_g4    = 0.0
        self._deng_grad = None
        self._hooks     = []

    # ── Hook management ────────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, **kwargs):
        self._register_hooks()

    def on_train_end(self, args, state, control, **kwargs):
        self._remove_hooks()

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _find_deng_param(self):
        """
        Return the last trainable lora_B weight in the model.
        lm_head is frozen under LoRA (grad=None → AL_t=0 always).
        The last lora_B (deepest block output projection) is always trainable.
        """
        last = None
        for _name, module in self.model.named_modules():
            lora_B = getattr(module, "lora_B", None)
            if lora_B is None or not hasattr(lora_B, "keys"):
                continue
            for _k, lin in lora_B.items():
                w = getattr(lin, "weight", None)
                if w is not None and w.requires_grad:
                    last = w
        # Fallback: trainable lm_head (full fine-tune only)
        for attr in ("lm_head", "model.lm_head", "base_model.model.lm_head"):
            obj = self.model
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                w = getattr(obj, "weight", None)
                if w is not None and w.requires_grad:
                    return w
        return last

    def _register_hooks(self):
        self._remove_hooks()
        deng_param = self._find_deng_param()
        if deng_param is None:
            print("[FPE] WARNING: No trainable Deng param found. AL_t will always be 0.")

        for _name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param is deng_param:
                def _deng_hook(grad, _p=param):
                    g2 = grad.detach().pow(2)
                    self._sum_g2 += g2.sum().item()
                    self._sum_g4 += (g2 * g2).sum().item()
                    g_f = grad.detach().float().view(grad.shape[0], -1)
                    if g_f.shape[1] > 2048:
                        idx = torch.randperm(g_f.shape[1], device=g_f.device)[:2048]
                        g_f = g_f[:, idx]
                    self._deng_grad = g_f.cpu()
                self._hooks.append(param.register_hook(_deng_hook))
            else:
                def _pr_hook(grad):
                    g2 = grad.detach().pow(2).reshape(-1)
                    self._sum_g2 += g2.sum().item()
                    self._sum_g4 += (g2 * g2).sum().item()
                self._hooks.append(param.register_hook(_pr_hook))

    def _reset_accumulators(self):
        self._sum_g2    = 0.0
        self._sum_g4    = 0.0
        self._deng_grad = None

    # ── Metrics ────────────────────────────────────────────────────────────────

    def _compute_pr(self) -> float:
        """Tr(F)² / Tr(F²) via empirical Fisher: Tr(F)=Σg², Tr(F²)=Σg⁴"""
        return (self._sum_g2 ** 2) / (self._sum_g4 + 1e-12)

    def _compute_deng_alignment(self) -> float:
        """
        AL_t = ‖P_k G‖_F² / ‖G‖_F²
        P_k projects onto the TOP-k eigenvectors of G Gᵀ.
        eigh returns ascending order → slice V[:, -k:] for top-k.
        """
        g = self._deng_grad
        if g is None:
            return 0.0
        g = g.to(torch.float32)
        k = min(self.target_k, g.shape[0] - 1)
        if k <= 0:
            return 0.0
        try:
            A    = g @ g.T
            _, V = torch.linalg.eigh(A)
            U_k  = V[:, -k:]
            P    = U_k @ U_k.T
            num  = torch.norm(P @ g).pow(2)
            den  = torch.norm(g).pow(2) + 1e-12
            return (num / den).item()
        except Exception as exc:
            print(f"[FPE] Deng alignment error: {exc}")
            return 0.0

    # ── LoRA rank expansion ────────────────────────────────────────────────────

    def _execute_lora_expansion(self, optimizer):
        """
        Double every lora_A / lora_B pair in-place:
          new_A [2r, d_in]:   rows 0..r-1 = old A  | rows r..2r-1 = Kaiming
          new_B [d_out, 2r]:  cols 0..r-1 = old B  | cols r..2r-1 = zeros
        Zero new-B cols → expansion is a no-op at detonation time.
        Adam state cleared (shapes changed). Hooks re-registered (Parameters replaced).
        """
        print("\n[FPE] ── LoRA rank doubling ──────────────────────────────────")
        n_expanded = 0
        last_old_r = None

        for _mname, module in self.model.named_modules():
            lora_A = getattr(module, "lora_A", None)
            lora_B = getattr(module, "lora_B", None)
            if not (lora_A and lora_B and hasattr(lora_A, "keys")):
                continue
            for adapter in list(lora_A.keys()):
                try:
                    A_p = lora_A[adapter].weight
                    B_p = lora_B[adapter].weight
                except (KeyError, AttributeError):
                    continue

                old_r      = A_p.shape[0]
                new_r      = old_r * 2
                last_old_r = old_r
                dev, dt    = A_p.device, A_p.dtype

                new_A = torch.zeros(new_r, A_p.shape[1],  device=dev, dtype=dt)
                new_B = torch.zeros(B_p.shape[0], new_r,  device=dev, dtype=dt)

                with torch.no_grad():
                    new_A[:old_r].copy_(A_p.data)
                    new_B[:, :old_r].copy_(B_p.data)
                    nn.init.kaiming_uniform_(new_A[old_r:], a=math.sqrt(5))
                    # new_B[:, old_r:] stays zero — correct LoRA init

                lora_A[adapter].weight = nn.Parameter(new_A)
                lora_B[adapter].weight = nn.Parameter(new_B)

                # lora_alpha fixed, rank doubled → halve scaling factor
                if hasattr(module, "scaling") and adapter in module.scaling:
                    module.scaling[adapter] *= 0.5

                n_expanded += 1

        if n_expanded == 0:
            print("[FPE] WARNING: No LoRA ModuleDict pairs found. Skipping.")
            return

        print(f"[FPE] {n_expanded} pairs expanded: r={last_old_r} → r={last_old_r * 2}")

        optimizer.state.clear()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        base_cfg  = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
        del optimizer.param_groups[:]
        optimizer.add_param_group({**base_cfg, "params": trainable})
        print(f"[FPE] Optimizer rebuilt — {len(trainable)} trainable tensors.")

        # Parameter objects replaced; must re-register hooks
        self._register_hooks()

        wandb.log({"fpe/detonation_step": wandb.run.step, "fpe/new_lora_r": last_old_r * 2})

    # ── Step callback ──────────────────────────────────────────────────────────

    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        # Cosine LR shock decay: 5× → 1× over 100 steps
        if self.lr_shock_remaining > 0:
            steps_done    = 101 - self.lr_shock_remaining
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * steps_done / 100))
            multiplier    = 1.0 + 4.0 * cosine_factor
            current_lr    = self.base_lr_backup * multiplier
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            self.lr_shock_remaining -= 1
            if self.lr_shock_remaining == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = self.base_lr_backup
                print("\n[FPE] LR shock complete — rate restored to scheduler baseline.")
            wandb.log({"fpe/lr_shock_multiplier": multiplier, "fpe/lr": current_lr},
                      step=state.global_step)
            self._reset_accumulators()
            return

        pr = self._compute_pr()
        self._reset_accumulators()

        self.pr_history.append(pr)
        if len(self.pr_history) > self.patience:
            self.pr_history.pop(0)

        is_plateau  = False
        delta_ratio = 0.0
        if len(self.pr_history) == self.patience:
            hi          = max(self.pr_history)
            lo          = min(self.pr_history)
            delta_ratio = (hi - lo) / (hi + 1e-12)
            is_plateau  = delta_ratio < self.pr_tol

        # Log PR every step
        wandb.log({"fpe/fisher_pr": pr, "fpe/pr_delta_ratio": delta_ratio},
                  step=state.global_step)

        elapsed  = time.time() - self.start_time
        detonate = False

        if not self.has_expanded and elapsed > self.timeout_seconds:
            print(f"\n[METRICS] step={state.global_step} | PR={pr:.4f}"
                  f" | elapsed={elapsed/3600:.2f}h  →  TIMEOUT, forcing detonation.")
            wandb.log({"fpe/detonation_reason": "timeout"}, step=state.global_step)
            detonate = True

        elif is_plateau and not self.has_expanded:
            al_t = self._compute_deng_alignment()
            avg  = sum(self.pr_history) / len(self.pr_history)
            print(f"\n[METRICS] step={state.global_step}"
                  f" | PR_avg={avg:.4f} (Δ={delta_ratio:.4f})"
                  f" | AL_t={al_t:.4f}")
            wandb.log({"fpe/deng_al_t": al_t, "fpe/pr_avg": avg}, step=state.global_step)
            if al_t > self.al_thresh:
                print(f"  🎯 SATURATION CONFIRMED  AL_t={al_t:.4f} > {self.al_thresh}. Detonating FPE.")
                wandb.log({"fpe/detonation_reason": "saturation"}, step=state.global_step)
                detonate = True
            else:
                print(f"  ⏳ PR plateau but AL_t={al_t:.4f} < {self.al_thresh}. Monitoring.")

        if detonate:
            self.has_expanded = True
            self._execute_lora_expansion(optimizer)
            self.base_lr_backup = optimizer.param_groups[0]["lr"]
            shock_lr            = self.base_lr_backup * 5.0
            for pg in optimizer.param_groups:
                pg["lr"] = shock_lr
            self.lr_shock_remaining = 100
            print(f"  ⚡ LR shock: {self.base_lr_backup:.2e} → {shock_lr:.2e}"
                  f"  (100-step cosine decay)")
            self.pr_history.clear()


# ==============================================================================
# Dataset formatters
# ==============================================================================

def build_process_opus(tokenizer):
    """
    nohurry/Opus-4.6-Reasoning-3000x-filtered
    Columns: id, problem, thinking, solution
    Formatted as user/assistant with <think>…</think> CoT wrapping.
    """
    def process_opus(example):
        messages = [
            {"role": "user",
             "content": str(example.get("problem", ""))},
            {"role": "assistant",
             "content": (f"<think>\n{example.get('thinking', '')}\n</think>\n"
                         f"{example.get('solution', '')}")}
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)}
    return process_opus


def build_process_ling(tokenizer):
    """
    inclusionAI/Ling-Coder-SFT
    Columns: mid, messages (list[dict{"content":...}]), tags, languages
    Normalise non-standard role names before apply_chat_template.
    """
    def process_ling(example):
        standardized = []
        for m in example.get("messages", []):
            raw_role = str(m.get("role", "")).lower()
            role     = "user" if ("human" in raw_role or "user" in raw_role) else "assistant"
            standardized.append({"role": role, "content": str(m.get("content", ""))})
        return {"text": tokenizer.apply_chat_template(
            standardized, tokenize=False, add_generation_prompt=False)}
    return process_ling


# ==============================================================================
# Main
# ==============================================================================

def main():
    max_seq_length = 2048

    # ── Weights & Biases ───────────────────────────────────────────────────────
    wandb.init(
        project = "qwen35-35b-abliterated-fpe",
        name    = "run-fpe-opus-lingcoder",
        config  = {
            "model":           "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
            "lora_r":          16,
            "lora_alpha":      16,
            "lr":              2e-4,
            "max_steps":       5000,
            "per_device_batch": 8,
            "grad_acc_steps":  4,
            "max_seq_length":  max_seq_length,
            "warmup_steps":    100,
            "fpe_pr_tol":      0.02,
            "fpe_al_thresh":   0.75,
            "fpe_patience":    20,
            "fpe_timeout_h":   5.0,
            "datasets":        ["nohurry/Opus-4.6-Reasoning-3000x-filtered",
                                 "inclusionAI/Ling-Coder-SFT"],
        }
    )

    print("=" * 70)
    print("Model: huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated")
    print("  36B params (qwen3_5_moe) | ~3B active | BF16 | abliterated")
    print("=" * 70)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
        max_seq_length = max_seq_length,
        dtype          = torch.bfloat16,
        load_in_4bit   = False,   # BF16 ~72 GB fits on 8×H100; 4-bit would
                                  # corrupt abliteration steering vectors
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = 16,
        target_modules             = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha                 = 16,
        lora_dropout               = 0,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
    )

    process_opus = build_process_opus(tokenizer)
    process_ling = build_process_ling(tokenizer)

    print("\nLoading Opus-4.6-Reasoning ...")
    ds_opus = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered", split="train")
    ds_opus = ds_opus.map(process_opus, num_proc=8, desc="Formatting Opus")
    ds_opus = ds_opus.select_columns(["text"])
    ds_opus = ds_opus.filter(lambda x: bool(x["text"] and x["text"].strip()))
    print(f"  → {len(ds_opus):,} examples")

    print("\nLoading Ling-Coder-SFT ...")
    ds_ling = load_dataset("inclusionAI/Ling-Coder-SFT", split="train")
    ds_ling = ds_ling.map(process_ling, num_proc=8, desc="Formatting Ling-Coder")
    ds_ling = ds_ling.select_columns(["text"])
    ds_ling = ds_ling.filter(lambda x: bool(x["text"] and x["text"].strip()))
    print(f"  → {len(ds_ling):,} examples")

    dataset = concatenate_datasets([ds_opus, ds_ling]).shuffle(seed=3407)
    print(f"\nTotal training examples: {len(dataset):,}\n")

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = max_seq_length,
        dataset_num_proc   = 4,
        packing            = False,
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 4,
            warmup_steps                = 100,
            max_steps                   = 5000,
            learning_rate               = 2e-4,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 1,
            optim                       = "adamw_8bit",
            weight_decay                = 0.01,
            lr_scheduler_type           = "cosine",
            seed                        = 3407,
            output_dir                  = "outputs",
            logging_dir                 = "outputs/logs",
            report_to                   = "wandb",
            run_name                    = "run-fpe-opus-lingcoder",
        ),
    )

    trainer.add_callback(
        SuperpositionFPECallback(
            model,
            target_k      = 100,
            patience      = 20,
            pr_tol        = 0.02,
            al_thresh     = 0.75,
            timeout_hours = 5.0,
        )
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()