import json

path = '/home/ubuntu/superposition-metrics/colab/progressive_quantization_matrix.ipynb'
out_path = '/home/ubuntu/superposition-metrics/colab/exact_resume.ipynb'

try:
    with open(path, 'r') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # The Toy Models Matrix has loops starting with:
            # k = f"{q}_{r['name']}"
            if "k = f\"{q}_{r['name']}\"" in source and "results[k] = run_experiment(" in source:
                skip_toy_code = """k = f"{q}_{r['name']}"
            print(f"⏩ Skipping Part 1 / Part 3 Toy Model Experiment: {k}")
            continue
"""
                new_source = source.replace("k = f\"{q}_{r['name']}\"", skip_toy_code, 1)
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source'][-1] == '\n':
                    cell['source'].pop()
                    
            # The Transformer Matrix (Part 2 and Part 4) has loops starting with:
            # k = f"L3_{l3c}_{q}_{r['name']}"
            if "k = f\"L3_{l3c}_{q}_{r['name']}\"" in source and "results[k] = run_transformer_experiment(" in source:
                skip_transformer_code = """k = f"L3_{l3c}_{q}_{r['name']}"
                
                # We specifically only want to skip L3=0, W8A16, for ALL regimes BEFORE Fixed_FP32
                if l3c == 0 and q == 'W8A16' and r['name'] in ['Ageing', 'Fixed', 'Fixed_FP32', 'Ageing_FP32']:
                    print(f"⏩ Fast-forwarding past completed transformer experiment {k}...")
                    continue
"""
                new_source = source.replace("k = f\"L3_{l3c}_{q}_{r['name']}\"", skip_transformer_code, 1)
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source'][-1] == '\n':
                    cell['source'].pop()

    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"\n✅ Created EXACT resume notebook at {out_path}!")
except Exception as e:
    print(f"Error process notebook: {e}")
