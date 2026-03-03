import json
import os

path = '/home/ubuntu/superposition-metrics/colab/progressive_quantization_matrix.ipynb'
out_path = '/home/ubuntu/superposition-metrics/colab/parallel_matrix.ipynb'

try:
    with open(path, 'r') as f:
        nb = json.load(f)
        
    filtered_cells = []
    
    # We always need Part 0 (Imports and setup, usually first 2 cells)
    for cell in nb['cells']:
        source = "".join(cell.get('source', []))
        if "pip install" in source:
            # We want to keep the imports, but let's inject our live logger too
            logger = "import sys\nclass _DL:\n def __init__(self, f):\n  self.t=sys.stdout; self.l=open(f, 'a')\n def write(self, m):\n  self.t.write(m); self.l.write(m); self.l.flush()\n def flush(self):\n  self.t.flush(); self.l.flush()\nsys.stdout = _DL('parallel_live_output.log')\n"
            cell['source'] = [line + ('\\n' if not line.endswith('\\n') else '') for line in (logger + source).split('\\n')]
            filtered_cells.append(cell)
            continue
            
        # Skip the Toy Model Matrix Execution (Part 1) entirely
        if "Part 1" in source or ("ProgressiveToyModel" in source and "toy_model" in source and "toy_senn" not in source):
             # Keep the markdown cell but we will drop the code cell
             if cell['cell_type'] == 'code': 
                  continue
                  
        # Skip the Transformer Matrix Execution (Part 2) entirely
        if "Part 2" in source or ("run_transformer_experiment" in source and "toy_senn" not in source and "senn" not in source):
             # We actually need the Transformer MODEL DEFINITIONS from Part 2!
             # But we don't want to run the MAIN function if it's the 10-exp loop.
             if cell['cell_type'] == 'code' and "main()" in source and "run_transformer_experiment" in source:
                  # We intercept it by nuking main
                  source = source.replace("main()\\n", "# main() killed for parallel split\\n")
                  cell['source'] = [line + ('\\n' if not line.endswith('\\n') else '') for line in source.split('\\n')]
             
        # Keep everything else (Part 3 SENN Toy, Part 4 SENN Transformer, Part 5 Quant L3, Part 6 Deng Alignment)
        filtered_cells.append(cell)

    nb['cells'] = filtered_cells

    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"\\n✅ Extracted Parts 3, 4, 5, and 6 cleanly into {out_path}!")
except Exception as e:
    print(f"Error process notebook: {e}")
