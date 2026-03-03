import json
import os

path = '/home/ubuntu/superposition-metrics/colab/progressive_quantization_matrix.ipynb'
out_paths = {
    'part3': '/home/ubuntu/superposition-metrics/colab/node_part3_senn_toy.ipynb',
    'part4': '/home/ubuntu/superposition-metrics/colab/node_part4_senn_transformer.ipynb',
    'part6': '/home/ubuntu/superposition-metrics/colab/node_part6_alignment.ipynb'
}

try:
    with open(path, 'r') as f:
        nb = json.load(f)
        
    core_imports = []
    part3_cells = []
    part4_cells = []
    part6_cells = []
    
    current_part = 0
    
    for cell in nb['cells']:
        source = "".join(cell.get('source', []))
        
        # Capture Core Imports (Part 0)
        if "pip install" in source:
             core_imports.append(cell)
             continue
             
        # Detect partitions
        if "Part 3:" in source: current_part = 3
        elif "Part 4:" in source: current_part = 4
        elif "Part 5:" in source: current_part = 5 # QTIP / YAQA (User wants this DELETED)
        elif "Part 6:" in source: current_part = 6
        
        # Fast exit unwanted blocks (Part 1, Part 2, Part 5)
        if current_part in [1, 2, 5]:
            continue
            
        # --- GLOBAL TOLERANCE SWEEP ---
        # Lower expansion tolerance from 0.1 to 0.05
        if "--tolerance" in source:
             source = source.replace('default=0.10', 'default=0.05').replace('default=0.1)', 'default=0.05)')
             source = source.replace('default=0.5', 'default=0.05')
             cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.split('\n')]
            
        # Distribute cells to nodes
        if current_part == 3: part3_cells.append(cell)
        elif current_part == 4: part4_cells.append(cell)
        elif current_part == 6: part6_cells.append(cell)
        
    # Inject Live Logger to imports
    logger_cell = json.loads(json.dumps(core_imports[0])) # deep copy
    logger_source = "import sys\nclass _DL:\n def __init__(self, f):\n  self.t=sys.stdout; self.l=open(f, 'a')\n def write(self, m):\n  self.t.write(m); self.l.write(m); self.l.flush()\n def flush(self):\n  self.t.flush(); self.l.flush()\nsys.stdout = _DL('live_output.log')\n"
    logger_cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in (logger_source + "".join(logger_cell['source'])).split('\n')]
    
    for pt, c in [('part3', part3_cells), ('part4', part4_cells), ('part6', part6_cells)]:
        nb_copy = json.loads(json.dumps(nb))
        nb_copy['cells'] = [logger_cell] + c
        with open(out_paths[pt], 'w') as f:
            json.dump(nb_copy, f, indent=2)
            
    print(f"\n✅ Successfully split notebook. Part 5 (QTIP) has been completely PURGED!")
except Exception as e:
    print(f"Error process notebook: {e}")
