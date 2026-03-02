import json
import sys

# Replace with the path to the executed notebook
path = '/root/superposition-metrics/colab/executed_matrix.ipynb'
# Or if you used the other name:
# path = '/root/superposition-metrics/executed_matrix.ipynb'
# or check whatever it's called locally

try:
    with open(path, 'r') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                # nbconvert stores standard print output as "stream" data
                if output.get('output_type') == 'stream':
                    text = "".join(output.get('text', []))
                    if 'Step' in text or 'Detonated' in text or 'E-Rnk' in text:
                        print(text, end="")
except FileNotFoundError:
    print(f"Could not find the executed notebook at {path}. Please adjust the path in the script!")
except Exception as e:
    print(f"Error parsing notebook: {e}")
