import json
import re

with open("SVR_Valuation_Pipeline.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_lines = cell["source"]
        new_source = []
        skip_next = False
        for i, line in enumerate(source_lines):
            if skip_next:
                skip_next = False
                continue
            
            # Clean up the broken plt.title fragments
            if re.search(r'^\s*["\'].*?fontsize=14', line):
                continue
            if re.search(r'^\s*fontsize=14.*?pad=15', line):
                continue
            if 'fig.suptitle(' in line:
                # the previous script might have broken fig.suptitle too. Let's see if it left fragments.
                pass
            if re.search(r'^fig\.suptitle\(.*?\)\n?', line):
                continue
            if re.search(r'^\s*["\'].*?fontsize=16\)', line): # cleanup suptitle fragment if any
                continue
            
            new_source.append(line)
        
        # fix suptitle fragment
        final_source = []
        for line in new_source:
            if '", fontsize=16)' in line or '", fontsize=16' in line:
                continue
            final_source.append(line)
            
        cell["source"] = final_source

with open("SVR_Valuation_Pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Fixed syntax errors.")
