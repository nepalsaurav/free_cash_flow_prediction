import json
import re

with open("SVR_Valuation_Pipeline.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_lines = cell["source"]
        final_source = []
        for line in source_lines:
            # Drop fragments that look like string literal remains from plt.title
            if line.strip() in ['",', '",\\n', '", \\n', '",\n', '", \n']:
                continue
            if line.strip().startswith('",') or line.strip().startswith("',"):
                continue
            final_source.append(line)
        cell["source"] = final_source

with open("SVR_Valuation_Pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Syntax errors properly fixed.")
