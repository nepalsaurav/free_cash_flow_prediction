import json
import re

with open("SVR_Valuation_Pipeline.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. DPI Upgrade to 600
        source = re.sub(r'dpi=\d+', 'dpi=600', source)
        
        # 2. Global theme: Vibrant and classy
        source = re.sub(r'sns\.set_theme\(style=.*?\)', 'sns.set_theme(style="whitegrid", palette="husl")', source)
        
        # 3. Specific vibrant, classy hex codes
        # Replace the hardcoded colors I used previously with extremely vibrant, HD, classy colors
        # #1f77b4 / #2b5b84 -> Vibrant Azure Blue (#007BFF or #4A90E2)
        source = source.replace('#1f77b4', '#4A90E2')
        source = source.replace('#2b5b84', '#4A90E2')
        
        # #ff7f0e / #FF9F43 -> Vibrant Coral/Orange (#FF5A5F)
        source = source.replace('#ff7f0e', '#FF5A5F')
        source = source.replace('#FF9F43', '#FF5A5F')
        
        # #2ca02c / #00D2D3 -> Vibrant Emerald Green (#00B894)
        source = source.replace('#2ca02c', '#00B894')
        source = source.replace('#00D2D3', '#00B894')
        
        cell["source"] = [s + "\n" for s in source.split("\n")][:-1]

with open("SVR_Valuation_Pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Aesthetics upgraded to 600 DPI and vibrant classy colors.")
