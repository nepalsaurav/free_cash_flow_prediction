import json
import re

with open("SVR_Valuation_Pipeline.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. Vibrant Theme
        source = re.sub(r'sns\.set_theme\(style=["\']ticks["\']\)', 'sns.set_theme(style="whitegrid", palette="deep")', source)
        
        # 2. Change Y-axis from Billions to Millions
        source = source.replace("ax.set_ylabel('Billions (Rs)', fontsize=12)", "ax.set_ylabel('Millions (Rs)', fontsize=12, fontweight='bold')")
        source = source.replace("top_5['Trad_Eq'] / 1e9", "top_5['Trad_Eq'] / 1e6")
        source = source.replace("top_5['SVR_Eq'] / 1e9", "top_5['SVR_Eq'] / 1e6")
        source = source.replace("top_5['Actual_Cap'] / 1e9", "top_5['Actual_Cap'] / 1e6")
        
        # 3. Use vibrant colors for the valuation plot
        source = source.replace("color='#cccccc'", "color='#FF9F43'") # Vibrant orange
        source = source.replace("color='#2ca02c'", "color='#00D2D3'") # Vibrant cyan
        
        # For the fix I made earlier changing #1f77b4 to #2f4f4f, let's change back to vibrant
        source = source.replace('#2f4f4f', '#1f77b4') # Blue
        source = source.replace('#36454F', '#ff7f0e') # Orange
        source = source.replace('#008080', '#2ca02c') # Green
        
        # If there are manual grid lines that look bad, we remove them since we use whitegrid
        # (Though we added them in previous script, we can just leave them or let whitegrid handle it)
        
        cell["source"] = [s + "\n" for s in source.split("\n")][:-1]

with open("SVR_Valuation_Pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook aesthetics updated.")
