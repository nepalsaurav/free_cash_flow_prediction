import json
import re

with open("SVR_Valuation_Pipeline.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. Theme
        source = re.sub(r'sns\.set_theme\(.*?style="whitegrid".*?\)', 'sns.set_theme(style="ticks")', source)
        
        # 2. Colors
        source = source.replace('#1f77b4', '#2f4f4f') # Deep slate blue
        source = source.replace('#ff7f0e', '#36454F') # Charcoal
        source = source.replace('#2ca02c', '#008080') # Dark teal
        
        # 3. Titles
        source = re.sub(r'plt\.title\(.*?\)\n?', '', source)
        source = re.sub(r'ax\.set_title\(.*?\)\n?', '', source)
        source = re.sub(r'fig\.suptitle\(.*?\)\n?', '', source)
        
        # 4. Axes fontsizes
        # For standard plt.xlabel and ylabel
        source = re.sub(r'(plt\.xlabel\(["\'][^"\']+["\'])(\))', r'\1, fontsize=12\2', source)
        source = re.sub(r'(plt\.ylabel\(["\'][^"\']+["\'])(\))', r'\1, fontsize=12\2', source)
        source = re.sub(r'(ax\.set_ylabel\(["\'][^"\']+["\'])(\))', r'\1, fontsize=12\2', source)
        
        # Replace existing fontsizes in labels if they already have them (from our previous code)
        source = re.sub(r'(plt\.xlabel\(.*?)fontsize=\d+(.*?\))', r'\g<1>fontsize=12\g<2>', source)
        source = re.sub(r'(plt\.ylabel\(.*?)fontsize=\d+(.*?\))', r'\g<1>fontsize=12\g<2>', source)
        source = re.sub(r'(ax\.set_ylabel\(.*?)fontsize=\d+(.*?\))', r'\g<1>fontsize=12\g<2>', source)
        
        # 5. Add despine, grid, tick params before savefig
        # Find plt.savefig and insert the styling lines right before it
        lines = source.split('\n')
        new_lines = []
        for line in lines:
            if 'plt.savefig(' in line:
                # Need to add styling block
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + "sns.despine()")
                new_lines.append(indent + "plt.tick_params(labelsize=11)")
                # If we have `ax`, we can do ax.yaxis.grid, but to be safe we use plt.gca()
                new_lines.append(indent + "plt.gca().yaxis.grid(True, linestyle=\"--\", alpha=0.4)")
                new_lines.append(indent + "plt.gca().xaxis.grid(False)")
            new_lines.append(line)
            
        # Rejoin and update
        source = "\n".join(new_lines)
        
        # Put back into cell
        cell["source"] = [s + "\n" for s in source.split("\n")][:-1]

with open("SVR_Valuation_Pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook plots refactored successfully.")
