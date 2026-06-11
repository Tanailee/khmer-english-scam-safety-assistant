from pathlib import Path
import ast

import nbformat


NOTEBOOK = Path(__file__).resolve().parent / "Scam_or_Not_Scam_IR_Project.ipynb"
MARKER = "Defense-Ready Evaluation, Ablation Study, and IR Analysis"

nb = nbformat.read(NOTEBOOK, as_version=4)

inside_defense_section = False
fixed = 0
for cell in nb.cells:
    source = cell.get("source", "")
    if MARKER in source:
        inside_defense_section = True
    if inside_defense_section and cell.cell_type == "code":
        lines = source.splitlines()
        if len(lines) > 1 and lines[0] and not lines[0].startswith(" ") and lines[1].startswith("        "):
            candidate = "\n".join([lines[0], *[(line[8:] if line.startswith("        ") else line) for line in lines[1:]]]).strip()
        else:
            candidate = source.strip()
        try:
            ast.parse(candidate)
        except SyntaxError:
            candidate = source.strip()
        if candidate != source:
            cell["source"] = candidate
            fixed += 1

nbformat.write(nb, NOTEBOOK)
print(f"Fixed indentation in {fixed} defense-section code cells.")
