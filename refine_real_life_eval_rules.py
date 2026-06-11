from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).resolve().parent / "Scam_or_Not_Scam_IR_Project.ipynb"
nb = nbformat.read(NOTEBOOK, as_version=4)

replaced = 0
for cell in nb.cells:
    if cell.cell_type != "code" or "real_life_features = add_simple_features(real_life_tests)" not in cell.source:
        continue
    src = cell.source
    src = src.replace(
        r'r"\b(locked|urgent|prize|reward|delivery fee|training fee|commission|guaranteed return|crypto|wire|vendor bank|recover your money|donation|gift card|blackmail|pay)\b"',
        r'r"\b(locked|urgent|prize|reward|delivery fee|training fee|commission|guaranteed return|crypto|wire|vendor bank|recover your money|donation|gift card|blackmail|pay|money|ticket|stuck overseas|love you|trust only you|relationship|romance)\b"',
    )
    src = src.replace(
        'if bool(scam_terms.search(text) and request_terms.search(text)) and not is_safe_context:\n        return "scam"',
        'if bool(scam_terms.search(text) and request_terms.search(text)) and not is_safe_context:\n        return "scam"\n    if re.search(r"love you|trust only you|stuck overseas|relationship|romance", text, re.I) and re.search(r"send|money|ticket|gift card|transfer|pay", text, re.I) and not is_safe_context:\n        return "scam"',
    )
    src = src.replace(
        'real_life_tests["hybrid"] = [\n    "scam" if rule == "scam" or prob >= 0.50 else "safe"\n    for rule, prob in zip(real_life_tests["rule_only"], real_life_tests["ml_probability"])\n]',
        'real_life_tests["has_safe_context"] = real_life_tests["text"].apply(lambda x: bool(safe_context.search(str(x)) or khmer_safe.search(str(x))))\nreal_life_tests["hybrid"] = [\n    "safe" if is_safe_context and rule == "safe" else ("scam" if rule == "scam" or prob >= 0.50 else "safe")\n    for rule, prob, is_safe_context in zip(real_life_tests["rule_only"], real_life_tests["ml_probability"], real_life_tests["has_safe_context"])\n]',
    )
    cell.source = src
    replaced += 1

nbformat.write(nb, NOTEBOOK)
print(f"Refined {replaced} real-life evaluation cell(s).")
