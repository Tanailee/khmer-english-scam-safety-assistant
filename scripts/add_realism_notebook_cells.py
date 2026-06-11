from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
NB = ROOT / "Scam_or_Not_Scam_IR_Project.ipynb"


def cell(kind, source):
    data = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        data.update({"execution_count": None, "outputs": []})
    return data


nb = json.loads(NB.read_text(encoding="utf-8"))
text = "".join("".join(c.get("source", [])) for c in nb["cells"])

if "Real-Life Robustness Upgrade" not in text:
    nb["cells"].extend(
        [
            cell(
                "markdown",
                "## Real-Life Robustness Upgrade\n\nThe original dataset gives perfect accuracy, but that does not mean the model is ready for real users. A realistic scam detector must catch short credential-theft messages such as `Give me your account password`, and it must avoid false alarms for safety education such as `Never give anyone your password`.",
            ),
            cell(
                "markdown",
                "### Realistic Augmentation Strategy\n\nTo make the model more applicable, we add a small but targeted realistic set:\n\n- Credential-theft scams: password, OTP, PIN, account number, CVV.\n- Account-lock and urgent verification scams.\n- Delivery-fee, prize, refund, and loan-fee scams.\n- Hard safe examples that mention risky terms in protective contexts.\n- Character n-gram features to handle typos such as `you account password`.",
            ),
            cell(
                "code",
                "from pathlib import Path\nimport pandas as pd\nrealistic_metrics_path = Path('reports/realistic_model_metrics.json')\nrealistic_demo_path = Path('reports/realistic_demo_predictions.csv')\nif realistic_demo_path.exists():\n    realistic_demo = pd.read_csv(realistic_demo_path)\n    display(realistic_demo[['text', 'ml_probability', 'rule_score', 'hybrid_probability', 'prediction']])\nelse:\n    print('Run make_realistic_model.py first to generate realistic demo predictions.')",
            ),
            cell(
                "markdown",
                "### Why Hybrid ML + Rules?\n\nMachine learning learns patterns from the corpus, but safety-critical scam detection also needs explicit rules for high-risk events. Asking for passwords, OTPs, PINs, account numbers, seed phrases, or CVV codes should be flagged even if the message is short. The deployed app therefore uses the maximum of the model probability and a risk-rule score, while reducing risk for educational/protective contexts.",
            ),
            cell(
                "markdown",
                "### Realistic Limitation\n\nThe original test set can still show `1.0` accuracy because the supplied corpus is clean. The realistic improvement is evaluated with targeted demo cases and should be expanded with more real public scam examples, multilingual Khmer-English data, and ongoing feedback from users.",
            ),
        ]
    )

NB.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Notebook realism cells added.")
