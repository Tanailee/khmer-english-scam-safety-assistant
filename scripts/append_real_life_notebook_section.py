from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Scam_or_Not_Scam_IR_Project.ipynb"


SECTION_TITLE = "## Real-Life Scam Safety Assistant Upgrade"


def main() -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    if any(cell.cell_type == "markdown" and SECTION_TITLE in cell.source for cell in nb.cells):
        print("Notebook already contains the real-life upgrade section.")
        return

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            SECTION_TITLE
            + """

The project was upgraded from a simple `Safe` / `Scam` classifier into a Khmer-English scam-prevention assistant for real messages, emails, chats, and screenshots.

Key real-life improvements:

- Hybrid ML + rule-based decision making.
- Critical overrides for OTP, password, PIN, CVV, seed phrase, account number, urgent verification, suspicious links, and payment/deposit pressure.
- Scam type classification: OTP theft, fake bank support, delivery fee, prize/reward, loan, job, gambling/deposit, investment/crypto, email phishing, and account-locked verification scams.
- Khmer-English OCR support for SMS, Telegram/Facebook, and email screenshots.
- OCR cleanup before prediction to remove phone UI headers, dates, duplicated lines, and noisy symbols.
- User-facing advice: why suspicious, what not to share, how to verify, and how to report/block.
- Feedback capture for future active learning.
"""
        )
    )
    nb.cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json

root = Path.cwd()
metrics_path = root / "reports" / "realistic_model_metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

print("Total realistic samples:", metrics.get("n_samples_total"))
print("Augmented examples:", metrics.get("n_augmented"))
print("Real-life upgrade focus:")
for item in metrics.get("real_life_upgrade", {}).get("focus", []):
    print("-", item)
"""
        )
    )
    nb.cells.append(
        nbf.v4.new_code_cell(
            """# Demo cases for the deployed safety assistant
demo_messages = [
    "Give me your account password",
    "Your ABA account is locked. Verify now and enter your OTP.",
    "Casino bonus: 60% first deposit and 8.8 USD app bonus. Send ABA deposit screenshot.",
    "Never share your OTP, password, PIN, CVV, or bank account number with anyone.",
    "Official public notice: heavy rain is expected in several provinces. Stay safe.",
]

pd.DataFrame({"demo_message": demo_messages})
"""
        )
    )
    nbf.write(nb, NOTEBOOK)
    print("Notebook real-life upgrade section appended.")


if __name__ == "__main__":
    main()
