from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT = DATA_DIR / "real_world_message_dataset.csv"
META = DATA_DIR / "real_world_dataset_sources.json"


SOURCES = [
    {
        "name": "UCI SMS Spam Collection",
        "type": "public_dataset",
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "note": "Public ham/spam SMS corpus by Almeida, Hidalgo, and Yamakami.",
    },
    {
        "name": "FTC / public consumer warnings",
        "type": "public_report_patterns",
        "url": "https://consumer.ftc.gov/",
        "note": "Used only to create paraphrased scam-pattern examples: tolls, delivery, job/task scams, prizes, account warnings.",
    },
    {
        "name": "USPS / postal inspection smishing warnings",
        "type": "public_report_patterns",
        "url": "https://www.uspis.gov/",
        "note": "Used only to create paraphrased delivery-smishing examples and safe advice examples.",
    },
    {
        "name": "Smishing Dataset I: Phishing SMS Dataset from Smishtank.com",
        "type": "research_reference",
        "url": "https://arxiv.org/abs/2402.18430",
        "note": "Research reference for smishing data categories. Direct bulk download was not assumed.",
    },
    {
        "name": "Public Khmer/manual examples",
        "type": "manual_and_synthetic",
        "url": "local",
        "note": "Manually authored Khmer/English examples for Cambodian banking, delivery, OTP, prize, loan, job, and safe education contexts.",
    },
]


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def try_download_uci() -> list[dict]:
    rows: list[dict] = []
    url = SOURCES[0]["url"]
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            raw = zf.read("SMSSpamCollection").decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(raw), delimiter="\t")
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                continue
            original_label, text = row[0], row[1]
            label = "safe" if original_label == "ham" else "scam"
            rows.append(
                {
                    "id": f"uci_sms_{i:05d}",
                    "label": label,
                    "text": clean_text(text),
                    "scam_type": "spam_or_unspecified" if label == "scam" else "safe_personal",
                    "language": "en",
                    "source": "UCI SMS Spam Collection",
                    "collection_method": "downloaded_public_dataset",
                }
            )
    except Exception as exc:
        SOURCES[0]["download_error"] = str(exc)
    return rows


PUBLIC_REPORT_STYLE_SCAMS = [
    ("Delivery fee scam", "Your parcel cannot be delivered because a small customs fee is unpaid. Pay here: http://parcel-help.example"),
    ("Delivery fee scam", "Package delivery failed due to incomplete address. Update your card and address at delivery-track.example"),
    ("Toll / traffic fee scam", "Final notice: unpaid road toll balance. Pay immediately to avoid penalty: http://toll-pay.example"),
    ("Toll / traffic fee scam", "You have an outstanding parking violation. Submit payment today or additional fees apply."),
    ("Fake bank support", "Security alert: your bank account is locked. Verify your login and OTP now."),
    ("Fake bank support", "Bank support detected unusual activity. Reply with your account number to restore access."),
    ("Phone provider scam", "Your phone reward points expire today. Login with your account password to claim your gift."),
    ("Phone provider scam", "We need to confirm your mobile account. Send the code we just sent you."),
    ("Job scam", "Work from home and earn $300 per day. Pay the registration fee to start your first task."),
    ("Job scam", "Congratulations, you are selected for an online job. Deposit money to unlock higher salary tasks."),
    ("Prize or reward scam", "You won a cash reward. Provide your bank details to receive payment today."),
    ("Prize or reward scam", "Congratulations! Claim your bonus by entering your card number and CVV."),
    ("Loan scam", "Your loan is approved instantly. Pay the processing fee before release."),
    ("Loan scam", "Fast cash loan available with no checks. Send ID and bank account number now."),
    ("Romance or trust scam", "I trust you. Please send money urgently; I promise to pay you back tomorrow."),
    ("Investment scam", "Double your money in 24 hours. Transfer funds now to join the private investment group."),
    ("Account locked / urgent verification scam", "Your email account will expire today. Confirm password to keep access."),
    ("Account locked / urgent verification scam", "Final warning: your wallet is suspended. Enter seed phrase to recover it."),
]


KHMER_SCAMS = [
    ("Password / OTP theft", "សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នកដើម្បីបញ្ជាក់"),
    ("Password / OTP theft", "សូមបញ្ចូលលេខគណនី និងពាក្យសម្ងាត់ ដើម្បីទទួលរង្វាន់"),
    ("Fake bank support", "គណនី ABA របស់អ្នកត្រូវបានចាក់សោ សូមផ្ញើ OTP ឥឡូវនេះ"),
    ("Fake bank support", "ធនាគារត្រូវការបញ្ជាក់ សូមផ្តល់លេខកូដ OTP និងលេខគណនី"),
    ("Prize or reward scam", "អ្នកឈ្នះរង្វាន់ធំ សូមផ្តល់លេខគណនីធនាគារដើម្បីទទួលប្រាក់"),
    ("Delivery fee scam", "បន្ទាន់ សូមបង់ថ្លៃដឹកជញ្ជូនតាមតំណនេះ"),
    ("Loan scam", "កម្ចីរបស់អ្នកបានអនុម័ត សូមផ្ទេរប្រាក់ថ្លៃសេវាមុន"),
    ("Job scam", "ការងារតាមអនឡាញ ប្រាក់ខែខ្ពស់ សូមបង់ថ្លៃចុះឈ្មោះ"),
    ("Account locked / urgent verification scam", "គណនីរបស់អ្នកនឹងបិទថ្ងៃនេះ សូមបញ្ចូលលេខសម្ងាត់"),
    ("Unknown suspicious message", "សូមចុចតំណនេះដើម្បីទទួលប្រាក់រង្វាន់ឥឡូវនេះ"),
]


SYNTHETIC_SCAMS = []
brands = ["ABA", "ACLEDA", "Wing", "Bakong", "TrueMoney", "Facebook", "Telegram"]
actions = [
    "send your OTP",
    "enter your password",
    "provide your account number",
    "confirm your card number",
    "pay a small verification fee",
]
reasons = ["account locked", "reward pending", "delivery failed", "security update", "loan approved"]
for brand in brands:
    for reason in reasons:
        for action in actions:
            SYNTHETIC_SCAMS.append(("Synthetic scam", f"{brand} notice: {reason}. Please {action} now."))


SAFE_EDUCATIONAL = [
    "Never share your password, OTP, PIN, CVV, or account number with anyone.",
    "A real bank will not ask for your password or OTP by SMS.",
    "If a delivery text asks for payment, check the courier app directly.",
    "Do not click links from unknown senders.",
    "Report suspicious messages and delete them after saving evidence.",
    "Use the official banking app to verify account warnings.",
    "Do not pay a job registration fee before verifying the company.",
    "Do not send money to someone you only know online.",
    "កុំផ្ញើលេខសម្ងាត់ ឬ OTP ទៅអ្នកណាម្នាក់។",
    "ធនាគារពិតប្រាកដមិនសួរពាក្យសម្ងាត់តាមសារទេ។",
    "បើសារដឹកជញ្ជូនសួរបង់ប្រាក់ សូមពិនិត្យតាមកម្មវិធីផ្លូវការ។",
    "កុំចុចតំណពីអ្នកផ្ញើមិនស្គាល់។",
    "សូមរាយការណ៍សារសង្ស័យ ហើយកុំផ្ញើព័ត៌មានផ្ទាល់ខ្លួន។",
    "យើងរៀនអំពីសុវត្ថិភាពគណនី និងការការពារពាក្យសម្ងាត់។",
]


def generated_rows() -> list[dict]:
    rows: list[dict] = []
    for i, (stype, text) in enumerate(PUBLIC_REPORT_STYLE_SCAMS, 1):
        rows.append(
            {
                "id": f"public_pattern_scam_{i:04d}",
                "label": "scam",
                "text": clean_text(text),
                "scam_type": stype,
                "language": "en",
                "source": "Public report pattern synthesis",
                "collection_method": "paraphrased_from_public_warning_categories",
            }
        )
    for i, (stype, text) in enumerate(KHMER_SCAMS, 1):
        rows.append(
            {
                "id": f"khmer_manual_scam_{i:04d}",
                "label": "scam",
                "text": clean_text(text),
                "scam_type": stype,
                "language": "km",
                "source": "Manual Khmer scam examples",
                "collection_method": "manual_safety_dataset_authoring",
            }
        )
    for i, (stype, text) in enumerate(SYNTHETIC_SCAMS, 1):
        rows.append(
            {
                "id": f"synthetic_scam_{i:04d}",
                "label": "scam",
                "text": clean_text(text),
                "scam_type": stype,
                "language": "en",
                "source": "Synthetic realistic scam templates",
                "collection_method": "controlled_template_generation",
            }
        )
    for i, text in enumerate(SAFE_EDUCATIONAL, 1):
        rows.append(
            {
                "id": f"safe_education_{i:04d}",
                "label": "safe",
                "text": clean_text(text),
                "scam_type": "safe_education",
                "language": "km" if any("\u1780" <= ch <= "\u17FF" for ch in text) else "en",
                "source": "Manual safe educational contrast examples",
                "collection_method": "manual_safety_dataset_authoring",
            }
        )
    return rows


def main() -> None:
    rows = []
    rows.extend(try_download_uci())
    rows.extend(generated_rows())
    df = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    summary = {
        "rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "language_counts": df["language"].value_counts().to_dict(),
        "source_counts": df["source"].value_counts().to_dict(),
        "sources": SOURCES,
        "ethics": "Only public datasets and manually/synthetically authored non-private messages are used. Do not collect private user SMS without consent.",
    }
    META.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
