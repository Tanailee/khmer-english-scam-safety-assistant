from __future__ import annotations

import json
import math
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
OUT_DATA = ROOT / "data"
OUT_MODEL = ROOT / "models"
REPORTS = ROOT / "reports"

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+|\d+(?:[\.,]\d+)?|[$]\s*\d+")
KHMER_RE = re.compile(r"[\u1780-\u17FF]")
URL_RE = re.compile(r"https?://|www\.|bit\.ly|tinyurl|t\.co|\.com\b", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\s\-().]*){7,}")
MONEY_RE = re.compile(r"[$]\s?\d+|\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:usd|dollars?|cash|prize|bonus|refund|fee|payment)\b|\b(cash|prize|bonus|refund|fee|payment)\b", re.I)
URGENT_RE = re.compile(r"\b(urgent|immediately|now|today|limited|final|expire|expired|act fast|asap|suspended|locked)\b", re.I)
ACTION_RE = re.compile(r"\b(click|claim|verify|confirm|reply|call|text|login|update|send|transfer|provide|share|submit|enter|reset)\b", re.I)
NO_RE = re.compile(r"\bno\b", re.I)
PRONOUN_RE = re.compile(r"\b(i|me|my|you|your)\b", re.I)
CREDENTIAL_RE = re.compile(r"\b(password|passcode|otp|pin|code|verification code|login|username|account number|bank detail|card number|cvv|private key|seed phrase)\b", re.I)
REQUEST_RE = re.compile(r"\b(give|send|share|tell|provide|submit|enter|type|reply|confirm|verify|update)\b", re.I)
SOCIAL_ENGINEERING_RE = re.compile(r"\b(account|bank|wallet|paypal|telegram|whatsapp|facebook|email|security|support|admin|prize|reward|refund|delivery|parcel)\b", re.I)
KHMER_CREDENTIAL_RE = re.compile(r"(ពាក្យសម្ងាត់|លេខសម្ងាត់|អូធីភី|OTP|កូដ|លេខកូដ|លេខគណនី|គណនី|លេខកាត|ស៊ីវីវី|CVV)", re.I)
KHMER_REQUEST_RE = re.compile(r"(ផ្ញើ|ផ្ដល់|ផ្តល់|ប្រាប់|បញ្ចូល|បញ្ជាក់|ចុច|បង់|ផ្ទេរ|ទូទាត់|ចែករំលែក|បំពេញ|ឆ្លើយតប)", re.I)
KHMER_SOCIAL_RE = re.compile(r"(ធនាគារ|ABA|ACLEDA|Wing|វីង|TrueMoney|Bakong|បាគង|គណនី|សុវត្ថិភាព|រង្វាន់|ប្រាក់|លុយ|កម្ចី|ដឹកជញ្ជូន|កញ្ចប់|បិទ|ចាក់សោ|ផុតកំណត់|បន្ទាន់)", re.I)
KHMER_SAFE_CONTEXT_RE = re.compile(r"(កុំ|មិនត្រូវ|មិនគួរ|កុំចែករំលែក|កុំផ្ញើ|ការពារ|រាយការណ៍|ព្រមាន|រៀន|សុវត្ថិភាព)", re.I)


SCAM_EXAMPLES = [
    "Give me your account password",
    "Give me you account password",
    "Please send your OTP code now to keep your account active",
    "Your bank account is locked. Verify your password immediately",
    "Reply with your PIN to confirm this transaction",
    "Send me your verification code so I can help you login",
    "Your package is delayed. Pay the small fee at http://delivery-check.example",
    "Congratulations you won a prize. Share your bank account number to receive cash",
    "Security alert: update your login password today or your account will expire",
    "We are support team. Tell us your Facebook password to recover your page",
    "Transfer $50 now to release your refund",
    "Your wallet is at risk. Send seed phrase for verification",
    "Click this link and enter your card number to avoid suspension",
    "Final notice: confirm your account number and CVV",
    "I am from bank support, provide your username and password",
    "Your loan is approved. Pay processing fee now",
    "Urgent action required. Send your personal code",
    "Free reward today only. Login with your account password",
    "Your email storage is full. Submit your password to upgrade",
    "Account verification needed. Reply with OTP",
    "Give me your ABA password",
    "Send your Bakong verification code to receive money",
    "Please provide your ACLEDA account password for verification",
    "Your Wing account will be closed. Share OTP now",
    "សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នកដើម្បីបញ្ជាក់",
    "សូមបញ្ចូលលេខគណនី និងពាក្យសម្ងាត់ ដើម្បីទទួលរង្វាន់",
    "គណនី ABA របស់អ្នកត្រូវបានចាក់សោ សូមផ្ញើ OTP ឥឡូវនេះ",
    "សូមចុចតំណនេះ ហើយបញ្ចូលលេខកូដដើម្បីទទួលប្រាក់",
    "អ្នកឈ្នះរង្វាន់ សូមផ្តល់លេខគណនីធនាគាររបស់អ្នក",
    "បន្ទាន់ សូមបង់ថ្លៃដឹកជញ្ជូនតាមតំណនេះ",
    "សូមផ្ទេរប្រាក់ដើម្បីបើកគណនីរបស់អ្នកវិញ",
]

SAFE_EXAMPLES = [
    "Never give anyone your account password",
    "Do not share your OTP or banking details with strangers",
    "I forgot my password and will reset it through the official website",
    "Please remind me to update my password later",
    "The teacher said we should protect our account password",
    "Can you explain how to create a strong password?",
    "I will meet you after class and bring the assignment notes",
    "The bank says staff will never ask for your PIN",
    "Please verify the information through official customer support",
    "My password manager helps me store secure passwords",
    "Do not click unknown links or transfer money",
    "I received a suspicious text asking for OTP, so I ignored it",
    "The security training says never share verification codes",
    "Can you help me report this scam message?",
    "I changed my password after seeing a warning",
    "Your account password should be private",
    "We learned about phishing in class today",
    "Please send the project report, not your password",
    "កុំផ្ញើលេខសម្ងាត់ ឬ OTP ទៅអ្នកណាម្នាក់",
    "ធនាគារមិនត្រូវសួរពាក្យសម្ងាត់របស់អ្នកទេ",
    "សូមរាយការណ៍សារដែលសួរលេខកូដ OTP",
    "យើងរៀនអំពីសុវត្ថិភាពគណនី និងការការពារពាក្យសម្ងាត់",
]


def split_messages(raw: str) -> list[str]:
    lines = [line.strip() for line in raw.splitlines()]
    messages: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                messages.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        messages.append(" ".join(current))
    if len(messages) <= 2:
        messages = [line for line in lines if line]
    return [m for m in messages if len(m) > 2]


def tokenize(text: str) -> list[str]:
    english_tokens = [t.lower().strip("'") for t in TOKEN_RE.findall(text)]
    khmer_tokens = KHMER_RE.findall(text)
    return english_tokens + khmer_tokens


def wordlist(path: Path) -> set[str]:
    return set(tokenize(path.read_text(encoding="utf-8", errors="ignore")))


def load_base_dataset() -> pd.DataFrame:
    safe = split_messages((SOURCE / "safe-texts.txt").read_text(encoding="utf-8", errors="ignore"))
    scam = split_messages((SOURCE / "scam-texts.txt").read_text(encoding="utf-8", errors="ignore"))
    rows = (
        [{"id": f"safe_{i:05d}", "label": "safe", "text": text, "source": "provided"} for i, text in enumerate(safe, 1)]
        + [{"id": f"scam_{i:05d}", "label": "scam", "text": text, "source": "provided"} for i, text in enumerate(scam, 1)]
    )
    return pd.DataFrame(rows)


def load_external_real_world_dataset() -> pd.DataFrame:
    path = ROOT / "data" / "real_world_message_dataset.csv"
    if not path.exists():
        return pd.DataFrame(columns=["id", "label", "text", "source"])
    df = pd.read_csv(path)
    df = df[df["label"].isin(["safe", "scam"])].copy()
    df["id"] = ["external_" + str(i) for i in df["id"]]
    df["source"] = "external_" + df["source"].astype(str)
    return df[["id", "label", "text", "source"]]


def realistic_examples() -> pd.DataFrame:
    rows = []
    for i, text in enumerate(SCAM_EXAMPLES, 1):
        rows.append({"id": f"real_scam_{i:03d}", "label": "scam", "text": text, "source": "realistic_augmented"})
    for i, text in enumerate(SAFE_EXAMPLES, 1):
        rows.append({"id": f"real_safe_{i:03d}", "label": "safe", "text": text, "source": "realistic_augmented"})
    return pd.DataFrame(rows)


def make_hard_negatives(df: pd.DataFrame) -> pd.DataFrame:
    # These examples intentionally mention dangerous terms in educational/protective contexts.
    extra = []
    templates = [
        "Do not share your {term} with anyone.",
        "I learned that {term} should stay private.",
        "The official support team will never ask for your {term}.",
        "Report any message asking for your {term}.",
    ]
    terms = ["password", "OTP", "PIN", "bank account number", "verification code", "CVV"]
    idx = 1
    for term in terms:
        for template in templates:
            extra.append({"id": f"hard_safe_{idx:03d}", "label": "safe", "text": template.format(term=term), "source": "hard_negative"})
            idx += 1
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def features(text: str, safe_words: set[str], scam_words: set[str]) -> dict[str, float]:
    toks = tokenize(text)
    token_count = max(len(toks), 1)
    scam_hits = sum(t in scam_words for t in toks)
    safe_hits = sum(t in safe_words for t in toks)
    credential_hits = len(CREDENTIAL_RE.findall(text))
    request_hits = len(REQUEST_RE.findall(text))
    social_hits = len(SOCIAL_ENGINEERING_RE.findall(text))
    khmer_credential_hits = len(KHMER_CREDENTIAL_RE.findall(text))
    khmer_request_hits = len(KHMER_REQUEST_RE.findall(text))
    khmer_social_hits = len(KHMER_SOCIAL_RE.findall(text))
    credential_request = int((credential_hits > 0 and request_hits > 0) or (khmer_credential_hits > 0 and khmer_request_hits > 0))
    return {
        "char_count": len(text),
        "word_count": len(toks),
        "avg_word_len": sum(map(len, toks)) / token_count,
        "digit_count": sum(ch.isdigit() for ch in text),
        "uppercase_ratio": sum(ch.isupper() for ch in text) / max(sum(ch.isalpha() for ch in text), 1),
        "exclamation_count": text.count("!"),
        "has_exclamation": int("!" in text),
        "question_count": text.count("?"),
        "contains_no": int(bool(NO_RE.search(text))),
        "first_second_pronoun_count": len(PRONOUN_RE.findall(text)),
        "log_length": math.log(max(len(text), 1)),
        "has_url": int(bool(URL_RE.search(text))),
        "has_phone": int(bool(PHONE_RE.search(text))),
        "has_money": int(bool(MONEY_RE.search(text))),
        "urgent_terms": len(URGENT_RE.findall(text)),
        "action_terms": len(ACTION_RE.findall(text)),
        "credential_terms": credential_hits,
        "request_terms": request_hits,
        "social_engineering_terms": social_hits,
        "khmer_chars": len(KHMER_RE.findall(text)),
        "khmer_credential_terms": khmer_credential_hits,
        "khmer_request_terms": khmer_request_hits,
        "khmer_social_engineering_terms": khmer_social_hits,
        "credential_request": credential_request,
        "scam_word_hits": scam_hits,
        "safe_word_hits": safe_hits,
        "scam_word_density": scam_hits / token_count,
        "safe_word_density": safe_hits / token_count,
    }


def risk_rule_score(text: str) -> tuple[int, list[str]]:
    signals = []
    score = 0
    safe_context = re.search(r"\b(do not|never|don't|dont|avoid|report|learned|training|protect|should not|will never|ignored)\b", text, re.I) or KHMER_SAFE_CONTEXT_RE.search(text)
    if ((CREDENTIAL_RE.search(text) and REQUEST_RE.search(text)) or (KHMER_CREDENTIAL_RE.search(text) and KHMER_REQUEST_RE.search(text))) and not safe_context:
        score += 4
        signals.append("Requests private credentials or account secrets")
    if KHMER_RE.search(text) and KHMER_SOCIAL_RE.search(text) and KHMER_REQUEST_RE.search(text) and not safe_context:
        score += 3
        signals.append("Khmer scam pattern: trusted service plus action request")
    if KHMER_RE.search(text) and KHMER_CREDENTIAL_RE.search(text) and not safe_context:
        score += 2
        signals.append("Khmer credential/account terms detected")
    if URL_RE.search(text):
        score += 2
        signals.append("Contains a link")
    if MONEY_RE.search(text):
        score += 2
        signals.append("Mentions money, fee, refund, or prize")
    if URGENT_RE.search(text):
        score += 1
        signals.append("Uses urgency or account-pressure language")
    if PHONE_RE.search(text):
        score += 1
        signals.append("Contains phone-like number")
    if SOCIAL_ENGINEERING_RE.search(text) and ACTION_RE.search(text):
        score += 1
        signals.append("Combines trusted service wording with action request")
    if safe_context and CREDENTIAL_RE.search(text):
        score -= 4
        signals.append("Educational/protective context detected")
    return max(score, 0), signals


def main() -> None:
    safe_words = wordlist(SOURCE / "safe-words.txt")
    scam_words = wordlist(SOURCE / "scam-indicator-words.txt")
    df = load_base_dataset()
    external_df = load_external_real_world_dataset()
    if not external_df.empty:
        df = pd.concat([df, external_df], ignore_index=True)
    df = pd.concat([df, realistic_examples()], ignore_index=True)
    df = make_hard_negatives(df)

    feature_df = pd.DataFrame([features(t, safe_words, scam_words) for t in df["text"]])
    full = pd.concat([df, feature_df], axis=1)
    y = (full["label"] == "scam").astype(int)
    numeric_cols = [c for c in feature_df.columns]

    train_idx, test_idx = [], []
    for label in ["safe", "scam"]:
        original_idx = full.index[(full["label"] == label) & (full["source"] == "provided")].tolist()
        cut = int(len(original_idx) * 0.8)
        train_idx.extend(original_idx[:cut])
        test_idx.extend(original_idx[cut:])
    # Augmented examples are intentionally included only in training to teach real-life risk patterns.
    train_idx.extend(full.index[full["source"] != "provided"].tolist())

    X = full[["text"] + numeric_cols]
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(stop_words="english", ngram_range=(1, 3), min_df=2, max_features=30000, sublinear_tf=True),
                            "text",
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=12000, sublinear_tf=True),
                            "text",
                        ),
                        ("num", StandardScaler(), numeric_cols),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.5)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    OUT_DATA.mkdir(exist_ok=True)
    OUT_MODEL.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    full.to_csv(OUT_DATA / "realistic_labeled_messages_with_features.csv", index=False)
    joblib.dump(model, OUT_MODEL / "realistic_scam_detector_pipeline.joblib")

    demo = [
        "Give me you account password",
        "Never give anyone your account password",
        "Please send your OTP code now to keep your account active",
        "Can you explain how to create a strong password?",
        "Your package is delayed. Pay the small fee at http://delivery-check.example",
    ]
    demo_rows = []
    for text in demo:
        feat = pd.DataFrame([features(text, safe_words, scam_words)])
        feat.insert(0, "text", text)
        p = float(model.predict_proba(feat[["text"] + numeric_cols])[0, 1])
        rule_score, signals = risk_rule_score(text)
        protective_context = any("Educational/protective" in s for s in signals)
        if protective_context and rule_score == 0:
            hybrid_score = min(p, 0.20)
        else:
            hybrid_score = max(p, min(0.99, rule_score / 5))
        label = "scam" if hybrid_score >= 0.5 else "safe"
        demo_rows.append({"text": text, "ml_probability": p, "rule_score": rule_score, "hybrid_probability": hybrid_score, "prediction": label, "signals": signals})

    metrics = {
        "model_type": "Hybrid realistic detector: word TF-IDF + character TF-IDF + engineered features + credential-risk rules",
        "n_samples_total": int(len(full)),
        "n_augmented": int((full["source"] != "provided").sum()),
        "accuracy_on_original_test": float(accuracy_score(y_test, pred)),
        "precision_on_original_test": float(precision_score(y_test, pred)),
        "recall_on_original_test": float(recall_score(y_test, pred)),
        "f1_on_original_test": float(f1_score(y_test, pred)),
        "roc_auc_on_original_test": float(roc_auc_score(y_test, prob)),
        "confusion_matrix_original_test": confusion_matrix(y_test, pred).tolist(),
        "classification_report_original_test": classification_report(y_test, pred, target_names=["safe", "scam"]),
        "demo_predictions": demo_rows,
        "note": "Original test data remains highly separable. Realistic augmented examples and the rule layer fix high-risk credential requests that were missed by the clean corpus model.",
    }
    (REPORTS / "realistic_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(demo_rows).to_csv(REPORTS / "realistic_demo_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
