from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "streamlit_app.py"
FEEDBACK = ROOT / "data" / "user_feedback.csv"
EXAMPLES = ROOT / "data" / "real_life_scam_examples.csv"


app = r'''from pathlib import Path
import csv
import html
import math
import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "realistic_scam_detector_pipeline.joblib"
DATA_PATH = ROOT / "data" / "realistic_labeled_messages_with_features.csv"
FEEDBACK_PATH = ROOT / "data" / "user_feedback.csv"
SOURCE_DIR = ROOT.parent

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+|\d+(?:[\.,]\d+)?|[$]\s*\d+")
KHMER_RE = re.compile(r"[\u1780-\u17FF]")
URL_RE = re.compile(r"https?://\S+|www\.\S+|bit\.ly\S*|tinyurl\S*|t\.co\S*|\S+\.com\S*", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\s\-().]*){7,}")
MONEY_RE = re.compile(r"[$]\s?\d+|\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:usd|dollars?|cash|prize|bonus|refund|fee|payment|loan)\b|\b(cash|prize|bonus|refund|fee|payment|loan)\b", re.I)
URGENT_RE = re.compile(r"\b(urgent|immediately|now|today|limited|final|expire|expired|act fast|asap|suspended|locked|verify now)\b", re.I)
ACTION_RE = re.compile(r"\b(click|claim|verify|confirm|reply|call|text|login|update|send|transfer|provide|share|submit|enter|reset|pay)\b", re.I)
NO_RE = re.compile(r"\bno\b", re.I)
PRONOUN_RE = re.compile(r"\b(i|me|my|you|your)\b", re.I)
CREDENTIAL_RE = re.compile(r"\b(password|passcode|otp|pin|code|verification code|login|username|account number|bank detail|card number|cvv|private key|seed phrase)\b", re.I)
REQUEST_RE = re.compile(r"\b(give|send|share|tell|provide|submit|enter|type|reply|confirm|verify|update|pay)\b", re.I)
SOCIAL_ENGINEERING_RE = re.compile(r"\b(account|bank|wallet|paypal|telegram|whatsapp|facebook|email|security|support|admin|prize|reward|refund|delivery|parcel|loan|job|investment)\b", re.I)
SAFE_CONTEXT_RE = re.compile(r"\b(do not|never|don't|dont|avoid|report|learned|training|protect|should not|will never|ignored)\b", re.I)

KHMER_CREDENTIAL_RE = re.compile(r"(ពាក្យសម្ងាត់|លេខសម្ងាត់|អូធីភី|OTP|កូដ|លេខកូដ|លេខគណនី|គណនី|លេខកាត|ស៊ីវីវី|CVV)")
KHMER_REQUEST_RE = re.compile(r"(ផ្ញើ|ផ្ដល់|ផ្តល់|ប្រាប់|បញ្ចូល|បញ្ជាក់|ចុច|បង់|ផ្ទេរ|ទូទាត់|ចែករំលែក|បំពេញ|ឆ្លើយតប)")
KHMER_SOCIAL_RE = re.compile(r"(ធនាគារ|ABA|ACLEDA|Wing|វីង|TrueMoney|Bakong|បាគង|គណនី|សុវត្ថិភាព|រង្វាន់|ប្រាក់|លុយ|កម្ចី|ការងារ|វិនិយោគ|ដឹកជញ្ជូន|កញ្ចប់|បិទ|ចាក់សោ|ផុតកំណត់|បន្ទាន់)")
KHMER_SAFE_CONTEXT_RE = re.compile(r"(កុំ|មិនត្រូវ|មិនគួរ|កុំចែករំលែក|កុំផ្ញើ|ការពារ|រាយការណ៍|ព្រមាន|រៀន|សុវត្ថិភាព)")

HIGHLIGHT_PATTERNS = [
    r"OTP", r"password", r"passcode", r"PIN", r"CVV", r"seed phrase", r"account number",
    r"verify", r"urgent", r"locked", r"prize", r"refund", r"fee", r"click",
    r"ពាក្យសម្ងាត់", r"លេខសម្ងាត់", r"អូធីភី", r"លេខគណនី", r"គណនី", r"បន្ទាន់",
    r"រង្វាន់", r"ផ្ញើ", r"បញ្ចូល", r"បង់", r"ផ្ទេរ", r"ធនាគារ",
]

SCAM_EXAMPLES = pd.DataFrame([
    {"type": "Password / OTP theft", "message": "សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នកដើម្បីបញ្ជាក់", "why": "Asks for account password/secret code."},
    {"type": "Fake bank support", "message": "គណនី ABA របស់អ្នកត្រូវបានចាក់សោ សូមផ្ញើ OTP ឥឡូវនេះ", "why": "Uses bank name, account-lock pressure, and OTP request."},
    {"type": "Prize scam", "message": "អ្នកឈ្នះរង្វាន់ សូមផ្តល់លេខគណនីធនាគាររបស់អ្នក", "why": "Prize offer asks for bank account details."},
    {"type": "Delivery fee scam", "message": "បន្ទាន់ សូមបង់ថ្លៃដឹកជញ្ជូនតាមតំណនេះ", "why": "Urgent delivery fee with link/payment request."},
    {"type": "Password / OTP theft", "message": "Give me your account password", "why": "Directly requests a private password."},
    {"type": "Account locked / urgent verification", "message": "Your account is locked. Click now and enter your OTP.", "why": "Urgency plus link/action plus OTP."},
])


st.set_page_config(page_title="Scam Safety Assistant", page_icon="!", layout="wide")

st.markdown("""
<style>
.main .block-container {padding-top: 1.4rem; max-width: 1240px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 10px; background: linear-gradient(135deg, #102A43 0%, #1F7A8C 55%, #2A9D8F 100%); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2.25rem; margin: 0 0 .3rem 0;}
.hero p {font-size: 1.05rem; margin: 0; opacity: .95;}
.privacy {padding: .8rem 1rem; background:#FFF7ED; border:1px solid #FDBA74; color:#7C2D12; border-radius:8px; margin-bottom:1rem;}
.critical {padding: 1rem; border-radius: 8px; background: #7F1D1D; color: white;}
.high {padding: 1rem; border-radius: 8px; background: #FEE2E2; border: 1px solid #FCA5A5; color: #7F1D1D;}
.medium {padding: 1rem; border-radius: 8px; background: #FEF3C7; border: 1px solid #FCD34D; color: #78350F;}
.low {padding: 1rem; border-radius: 8px; background: #DCFCE7; border: 1px solid #86EFAC; color: #14532D;}
.highlight {background:#FEF08A; padding:0 .15rem; border-radius:3px; font-weight:700;}
.khmer-box {padding: 1rem; border-radius: 8px; background: #F8FAFC; border: 1px solid #CBD5E1;}
.small-note {color:#64748B; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


def tokenize(text):
    return [t.lower().strip("'") for t in TOKEN_RE.findall(text)] + KHMER_RE.findall(text)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_wordlists():
    safe_words = set(tokenize((SOURCE_DIR / "safe-words.txt").read_text(encoding="utf-8", errors="ignore")))
    scam_words = set(tokenize((SOURCE_DIR / "scam-indicator-words.txt").read_text(encoding="utf-8", errors="ignore")))
    return safe_words, scam_words


@st.cache_data
def load_dataset_preview():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH).sample(min(1200, len(pd.read_csv(DATA_PATH))), random_state=42)
    return pd.DataFrame()


def build_features(text, safe_words, scam_words):
    toks = tokenize(text)
    token_count = max(len(toks), 1)
    scam_hits = sum(t in scam_words for t in toks)
    safe_hits = sum(t in safe_words for t in toks)
    safe_context = bool(SAFE_CONTEXT_RE.search(text) or KHMER_SAFE_CONTEXT_RE.search(text))
    return pd.DataFrame([{
        "text": text,
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
        "credential_terms": len(CREDENTIAL_RE.findall(text)),
        "request_terms": len(REQUEST_RE.findall(text)),
        "social_engineering_terms": len(SOCIAL_ENGINEERING_RE.findall(text)),
        "khmer_chars": len(KHMER_RE.findall(text)),
        "khmer_credential_terms": len(KHMER_CREDENTIAL_RE.findall(text)),
        "khmer_request_terms": len(KHMER_REQUEST_RE.findall(text)),
        "khmer_social_engineering_terms": len(KHMER_SOCIAL_RE.findall(text)),
        "credential_request": int(bool(((CREDENTIAL_RE.search(text) and REQUEST_RE.search(text)) or (KHMER_CREDENTIAL_RE.search(text) and KHMER_REQUEST_RE.search(text))) and not safe_context)),
        "scam_word_hits": scam_hits,
        "safe_word_hits": safe_hits,
        "scam_word_density": scam_hits / token_count,
        "safe_word_density": safe_hits / token_count,
    }])


def rule_risk_score(text):
    score = 0
    signals = []
    safe_context = bool(SAFE_CONTEXT_RE.search(text) or KHMER_SAFE_CONTEXT_RE.search(text))
    if ((CREDENTIAL_RE.search(text) and REQUEST_RE.search(text)) or (KHMER_CREDENTIAL_RE.search(text) and KHMER_REQUEST_RE.search(text))) and not safe_context:
        score += 5
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
        signals.append("Mentions money, fee, refund, prize, or loan")
    if URGENT_RE.search(text):
        score += 1
        signals.append("Uses urgency or account-pressure language")
    if PHONE_RE.search(text):
        score += 1
        signals.append("Contains a phone-like number")
    if SOCIAL_ENGINEERING_RE.search(text) and ACTION_RE.search(text):
        score += 1
        signals.append("Combines trusted service wording with an action request")
    if safe_context and (CREDENTIAL_RE.search(text) or KHMER_CREDENTIAL_RE.search(text)):
        score = max(score - 5, 0)
        signals.append("Educational/protective context detected")
    return score, signals


def scam_type(text, signals):
    t = text.lower()
    if CREDENTIAL_RE.search(text) or KHMER_CREDENTIAL_RE.search(text) or "otp" in t:
        return "Password / OTP theft"
    if re.search(r"\b(bank|aba|acleda|wing|bakong|wallet|account|support|security)\b", t, re.I) or re.search(r"(ធនាគារ|ABA|ACLEDA|Wing|Bakong|គណនី|សុវត្ថិភាព)", text):
        return "Fake bank support"
    if re.search(r"\b(prize|reward|winner|bonus)\b", t, re.I) or re.search(r"(រង្វាន់|ឈ្នះ)", text):
        return "Prize or reward scam"
    if re.search(r"\b(delivery|parcel|package|shipping|fee)\b", t, re.I) or re.search(r"(ដឹកជញ្ជូន|កញ្ចប់|ថ្លៃ)", text):
        return "Delivery fee scam"
    if re.search(r"\b(loan|credit|processing fee)\b", t, re.I) or re.search(r"(កម្ចី|ឥណទាន)", text):
        return "Loan scam"
    if re.search(r"\b(job|salary|recruit|work from home)\b", t, re.I) or re.search(r"(ការងារ|ប្រាក់ខែ)", text):
        return "Job scam"
    if re.search(r"\b(love|romance|girlfriend|boyfriend|trust me)\b", t, re.I):
        return "Romance or trust scam"
    if re.search(r"\b(locked|suspended|expired|verify now)\b", t, re.I) or re.search(r"(ចាក់សោ|បិទ|ផុតកំណត់|បន្ទាន់)", text):
        return "Account locked / urgent verification scam"
    if signals:
        return "Unknown suspicious message"
    return "No scam type detected"


def hybrid_decision(text, ml_probability, threshold):
    rule_score, signals = rule_risk_score(text)
    protective = any("Educational/protective" in signal for signal in signals)
    if protective and rule_score == 0:
        hybrid_probability = min(ml_probability, 0.20)
    else:
        hybrid_probability = max(ml_probability, min(0.99, rule_score / 6))
    prediction = "Scam" if hybrid_probability >= threshold else "Safe"
    if rule_score >= 5 and not protective:
        level = "Critical"
    elif hybrid_probability >= 0.75:
        level = "High"
    elif hybrid_probability >= 0.50:
        level = "Medium"
    else:
        level = "Low"
    return prediction, hybrid_probability, rule_score, signals, level, scam_type(text, signals)


def highlight_message(text):
    escaped = html.escape(text)
    for pattern in sorted(HIGHLIGHT_PATTERNS, key=len, reverse=True):
        escaped = re.sub(f"({re.escape(pattern)})", r'<span class="highlight">\1</span>', escaped, flags=re.I)
    escaped = URL_RE.sub(lambda m: f'<span class="highlight">{html.escape(m.group(0))}</span>', escaped)
    escaped = PHONE_RE.sub(lambda m: f'<span class="highlight">{html.escape(m.group(0))}</span>', escaped)
    return escaped.replace("\n", "<br>")


def advice_for(level, category):
    base = [
        "Do not reply to the sender.",
        "Do not click links or open attachments.",
        "Do not send money or pay any fee.",
        "Do not share OTP, password, PIN, CVV, seed phrase, or account number.",
        "Verify using the official app, website, or phone number you already trust.",
        "Screenshot and save evidence.",
        "Block and report the sender.",
    ]
    if category == "Fake bank support":
        base.insert(4, "Call your bank from the official number or open the official banking app.")
    if category == "Delivery fee scam":
        base.insert(4, "Check delivery status only from the official courier website or app.")
    if category == "Job scam":
        base.insert(4, "Do not pay a recruitment or training fee before verifying the company.")
    return base


def save_feedback(message, prediction, risk, category, feedback):
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    exists = FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "message", "prediction", "risk", "category", "feedback"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": message,
            "prediction": prediction,
            "risk": risk,
            "category": category,
            "feedback": feedback,
        })


model = load_model()
safe_words, scam_words = load_wordlists()

st.markdown("""
<div class="hero">
  <h1>Scam Safety Assistant</h1>
  <p>Detect suspicious messages, explain the risk, and guide people on what to do next.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="privacy"><b>Privacy warning:</b> This app does not need your real password, OTP, bank account number, card number, or private code. Never paste real sensitive information.</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Analyze Message", "Scam Type", "Why Suspicious", "What To Do Next", "Khmer Scam Examples", "Report / Feedback", "Batch Check"])

if "last_result" not in st.session_state:
    st.session_state.last_result = None

with tabs[0]:
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        default_message = "សូមបញ្ចូលលេខគណនី និងពាក្យសម្ងាត់ ដើម្បីទទួលរង្វាន់"
        message = st.text_area("Paste a suspicious message", value=default_message, height=165)
        threshold = st.slider("Scam warning threshold", 0.10, 0.90, 0.50, 0.05)
        run = st.button("Analyze message", type="primary", use_container_width=True)
    with right:
        st.subheader("Never share")
        st.write("- Password / ពាក្យសម្ងាត់")
        st.write("- OTP / verification code")
        st.write("- PIN, CVV, card number")
        st.write("- Bank account number")
        st.write("- Seed phrase / private key")

    if run and message.strip():
        features = build_features(message, safe_words, scam_words)
        ml_probability = float(model.predict_proba(features)[0, 1])
        prediction, risk, rule_score, signals, level, category = hybrid_decision(message, ml_probability, threshold)
        st.session_state.last_result = {
            "message": message, "prediction": prediction, "risk": risk, "rule_score": rule_score,
            "signals": signals, "level": level, "category": category, "ml_probability": ml_probability,
        }

    result = st.session_state.last_result
    if result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", result["prediction"])
        c2.metric("Risk level", result["level"])
        c3.metric("Hybrid scam risk", f"{result['risk']:.1%}", help=f"ML probability: {result['ml_probability']:.1%}; rule score: {result['rule_score']}")
        c4.metric("Scam type", result["category"])
        css_class = {"Critical": "critical", "High": "high", "Medium": "medium", "Low": "low"}[result["level"]]
        if result["prediction"] == "Scam":
            st.markdown(f'<div class="{css_class}"><b>{result["level"]} risk:</b> This message may be dangerous. Follow the safety steps before taking action.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="low"><b>Lower risk:</b> This message looks safer, but still verify important requests through official channels.</div>', unsafe_allow_html=True)
        st.subheader("Highlighted message")
        st.markdown(f'<div class="khmer-box">{highlight_message(result["message"])}</div>', unsafe_allow_html=True)

with tabs[1]:
    result = st.session_state.last_result
    st.subheader("Scam Type Classification")
    if result:
        st.metric("Detected type", result["category"])
        st.write("This category is based on message content, risk rules, Khmer/English scam terms, and model signals.")
    else:
        st.info("Analyze a message first.")
    st.dataframe(pd.DataFrame([
        {"type": "Password / OTP theft", "description": "Requests password, OTP, PIN, CVV, account number, seed phrase."},
        {"type": "Fake bank support", "description": "Pretends to be bank/security/support and asks for action."},
        {"type": "Prize or reward scam", "description": "Promises money/reward and asks for details or payment."},
        {"type": "Delivery fee scam", "description": "Asks to pay delivery/customs fee through link."},
        {"type": "Loan scam", "description": "Promises loan approval and asks for fee or documents."},
        {"type": "Job scam", "description": "Fake job offer requiring payment or private details."},
        {"type": "Romance or trust scam", "description": "Uses personal trust/emotion to ask money or secrets."},
        {"type": "Account locked / urgent verification", "description": "Urgent account-lock warning asking verification."},
    ]), use_container_width=True)

with tabs[2]:
    st.subheader("Why is this suspicious?")
    result = st.session_state.last_result
    if result:
        if result["signals"]:
            for signal in result["signals"]:
                st.warning(signal)
        else:
            st.success("No strong risk signals found.")
        st.write("Model probability:", f"{result['ml_probability']:.1%}")
        st.write("Rule score:", result["rule_score"])
    else:
        st.info("Analyze a message first.")

with tabs[3]:
    st.subheader("What should I do now?")
    result = st.session_state.last_result
    if result and result["prediction"] == "Scam":
        for item in advice_for(result["level"], result["category"]):
            st.write(f"- {item}")
        st.markdown("#### How to verify safely")
        st.write("- Open the official app yourself. Do not use the link in the message.")
        st.write("- Call the official number from the bank/company website or card.")
        st.write("- Ask a trusted person before sending money or private information.")
    elif result:
        st.write("- If the message asks for money, password, OTP, or bank information, verify through official channels.")
        st.write("- Keep private information out of chat messages.")
    else:
        st.info("Analyze a message first.")

with tabs[4]:
    st.subheader("Khmer Scam Education")
    st.markdown("""
<div class="khmer-box">
<b>ពាក្យគួរប្រុងប្រយ័ត្ន:</b> ពាក្យសម្ងាត់, លេខសម្ងាត់, OTP, កូដ, លេខគណនី, បន្ទាន់, រង្វាន់, បង់ថ្លៃ, ផ្ទេរប្រាក់, ចុចតំណ។<br><br>
<b>ហេតុអ្វីគ្រោះថ្នាក់?</b> អ្នកបោកប្រាស់តែងតែប្រើពាក្យបន្ទាន់ ឬរង្វាន់ ដើម្បីឱ្យអ្នកផ្ញើលេខសម្ងាត់ OTP ឬប្រាក់។
</div>
""", unsafe_allow_html=True)
    st.markdown("#### Example scam messages")
    st.dataframe(SCAM_EXAMPLES, use_container_width=True)
    st.markdown("#### Safe vs scam comparison")
    st.write("- Safe: `កុំផ្ញើលេខសម្ងាត់ ឬ OTP ទៅអ្នកណាម្នាក់`")
    st.write("- Scam: `សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នកដើម្បីបញ្ជាក់`")

with tabs[5]:
    st.subheader("Report / Feedback")
    result = st.session_state.last_result
    if result:
        st.write("Help improve this project by marking the prediction.")
        f1, f2, f3 = st.columns(3)
        if f1.button("Correct", use_container_width=True):
            save_feedback(result["message"], result["prediction"], result["risk"], result["category"], "correct")
            st.success("Saved feedback: correct.")
        if f2.button("Wrong, this is scam", use_container_width=True):
            save_feedback(result["message"], result["prediction"], result["risk"], result["category"], "wrong_scam")
            st.success("Saved feedback: wrong, true label scam.")
        if f3.button("Wrong, this is safe", use_container_width=True):
            save_feedback(result["message"], result["prediction"], result["risk"], result["category"], "wrong_safe")
            st.success("Saved feedback: wrong, true label safe.")
        if FEEDBACK_PATH.exists():
            st.download_button("Download feedback CSV", FEEDBACK_PATH.read_bytes(), "user_feedback.csv", "text/csv")
    else:
        st.info("Analyze a message first.")

with tabs[6]:
    st.subheader("Batch Check")
    uploaded = st.file_uploader("Upload CSV with a text column", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        if "text" not in batch.columns:
            st.error("CSV must contain a text column.")
        else:
            rows = [build_features(str(t), safe_words, scam_words) for t in batch["text"].fillna("")]
            features = pd.concat(rows, ignore_index=True)
            probs = model.predict_proba(features)[:, 1]
            analyzed = [hybrid_decision(str(text), float(prob), 0.50) for text, prob in zip(batch["text"].fillna(""), probs)]
            result_df = batch.copy()
            result_df["ml_probability"] = probs
            result_df["hybrid_scam_risk"] = [a[1] for a in analyzed]
            result_df["risk_level"] = [a[4] for a in analyzed]
            result_df["scam_type"] = [a[5] for a in analyzed]
            result_df["prediction"] = [a[0].lower() for a in analyzed]
            st.dataframe(result_df, use_container_width=True)
            st.download_button("Download scored CSV", result_df.to_csv(index=False).encode("utf-8"), "scored_messages.csv", "text/csv")
'''

examples = """type,message,why
Password / OTP theft,សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នកដើម្បីបញ្ជាក់,Asks for account password
Fake bank support,គណនី ABA របស់អ្នកត្រូវបានចាក់សោ សូមផ្ញើ OTP ឥឡូវនេះ,Fake bank lock plus OTP
Prize scam,អ្នកឈ្នះរង្វាន់ សូមផ្តល់លេខគណនីធនាគាររបស់អ្នក,Prize plus bank details
Delivery fee scam,បន្ទាន់ សូមបង់ថ្លៃដឹកជញ្ជូនតាមតំណនេះ,Urgent delivery payment
Password / OTP theft,Give me your account password,Direct private credential request
"""

APP.write_text(app, encoding="utf-8")
FEEDBACK.parent.mkdir(exist_ok=True)
EXAMPLES.write_text(examples, encoding="utf-8")
print("Scam Safety Assistant upgraded.")
