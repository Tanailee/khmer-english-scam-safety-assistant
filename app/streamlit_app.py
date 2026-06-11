from pathlib import Path
import csv
import html
import io
import math
import os
import re
import shutil
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "realistic_scam_detector_pipeline.joblib"
DATA_PATH = ROOT / "data" / "realistic_labeled_messages_with_features.csv"
FEEDBACK_PATH = ROOT / "data" / "user_feedback.csv"
TESSDATA_DIR = ROOT / "tessdata"
SOURCE_DIR = ROOT.parent

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+|\d+(?:[\.,]\d+)?|[$]\s*\d+")
KHMER_RE = re.compile(r"[\u1780-\u17FF]")
URL_RE = re.compile(r"https?://\S+|www\.\S+|bit\.ly\S*|tinyurl\S*|t\.co\S*|\S+\.com\S*", re.I)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d[\s\-().]*){7,}")
MONEY_RE = re.compile(r"[$]\s?\d+|\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:usd|dollars?|cash|prize|bonus|refund|fee|payment|loan)\b|\b(cash|prize|bonus|refund|fee|payment|loan)\b", re.I)
URGENT_RE = re.compile(r"\b(urgent|immediately|now|today|limited|final|expire|expired|act fast|asap|suspended|locked|verify now)\b", re.I)
ACTION_RE = re.compile(r"\b(click|claim|verify|confirm|reply|call|text|login|update|send|transfer|provide|share|submit|enter|reset|pay)\b", re.I)
GAMBLING_RE = re.compile(r"\b(casino|baccarat|roulette|slot|bet|gaming|online entertainment|first deposit|deposit bonus|download.*app)\b", re.I)
INVESTMENT_RE = re.compile(r"\b(invest|investment|profit|guaranteed return|crypto|trading|double your money|commission|passive income)\b", re.I)
IMPERSONATION_RE = re.compile(r"\b(bank staff|support team|security team|admin|customer service|official support|from aba|from acleda|from wing|from bakong)\b", re.I)
PUBLIC_NOTICE_RE = re.compile(r"\b(public notice|weather alert|ministry|heavy rain|flood|stay safe|announcement|meeting reminder|delivered successfully)\b", re.I)
HEALTH_NOTICE_RE = re.compile(r"\b(nodrug|no drug|drug prevention|hiv|aids|public health|awareness|education campaign|safe sex)\b", re.I)
RELATIONSHIP_SCAM_RE = re.compile(r"\b(love you|romance|relationship|girlfriend|boyfriend|fiance|trust only you|visit you|stuck overseas|cannot video call|gift card)\b", re.I)
ACCOUNT_TAKEOVER_RE = re.compile(r"\b(account takeover|login code|verification code|2fa|two-factor|your page|facebook page|telegram account|sim card|identity|date of birth|id number)\b", re.I)
MARKETPLACE_SCAM_RE = re.compile(r"\b(marketplace|buyer|seller|escrow|shipping|courier|customs|seller protection|verification fee|release payment|order is confirmed)\b", re.I)
THREAT_SCAM_RE = re.compile(r"\b(blackmail|private video|leak|hacked|arrest|court|legal notice|police|fine|penalty|bitcoin within|send it to your contacts)\b", re.I)
JOB_SCAM_RE = re.compile(r"\b(remote job|work from home|part-time job|job approved|training fee|registration fee|employee account|daily commission|higher orders|hr department)\b", re.I)
UNEXPECTED_MONEY_RE = re.compile(r"\b(inheritance|cash prize|government refund|international transfer|fund pending|clearance fee|anti-money-laundering|claim your money|won a cash)\b", re.I)
BEC_SCAM_RE = re.compile(r"\b(ceo request|finance team|wire|vendor banking|supplier account|invoice bank account|new account below|gift cards for clients|process payment)\b", re.I)
RECOVERY_SCAM_RE = re.compile(r"\b(recovered your lost|recover your money|recovery expert|stolen funds|upfront fee|wallet validation|refund your investment|recovery tax)\b", re.I)
DONATION_SCAM_RE = re.compile(r"\b(donation|charity|fundraiser|disaster|sick children|temple rebuilding|personal aba account|no receipt|hospital needs cash)\b", re.I)
CRITICAL_TRIGGER_RE = re.compile(
    r"\b("
    r"send|give|share|tell|provide|enter|submit|confirm|verify|update|reply"
    r")\b.{0,45}\b("
    r"otp|password|passcode|pin|cvv|verification code|login code|account number|card number|seed phrase|private key"
    r")\b|"
    r"\b("
    r"otp|password|passcode|pin|cvv|verification code|login code|account number|card number|seed phrase|private key"
    r")\b.{0,45}\b("
    r"send|give|share|tell|provide|enter|submit|confirm|verify|update|reply"
    r")\b",
    re.I,
)
LINK_URGENCY_RE = re.compile(r"\b(click|link|login|verify|claim|pay|update)\b.*\b(now|urgent|immediately|today|locked|suspended|expired|final)\b|\b(now|urgent|immediately|today|locked|suspended|expired|final)\b.*\b(click|link|login|verify|claim|pay|update)\b", re.I)
PAYMENT_PRESSURE_RE = re.compile(r"\b(pay|transfer|deposit|fee|processing fee|clearance fee|customs fee|first deposit|bonus|withdrawal)\b", re.I)
NO_RE = re.compile(r"\bno\b", re.I)
PRONOUN_RE = re.compile(r"\b(i|me|my|you|your)\b", re.I)
CREDENTIAL_RE = re.compile(r"\b(password|passcode|otp|pin|code|verification code|login|username|account number|bank detail|card number|cvv|private key|seed phrase)\b", re.I)
ACCOUNT_INFO_RE = re.compile(r"\b(account information|account info|personal information|private information|bank information|bank details|bank detail|identity information|id card|national id|passport|date of birth|phone number)\b", re.I)
REQUEST_RE = re.compile(r"\b(give|send|share|tell|provide|submit|enter|type|reply|confirm|verify|update|pay)\b", re.I)
SOCIAL_ENGINEERING_RE = re.compile(r"\b(account|bank|wallet|paypal|telegram|whatsapp|facebook|email|security|support|admin|prize|reward|refund|delivery|parcel|loan|job|investment)\b", re.I)
SAFE_CONTEXT_RE = re.compile(r"\b(do not|never|don't|dont|avoid|report|learned|training|education|awareness|reminder|warning|protect|should not|will never|will not|does not ask|do not send|not ask|no real bank|ignored)\b", re.I)
DIRECT_PRIVATE_INFO_RE = re.compile(
    r"\b(send|give|share|tell|provide|submit|enter|type|reply|confirm|verify|update)\b.{0,80}\b("
    r"account information|account info|personal information|private information|bank information|bank details|account number|otp|pin|password|cvv|card number|seed phrase|private key|id card|passport"
    r")\b",
    re.I,
)
OFF_PLATFORM_CONTACT_RE = re.compile(r"\b(to|via|through|contact|email|telegram|whatsapp|line)\b.{0,25}([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|@[\w_]{4,}|(?:\+?\d[\s\-().]*){7,})", re.I)
KHMER_GAMBLING_RE = re.compile(r"(កាស៊ីណូ|បាការ៉ាត់|ហ្គេម|ហ្គេមអនឡាញ|ដាក់ប្រាក់|ប្រាក់បន្ថែម|រង្វាន់)")
KHMER_INVESTMENT_RE = re.compile(r"(វិនិយោគ|ចំណេញ|ប្រាក់ចំណេញ|គ្រីបតូ|ជួញដូរ|ប្រាក់កម្រៃ)")
KHMER_PUBLIC_NOTICE_RE = re.compile(r"(សារជូនដំណឹង|ក្រសួង|អាកាសធាតុ|ភ្លៀង|ទឹកជំនន់|ប្រុងប្រយ័ត្ន|សាធារណជន)")

KHMER_CREDENTIAL_RE = re.compile(r"(ពាក្យសម្ងាត់|លេខសម្ងាត់|អូធីភី|OTP|កូដ|លេខកូដ|លេខគណនី|គណនី|លេខកាត|ស៊ីវីវី|CVV)")
KHMER_REQUEST_RE = re.compile(r"(ផ្ញើ|ផ្ដល់|ផ្តល់|ប្រាប់|បញ្ចូល|បញ្ជាក់|ចុច|បង់|ផ្ទេរ|ទូទាត់|ចែករំលែក|បំពេញ|ឆ្លើយតប)")
KHMER_SOCIAL_RE = re.compile(r"(ធនាគារ|ABA|ACLEDA|Wing|វីង|TrueMoney|Bakong|បាគង|គណនី|សុវត្ថិភាព|រង្វាន់|ប្រាក់|លុយ|កម្ចី|ការងារ|វិនិយោគ|ដឹកជញ្ជូន|កញ្ចប់|បិទ|ចាក់សោ|ផុតកំណត់|បន្ទាន់)")
KHMER_SAFE_CONTEXT_RE = re.compile(r"(កុំ|មិនត្រូវ|មិនគួរ|កុំចែករំលែក|កុំផ្ញើ|ការពារ|រាយការណ៍|ព្រមាន|រៀន|សុវត្ថិភាព)")

KHMER_DIRECT_CREDENTIAL_RE = re.compile(
    r"("
    r"\u1795\u17d2\u1789\u17be|\u1795\u17d2\u178f\u179b\u17cb|\u1794\u1789\u17d2\u1785\u17bc\u179b|\u1794\u1789\u17d2\u1787\u17b6\u1780\u17cb|\u1794\u17d2\u179a\u17b6\u1794\u17cb|\u1785\u17c2\u1780\u179a\u17c6\u179b\u17c2\u1780|\u1786\u17d2\u179b\u17be\u1799\u178f\u1794"
    r").{0,55}("
    r"\u1796\u17b6\u1780\u17d2\u1799\u179f\u1798\u17d2\u1784\u17b6\u178f\u17cb|\u179b\u17c1\u1781\u179f\u1798\u17d2\u1784\u17b6\u178f\u17cb|OTP|PIN|CVV|\u1780\u17bc\u178a|\u179b\u17c1\u1781\u1780\u17bc\u178a|\u179b\u17c1\u1781\u1782\u178e\u1793\u17b8|\u1782\u178e\u1793\u17b8|\u179b\u17c1\u1781\u1780\u17b6\u178f"
    r")|("
    r"\u1796\u17b6\u1780\u17d2\u1799\u179f\u1798\u17d2\u1784\u17b6\u178f\u17cb|\u179b\u17c1\u1781\u179f\u1798\u17d2\u1784\u17b6\u178f\u17cb|OTP|PIN|CVV|\u1780\u17bc\u178a|\u179b\u17c1\u1781\u1780\u17bc\u178a|\u179b\u17c1\u1781\u1782\u178e\u1793\u17b8|\u1782\u178e\u1793\u17b8|\u179b\u17c1\u1781\u1780\u17b6\u178f"
    r").{0,55}("
    r"\u1795\u17d2\u1789\u17be|\u1795\u17d2\u178f\u179b\u17cb|\u1794\u1789\u17d2\u1785\u17bc\u179b|\u1794\u1789\u17d2\u1787\u17b6\u1780\u17cb|\u1794\u17d2\u179a\u17b6\u1794\u17cb|\u1785\u17c2\u1780\u179a\u17c6\u179b\u17c2\u1780|\u1786\u17d2\u179b\u17be\u1799\u178f\u1794"
    r")",
    re.I,
)

HIGHLIGHT_PATTERNS = [
    r"OTP", r"password", r"passcode", r"PIN", r"CVV", r"seed phrase", r"account number",
    r"account information", r"personal information", r"private information", r"bank details",
    r"verify", r"urgent", r"locked", r"prize", r"refund", r"fee", r"click", r"send",
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
.stApp {background: #F6F8FC;}
.main .block-container {padding-top: 1.1rem; max-width: 1280px;}
header[data-testid="stHeader"] {background: rgba(246,248,252,.86); backdrop-filter: blur(12px);}
section[data-testid="stSidebar"] {background: #0F172A;}
section[data-testid="stSidebar"] * {color: #E5E7EB;}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: #CBD5E1;}
.hero {
  padding: 1.65rem 1.8rem;
  border-radius: 14px;
  background:
    linear-gradient(135deg, rgba(15,23,42,.96) 0%, rgba(17,94,89,.94) 54%, rgba(20,184,166,.88) 100%),
    radial-gradient(circle at 86% 18%, rgba(255,255,255,.28), transparent 32%);
  color: white;
  margin-bottom: 1rem;
  box-shadow: 0 18px 45px rgba(15,23,42,.18);
}
.hero h1 {font-size: 2.45rem; line-height: 1.04; margin: 0 0 .45rem 0; letter-spacing: 0;}
.hero p {font-size: 1.02rem; margin: 0; color: #DFF7F4;}
.hero-grid {display:flex; justify-content:space-between; gap:1.2rem; align-items:flex-end; flex-wrap:wrap;}
.hero-copy {max-width: 760px;}
.hero-badges {display:flex; gap:.5rem; flex-wrap:wrap; margin-top:1rem;}
.badge {
  display:inline-flex; align-items:center; gap:.35rem;
  padding:.38rem .65rem; border:1px solid rgba(255,255,255,.22);
  background: rgba(255,255,255,.12); color:#F8FAFC; border-radius:999px; font-size:.86rem;
}
.hero-panel {
  min-width: 230px; padding:.85rem 1rem; border-radius:12px;
  background: rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2);
}
.hero-panel b {font-size:1.4rem; display:block;}
.status-card, .pro-card {
  padding: 1rem;
  border: 1px solid #E2E8F0;
  background: rgba(255,255,255,.96);
  border-radius: 12px;
  box-shadow: 0 12px 28px rgba(15,23,42,.07);
}
.metric-grid {display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:.85rem; margin:.3rem 0 1rem;}
.metric-card {
  padding: 1rem;
  border: 1px solid #E2E8F0;
  border-radius: 12px;
  background: #FFFFFF;
  box-shadow: 0 8px 20px rgba(15,23,42,.055);
}
.metric-card .label {font-size:.8rem; color:#64748B; margin-bottom:.35rem;}
.metric-card .value {font-size:1.45rem; font-weight:760; color:#0F172A; line-height:1.16;}
.metric-card .sub {font-size:.78rem; color:#64748B; margin-top:.35rem;}
.decision-card {
  padding: 1.25rem 1.35rem;
  border-radius: 16px;
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  box-shadow: 0 16px 36px rgba(15,23,42,.10);
  margin: .9rem 0 1rem;
}
.decision-card.critical-card {border-color:#FCA5A5; background: linear-gradient(135deg, #FFF1F2 0%, #FFFFFF 60%);}
.decision-card.safe-card {border-color:#86EFAC; background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 60%);}
.decision-kicker {font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:#64748B; font-weight:800;}
.decision-title {font-size:2rem; line-height:1.1; color:#0F172A; font-weight:850; margin:.25rem 0 .55rem;}
.decision-meta {display:flex; gap:.65rem; flex-wrap:wrap; margin-top:.75rem;}
.decision-pill {padding:.42rem .7rem; border-radius:999px; background:#F1F5F9; color:#334155; font-weight:750; font-size:.86rem;}
.risk-bar {height: 13px; border-radius: 999px; background: #E2E8F0; overflow: hidden; margin: .8rem 0 .2rem;}
.risk-fill {height: 100%; border-radius: 999px;}
.result-title {font-size: 1.15rem; font-weight: 700; margin-bottom: .25rem;}
.muted {color:#64748B;}
.privacy {padding: .82rem 1rem; background:#FFF7ED; border:1px solid #FDBA74; color:#7C2D12; border-radius:12px; margin-bottom:1rem;}
.critical, .high, .medium, .low {padding: 1rem 1.1rem; border-radius: 12px; margin:.7rem 0;}
.critical {background: linear-gradient(135deg, #7F1D1D, #B91C1C); color: white; box-shadow: 0 12px 26px rgba(127,29,29,.20);}
.high {background: #FEE2E2; border: 1px solid #FCA5A5; color: #7F1D1D;}
.medium {background: #FEF3C7; border: 1px solid #FCD34D; color: #78350F;}
.low {background: #DCFCE7; border: 1px solid #86EFAC; color: #14532D;}
.highlight {background:#FEF08A; padding:0 .15rem; border-radius:3px; font-weight:700;}
.khmer-box {padding: 1rem; border-radius: 12px; background: #FFFFFF; border: 1px solid #CBD5E1; box-shadow: inset 0 1px 0 rgba(255,255,255,.6);}
.small-note {color:#64748B; font-size:.9rem;}
.home-grid {display:grid; grid-template-columns: 1.1fr .9fr; gap:1rem; margin-bottom:1rem;}
.home-card {
  padding: 1rem 1.1rem; border-radius:12px; border:1px solid #E2E8F0;
  background:#FFFFFF; box-shadow: 0 10px 24px rgba(15,23,42,.06);
}
.footer {
  margin-top:2rem; padding:1.1rem; border-top:1px solid #E2E8F0;
  color:#64748B; text-align:center; font-size:.92rem;
}
div[data-testid="stTextArea"] textarea {
  border-radius: 12px;
  border: 1px solid #CBD5E1;
  background: #FFFFFF;
  box-shadow: inset 0 1px 2px rgba(15,23,42,.04);
}
div[data-testid="stTextArea"] textarea:focus {border-color:#14B8A6; box-shadow:0 0 0 3px rgba(20,184,166,.14);}
.stButton button, .stDownloadButton button {
  border-radius: 10px;
  border: 1px solid transparent;
  font-weight: 700;
  min-height: 2.75rem;
}
.stButton button[kind="primary"] {
  background: linear-gradient(135deg, #E11D48 0%, #F97316 100%);
  box-shadow: 0 10px 22px rgba(225,29,72,.20);
}
.stTabs [data-baseweb="tab-list"] {gap: .25rem; border-bottom: 1px solid #E2E8F0;}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px 10px 0 0;
  padding: .75rem .9rem;
  font-weight: 650;
}
.stTabs [aria-selected="true"] {background:#FFFFFF; box-shadow: inset 0 -2px 0 #E11D48;}
div[data-testid="stDataFrame"] {border-radius: 12px; overflow:hidden;}
@media (max-width: 900px) {
  .metric-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .home-grid {grid-template-columns: 1fr;}
  .hero h1 {font-size:2rem;}
}
@media (max-width: 560px) {
  .metric-grid {grid-template-columns: 1fr;}
}
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


@st.cache_data
def load_scam_retrieval_corpus(max_rows=3500):
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["id", "label", "text", "source", "category"])
    df = pd.read_csv(DATA_PATH, usecols=["id", "label", "text", "source"])
    scam_df = df[df["label"].astype(str).str.lower().eq("scam")].copy()
    scam_df["text"] = scam_df["text"].fillna("").astype(str)
    scam_df = scam_df[scam_df["text"].str.len() > 8]
    if len(scam_df) > max_rows:
        scam_df = scam_df.sample(max_rows, random_state=42)
    scam_df["category"] = scam_df["text"].apply(lambda value: scam_type(value, []))
    return scam_df.reset_index(drop=True)


@st.cache_resource
def build_tfidf_retrieval_index(max_rows=3500):
    scam_df = load_scam_retrieval_corpus(max_rows=max_rows)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(scam_df["text"].tolist()) if not scam_df.empty else None
    return scam_df, vectorizer, matrix


def retrieve_similar_scam_tfidf(query, top_k=5):
    scam_df, vectorizer, matrix = build_tfidf_retrieval_index()
    if scam_df.empty or matrix is None:
        return pd.DataFrame()
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    results = scam_df.iloc[top_indices][["category", "text", "source"]].copy()
    results.insert(0, "similarity", [round(float(scores[idx]), 4) for idx in top_indices])
    return results


@st.cache_resource
def build_embedding_retrieval_index(max_rows=1800):
    scam_df = load_scam_retrieval_corpus(max_rows=max_rows)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        return scam_df, None, None, f"sentence-transformers is not installed: {exc}"
    try:
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True,
        )
        embeddings = model.encode(
            scam_df["text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return scam_df, model, embeddings, None
    except Exception as exc:
        return scam_df, None, None, str(exc)


def retrieve_similar_scam_embeddings(query, top_k=5):
    scam_df, embedding_model, embeddings, error = build_embedding_retrieval_index()
    if error:
        return pd.DataFrame(), error
    if scam_df.empty or embedding_model is None or embeddings is None:
        return pd.DataFrame(), "Embedding index is empty."
    query_embedding = embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    scores = cosine_similarity(query_embedding, embeddings).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    results = scam_df.iloc[top_indices][["category", "text", "source"]].copy()
    results.insert(0, "similarity", [round(float(scores[idx]), 4) for idx in top_indices])
    return results, None


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
        "credential_request": int(bool(((CREDENTIAL_RE.search(text) and (REQUEST_RE.search(text) or ACCOUNT_INFO_RE.search(text))) or DIRECT_PRIVATE_INFO_RE.search(text) or (KHMER_CREDENTIAL_RE.search(text) and KHMER_REQUEST_RE.search(text)) or KHMER_DIRECT_CREDENTIAL_RE.search(text)) and not safe_context)),
        "scam_word_hits": scam_hits,
        "safe_word_hits": safe_hits,
        "scam_word_density": scam_hits / token_count,
        "safe_word_density": safe_hits / token_count,
    }])


def rule_risk_score(text):
    score = 0
    signals = []
    safe_context = bool(SAFE_CONTEXT_RE.search(text) or KHMER_SAFE_CONTEXT_RE.search(text))
    public_notice = bool(PUBLIC_NOTICE_RE.search(text) or KHMER_PUBLIC_NOTICE_RE.search(text))
    if (CRITICAL_TRIGGER_RE.search(text) or DIRECT_PRIVATE_INFO_RE.search(text) or KHMER_DIRECT_CREDENTIAL_RE.search(text)) and not safe_context:
        score += 6
        signals.append("Critical trigger: directly asks for OTP, password, PIN, CVV, account information, or private identity data")
    if EMAIL_RE.search(text) and (CREDENTIAL_RE.search(text) or ACCOUNT_INFO_RE.search(text)) and REQUEST_RE.search(text) and not safe_context:
        score += 6
        signals.append("Critical trigger: asks to send private account information to an email address")
    if OFF_PLATFORM_CONTACT_RE.search(text) and (CREDENTIAL_RE.search(text) or ACCOUNT_INFO_RE.search(text) or KHMER_CREDENTIAL_RE.search(text)) and not safe_context:
        score += 4
        signals.append("Off-platform contact request with private information")
    if ((CREDENTIAL_RE.search(text) and (REQUEST_RE.search(text) or ACCOUNT_INFO_RE.search(text))) or DIRECT_PRIVATE_INFO_RE.search(text) or (KHMER_CREDENTIAL_RE.search(text) and KHMER_REQUEST_RE.search(text)) or KHMER_DIRECT_CREDENTIAL_RE.search(text)) and not safe_context:
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
    if URL_RE.search(text) and (URGENT_RE.search(text) or LINK_URGENCY_RE.search(text)) and not safe_context:
        score += 4
        signals.append("Suspicious link combined with urgency or account pressure")
    if MONEY_RE.search(text):
        score += 2
        signals.append("Mentions money, fee, refund, prize, or loan")
    if PAYMENT_PRESSURE_RE.search(text) and not safe_context:
        score += 2
        signals.append("Payment, fee, deposit, or transfer pressure detected")
    if GAMBLING_RE.search(text) or KHMER_GAMBLING_RE.search(text):
        score += 3
        signals.append("Online gambling, casino, deposit bonus, or betting promotion detected")
    if INVESTMENT_RE.search(text) or KHMER_INVESTMENT_RE.search(text):
        score += 3
        signals.append("Investment, crypto, or guaranteed-profit promise detected")
    if RELATIONSHIP_SCAM_RE.search(text) and (MONEY_RE.search(text) or PAYMENT_PRESSURE_RE.search(text) or REQUEST_RE.search(text)):
        score += 3
        signals.append("Relationship or trust-based money request detected")
    if ACCOUNT_TAKEOVER_RE.search(text) and (CREDENTIAL_RE.search(text) or REQUEST_RE.search(text)):
        score += 4
        signals.append("Account or identity takeover pattern detected")
    if MARKETPLACE_SCAM_RE.search(text) and (MONEY_RE.search(text) or PAYMENT_PRESSURE_RE.search(text) or URL_RE.search(text)):
        score += 3
        signals.append("Buying/selling or marketplace payment-release pattern detected")
    if THREAT_SCAM_RE.search(text):
        score += 4
        signals.append("Threat, blackmail, legal-pressure, or extortion pattern detected")
    if JOB_SCAM_RE.search(text) and (MONEY_RE.search(text) or PAYMENT_PRESSURE_RE.search(text) or CREDENTIAL_RE.search(text)):
        score += 3
        signals.append("Job/employment scam pattern detected")
    if UNEXPECTED_MONEY_RE.search(text):
        score += 3
        signals.append("Unexpected money, prize, inheritance, or refund pattern detected")
    if BEC_SCAM_RE.search(text):
        score += 4
        signals.append("Business email compromise or invoice-payment change pattern detected")
    if RECOVERY_SCAM_RE.search(text):
        score += 3
        signals.append("Money recovery scam pattern detected")
    if DONATION_SCAM_RE.search(text) and (MONEY_RE.search(text) or PAYMENT_PRESSURE_RE.search(text) or URL_RE.search(text)):
        score += 3
        signals.append("Donation or charity payment pressure pattern detected")
    if (IMPERSONATION_RE.search(text) or re.search(r"\b(aba|acleda|wing|bakong|bank|wallet|telegram|facebook|email)\b", text, re.I)) and ACTION_RE.search(text):
        score += 3
        signals.append("Possible support/admin/bank impersonation with action request")
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
    if (public_notice or HEALTH_NOTICE_RE.search(text)) and not (CREDENTIAL_RE.search(text) or ACCOUNT_INFO_RE.search(text) or DIRECT_PRIVATE_INFO_RE.search(text) or KHMER_CREDENTIAL_RE.search(text) or KHMER_DIRECT_CREDENTIAL_RE.search(text) or URL_RE.search(text) or MONEY_RE.search(text)):
        score = max(score - 4, 0)
        signals.append("Public notice, education, or health-awareness context detected")
    return score, signals


def scam_type(text, signals):
    t = text.lower()
    if BEC_SCAM_RE.search(text):
        return "Business email compromise scam"
    if THREAT_SCAM_RE.search(text):
        return "Threat or extortion scam"
    if RECOVERY_SCAM_RE.search(text):
        return "Money recovery scam"
    if DONATION_SCAM_RE.search(text):
        return "Donation scam"
    if RELATIONSHIP_SCAM_RE.search(text):
        return "Relationship scam"
    if MARKETPLACE_SCAM_RE.search(text):
        return "Buying and selling scam"
    if JOB_SCAM_RE.search(text) or re.search(r"\b(job|salary|recruit|work from home)\b", t, re.I) or re.search(r"(ការងារ|ប្រាក់ខែ)", text):
        return "Jobs and employment scam"
    if UNEXPECTED_MONEY_RE.search(text):
        return "Unexpected money scam"
    if ACCOUNT_TAKEOVER_RE.search(text) or re.search(r"\b(locked|suspended|expired|verify now)\b", t, re.I) or re.search(r"(ចាក់សោ|បិទ|ផុតកំណត់|បន្ទាន់)", text):
        return "Account or identity takeover scam"
    if CREDENTIAL_RE.search(text) or ACCOUNT_INFO_RE.search(text) or DIRECT_PRIVATE_INFO_RE.search(text) or KHMER_CREDENTIAL_RE.search(text) or KHMER_DIRECT_CREDENTIAL_RE.search(text) or "otp" in t:
        return "Password / OTP theft"
    if GAMBLING_RE.search(text) or KHMER_GAMBLING_RE.search(text):
        return "Online gambling / deposit scam"
    if INVESTMENT_RE.search(text) or KHMER_INVESTMENT_RE.search(text):
        return "Investment scam"
    if re.search(r"\b(email quota|mailbox|deactivation|webmail|microsoft|gmail|outlook)\b", t, re.I):
        return "Phishing scam"
    if re.search(r"\b(bank|aba|acleda|wing|bakong|wallet|account|support|security)\b", t, re.I) or re.search(r"(ធនាគារ|ABA|ACLEDA|Wing|Bakong|គណនី|សុវត្ថិភាព)", text):
        return "Fake bank support"
    if re.search(r"\b(prize|reward|winner|bonus)\b", t, re.I) or re.search(r"(រង្វាន់|ឈ្នះ)", text):
        return "Prize or reward scam"
    if re.search(r"\b(delivery|parcel|package|shipping|fee)\b", t, re.I) or re.search(r"(ដឹកជញ្ជូន|កញ្ចប់|ថ្លៃ)", text):
        return "Delivery fee scam"
    if re.search(r"\b(loan|credit|processing fee)\b", t, re.I) or re.search(r"(កម្ចី|ឥណទាន)", text):
        return "Loan scam"
    if signals:
        return "Unknown suspicious message"
    return "No scam type detected"


def hybrid_decision(text, ml_probability, threshold):
    rule_score, signals = rule_risk_score(text)
    protective = any("Educational/protective" in signal for signal in signals)
    public_notice = any("Public notice" in signal for signal in signals)
    if protective and rule_score == 0:
        hybrid_probability = min(ml_probability, 0.20)
    elif public_notice and rule_score <= 2:
        hybrid_probability = min(ml_probability, 0.25)
    else:
        hybrid_probability = max(ml_probability, min(0.99, rule_score / 8))
    prediction = "Scam" if hybrid_probability >= threshold else "Safe"
    if any(signal.startswith("Critical trigger") for signal in signals) and not protective:
        prediction = "Scam"
        hybrid_probability = max(hybrid_probability, 0.95)
        level = "Critical"
    elif rule_score >= 7 and not protective:
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


def prepare_message_for_analysis(text):
    cleaned_lines = []
    normalized_text = normalize_ocr_text(text)
    is_ocr_text = "[English/header OCR]" in normalized_text or "[Khmer message OCR]" in normalized_text
    skip_patterns = [
        r"^\[English/header OCR\]$",
        r"^\[Khmer message OCR\]$",
        r"^Text Message\s*SMS",
        r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+\d{1,2}\s+\w+",
        r"^(Today|Yesterday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        r"^\D{0,8}\d{1,2}:\d{2}\s*(AM|PM|ព្រឹក|ល្ងាច|រសៀល)?\D{0,8}$",
        r"^[\W\d\s#&$|.,'\":;(){}\[\]<>/\\]+$",
        r"^\d{1,4}$",
    ]
    for raw_line in normalize_ocr_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(re.search(pattern, line, re.I) for pattern in skip_patterns):
            continue
        khmer_chars = len(KHMER_RE.findall(line))
        latin_chars = len(re.findall(r"[A-Za-z]", line))
        digit_chars = sum(ch.isdigit() for ch in line)
        if khmer_chars and khmer_chars < 4 and (digit_chars or line_noise_score(line) > 0.35):
            continue
        if is_ocr_text and latin_chars and not KHMER_RE.search(line):
            line = clean_english_ocr_line(line)
            if not english_ocr_line_is_useful(line):
                continue
        if is_ocr_text and line_noise_score(line) > 0.55 and not KHMER_RE.search(line):
            continue
        cleaned_lines.append(line)
    return normalize_ocr_text("\n".join(cleaned_lines)) or normalized_text


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


def risk_summary_text(result):
    if result["prediction"] == "Scam":
        return (
            f"{result['level']} risk {result['category']}. "
            "Do not reply, click links, send money, or share private codes. "
            "Verify through an official app, website, or phone number."
        )
    return (
        "Lower risk. No strong scam pattern was found, but important requests should still "
        "be verified through official channels."
    )


def khmer_explanation_text(result):
    if result["prediction"] == "Scam":
        return (
            "លទ្ធផល៖ សារនេះមានហានិភ័យខ្ពស់។ សូមកុំឆ្លើយតប កុំចុចតំណ "
            "កុំផ្ញើប្រាក់ និងកុំចែករំលែក OTP ពាក្យសម្ងាត់ PIN CVV លេខគណនី "
            "ឬព័ត៌មានឯកជន។ សូមផ្ទៀងផ្ទាត់តាមកម្មវិធីផ្លូវការ ឬលេខទូរសព្ទផ្លូវការ។"
        )
    return (
        "លទ្ធផល៖ សារនេះមើលទៅមានហានិភ័យទាបជាង ប៉ុន្តែបើមានការស្នើសុំប្រាក់ "
        "OTP ពាក្យសម្ងាត់ ឬព័ត៌មានធនាគារ សូមផ្ទៀងផ្ទាត់តាមប្រភពផ្លូវការជានិច្ច។"
    )


def show_similar_scam_examples(message, top_k=5):
    similar = retrieve_similar_scam_tfidf(message, top_k=top_k)
    if similar.empty:
        st.info("No similar scam examples were found in the current retrieval corpus.")
        return
    st.caption("Retrieved with TF-IDF cosine similarity against known scam examples. Higher scores mean closer text patterns.")
    st.dataframe(similar, width="stretch", hide_index=True)


def ocr_quality_warning(text, lang_codes):
    cleaned = normalize_ocr_text(text)
    if not cleaned:
        return "No readable OCR text was extracted. Try manual crop or type the message manually."
    khmer_chars = len(KHMER_RE.findall(cleaned))
    latin_chars = len(re.findall(r"[A-Za-z]", cleaned))
    noisy_lines = sum(1 for line in cleaned.splitlines() if line_noise_score(line) > 0.55 and not KHMER_RE.search(line))
    if "km" in lang_codes and khmer_chars < 12:
        return "Khmer OCR quality looks low. Please crop only the message bubble or manually correct the Khmer text before analysis."
    if noisy_lines >= 2:
        return "OCR contains noisy header/symbol lines. Review the editable text and remove phone UI, keyboard, dates, or sender headers if needed."
    if len(cleaned) < 20 and not (khmer_chars or latin_chars):
        return "OCR output is too short to analyze reliably. Try a tighter crop or type the message manually."
    return ""


def build_user_report(result):
    signals = result["signals"] or ["No strong risk signals found."]
    actions = advice_for(result["level"], result["category"]) if result["prediction"] == "Scam" else [
        "Verify important requests through official channels.",
        "Never share OTP, password, PIN, CVV, seed phrase, or account number.",
    ]
    return "\n".join([
        "Scam Safety Assistant Result",
        f"Prediction: {result['prediction']}",
        f"Risk level: {result['level']}",
        f"Hybrid scam risk: {result['risk']:.1%}",
        f"ML probability: {result['ml_probability']:.1%}",
        f"Rule score: {result['rule_score']}",
        f"Scam type: {result['category']}",
        "",
        "Message analyzed:",
        result["message"],
        "",
        "Detected risk signals:",
        *[f"- {signal}" for signal in signals],
        "",
        "Recommended next steps:",
        *[f"- {item}" for item in actions],
    ])


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


@st.cache_resource
def load_easyocr_reader(lang_tuple):
    import easyocr
    return easyocr.Reader(list(lang_tuple), gpu=False)


def get_tesseract_command():
    candidates = [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def preprocess_variants_for_ocr(image):
    from PIL import ImageFilter, ImageOps

    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    scaled = gray.resize((gray.width * 2, gray.height * 2))
    threshold = scaled.point(lambda px: 255 if px > 175 else 0)
    sharpened = scaled.filter(ImageFilter.SHARPEN)
    return [scaled, sharpened, threshold]


def ocr_quality_score(text, language):
    text = normalize_ocr_text(text)
    if not text:
        return 0
    khmer_chars = len(KHMER_RE.findall(text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    noisy_lines = sum(1 for line in text.splitlines() if line_noise_score(line) > 0.55 and not KHMER_RE.search(line))
    if language == "khm":
        return khmer_chars * 2 - latin_chars - noisy_lines * 12
    return latin_chars + sum(ch.isdigit() for ch in text) - noisy_lines * 8


def resize_for_easyocr(image, max_side=1400):
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((int(image.width * scale), int(image.height * scale)))


def auto_crop_message_bubbles(image):
    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # iOS/Telegram-style message bubbles are usually light gray with low color variance.
    gray_bubble = (
        (r >= 175) & (r <= 248)
        & (g >= 175) & (g <= 248)
        & (b >= 175) & (b <= 248)
        & (np.abs(r.astype(int) - g.astype(int)) < 18)
        & (np.abs(g.astype(int) - b.astype(int)) < 18)
    ).astype("uint8") * 255

    # Remove keyboard/status regions and reduce tiny UI artifacts.
    gray_bubble[: int(h * 0.08), :] = 0
    gray_bubble[int(h * 0.78) :, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9))
    gray_bubble = cv2.morphologyEx(gray_bubble, cv2.MORPH_CLOSE, kernel)
    gray_bubble = cv2.morphologyEx(gray_bubble, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(gray_bubble, connectivity=8)
    boxes = []
    for idx in range(1, count):
        x, y, bw, bh, area = stats[idx]
        if area < w * h * 0.003:
            continue
        if bh < h * 0.025 or bw < w * 0.14:
            continue
        if y < h * 0.06 or y + bh > h * 0.78:
            continue
        # Avoid huge background blocks and the input bar/keyboard region.
        if bw > w * 0.85 or bh > h * 0.28:
            continue
        pad = max(8, int(min(w, h) * 0.018))
        boxes.append((
            max(0, x - pad),
            max(0, y - pad),
            min(w, x + bw + pad),
            min(h, y + bh + pad),
        ))

    boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    if not boxes:
        return image, 0

    crops = [image.crop(box) for box in boxes]
    canvas_width = max(crop.width for crop in crops) + 24
    canvas_height = sum(crop.height for crop in crops) + 20 * (len(crops) + 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    y_cursor = 20
    for crop in crops:
        canvas.paste(crop, (12, y_cursor))
        y_cursor += crop.height + 20
    return canvas, len(crops)


def extract_text_with_tesseract(image, language):
    import pytesseract

    tesseract_cmd = get_tesseract_command()
    if not tesseract_cmd:
        return "", "Tesseract OCR is not installed."
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    config_parts = []
    if language == "khm":
        if not (TESSDATA_DIR / "khm.traineddata").exists():
            return "", "Khmer Tesseract language data `khm.traineddata` is missing."
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    elif "TESSDATA_PREFIX" in os.environ:
        os.environ.pop("TESSDATA_PREFIX", None)

    candidates = []
    psm_values = [6, 11, 4] if language == "khm" else [6, 11]
    for variant in preprocess_variants_for_ocr(image):
        for psm in psm_values:
            text = pytesseract.image_to_string(
                variant,
                lang=language,
                config=" ".join([*config_parts, f"--psm {psm}"]),
            )
            text = normalize_ocr_text(text)
            candidates.append((ocr_quality_score(text, language), text))
    best = max(candidates, key=lambda item: item[0])[1] if candidates else ""
    return best, None


def line_noise_score(line):
    line = line.strip()
    if not line:
        return 1.0
    useful = sum(ch.isalnum() or bool(KHMER_RE.match(ch)) for ch in line)
    symbols = sum(not ch.isspace() and not ch.isalnum() and not bool(KHMER_RE.match(ch)) for ch in line)
    return symbols / max(useful + symbols, 1)


def english_ocr_line_is_useful(line):
    lower = line.lower()
    if re.search(r"\b(nodrug|otp|password|pin|cvv|aba|acleda|wing|bakong|bank|telegram|facebook|gmail|outlook|http|www|casino|loan|job|prize|reward|verify|account|delivery|parcel|hiv|aids)\b", lower):
        return True
    common_words = {
        "the", "and", "for", "you", "your", "from", "with", "this", "that", "message",
        "account", "support", "security", "verify", "click", "pay", "send", "share",
        "today", "tomorrow", "warning", "official", "public", "health", "education",
    }
    words = re.findall(r"[A-Za-z]{2,}", line)
    if not words:
        return False
    if sum(word.lower() in common_words for word in words) < 2:
        return False
    avg_len = sum(len(word) for word in words) / len(words)
    vowel_ratio = sum(ch in "aeiouAEIOU" for ch in "".join(words)) / max(sum(len(word) for word in words), 1)
    dictionary_like = sum(bool(re.search(r"[aeiouAEIOU]", word)) and len(word) <= 14 for word in words)
    return avg_len <= 12 and vowel_ratio >= 0.22 and dictionary_like >= max(1, len(words) // 2)


def clean_english_ocr_line(line):
    known = re.findall(
        r"\b(nodrug|otp|password|pin|cvv|aba|acleda|wing|bakong|bank|telegram|facebook|gmail|outlook|https?://\S+|www\.\S+|casino|loan|job|prize|reward|verify|account|delivery|parcel|hiv|aids)\b",
        line,
        flags=re.I,
    )
    if known:
        unique = []
        seen = set()
        for item in known:
            key = item.lower()
            if key not in seen:
                unique.append(item)
                seen.add(key)
        return " ".join(unique)
    return line


def clean_ocr_lines(text, keep="mixed"):
    cleaned = []
    seen = set()
    for raw_line in normalize_ocr_text(text).splitlines():
        line = raw_line.strip(" -|•·")
        if not line:
            continue
        has_khmer = bool(KHMER_RE.search(line))
        has_latin = bool(re.search(r"[A-Za-z]", line))
        digit_count = sum(ch.isdigit() for ch in line)
        if keep == "khmer" and not has_khmer:
            continue
        if keep == "english" and not (has_latin or digit_count >= 2):
            continue
        if keep == "english" and has_latin and not english_ocr_line_is_useful(line):
            continue
        if keep == "english" and has_latin:
            line = clean_english_ocr_line(line)
        if line_noise_score(line) > 0.45 and not has_khmer:
            continue
        if len(line) < 3 and not digit_count:
            continue
        key = re.sub(r"\s+", " ", line).lower()
        if key in seen:
            continue
        cleaned.append(line)
        seen.add(key)
    return "\n".join(cleaned).strip()


def merge_ocr_results(khmer_text="", english_text=""):
    khmer_clean = clean_ocr_lines(khmer_text, keep="khmer")
    english_clean = clean_ocr_lines(english_text, keep="english")
    parts = []
    if english_clean:
        parts.append("[English/header OCR]\n" + english_clean)
    if khmer_clean:
        parts.append("[Khmer message OCR]\n" + khmer_clean)
    return normalize_ocr_text("\n\n".join(parts))


def extract_text_from_image(uploaded_file, lang_codes):
    easyocr_lang_codes = [code for code in lang_codes if code != "km"]
    khmer_requested = "km" in lang_codes
    if not easyocr_lang_codes:
        easyocr_lang_codes = ["en"]
    khmer_text = ""
    english_text = ""
    warnings = []
    try:
        from PIL import Image
        import numpy as np

        image = Image.open(uploaded_file).convert("RGB")

        if khmer_requested:
            khmer_text, khmer_error = extract_text_with_tesseract(image, "khm")
            if khmer_error:
                warnings.append(khmer_error)

        if easyocr_lang_codes:
            try:
                reader = load_easyocr_reader(tuple(easyocr_lang_codes))
                easy_image = resize_for_easyocr(image)
                lines = reader.readtext(np.array(easy_image), detail=0, paragraph=True)
                english_text = normalize_ocr_text("\n".join(str(line) for line in lines))
            except Exception:
                english_text, english_error = extract_text_with_tesseract(image, "eng")
                if english_error:
                    warnings.append(english_error)

        text = merge_ocr_results(khmer_text, english_text)
        if khmer_requested and not KHMER_RE.search(text):
            warnings.append("Khmer OCR was attempted with Tesseract, but no clear Khmer text was extracted. Please correct the text manually if needed.")
        return text, " ".join(warnings) if warnings else None
    except Exception as easyocr_error:
        try:
            from PIL import Image

            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")
            if khmer_requested:
                text, tesseract_error = extract_text_with_tesseract(image, "khm")
            else:
                text, tesseract_error = extract_text_with_tesseract(image, "eng")
            return text, tesseract_error
        except Exception as tesseract_error:
            return "", (
                "OCR could not extract text. For Khmer screenshots, type/correct the message manually in the text box and analyze it. "
                f"EasyOCR error: {easyocr_error}. "
                f"Tesseract error: {tesseract_error}."
            )


def normalize_ocr_text(text):
    text = str(text or "")
    text = text.replace("\x0c", " ").replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(r"[ \t\r]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


model = load_model()
safe_words, scam_words = load_wordlists()

st.markdown("""
<div class="hero">
  <div class="hero-grid">
    <div class="hero-copy">
      <h1>Scam Safety Assistant</h1>
      <p>Khmer-English message, email, screenshot, and batch screening for real-world scam prevention.</p>
      <div class="hero-badges">
        <span class="badge">Hybrid ML + rules</span>
        <span class="badge">Khmer-English support</span>
        <span class="badge">OCR workflow</span>
        <span class="badge">Safety guidance</span>
      </div>
    </div>
    <div class="hero-panel">
      <b>Critical override</b>
      <span>OTP, PIN, password, account data, payment pressure</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="privacy"><b>Privacy warning:</b> This app does not need your real password, OTP, bank account number, card number, or private code. Never paste real sensitive information.</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="home-grid">
  <div class="home-card">
    <div class="result-title">Home / About</div>
    <p class="muted">This project demonstrates Information Retrieval and text analytics for real-life scam awareness. It screens SMS, chat, email, and OCR-extracted screenshot text, then explains the risk in practical language.</p>
    <p><b>Purpose:</b> help people notice suspicious requests before they reply, click links, send money, or share private codes.</p>
    <p><b>Important limitation:</b> the dataset is useful for coursework and portfolio demonstration, but it is not enough for production because real scams change over time.</p>
  </div>
  <div class="home-card">
    <div class="result-title">Responsible Use</div>
    <p><b>Privacy:</b> do not paste real OTP, password, PIN, CVV, seed phrase, bank account number, or identity document details.</p>
    <p><b>Decision support:</b> this tool supports awareness and education. It is not legal, banking, or law-enforcement advice.</p>
    <p><b>Verify safely:</b> contact banks, companies, or public services only through official apps, websites, or phone numbers.</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Scam Safety")
    st.markdown("**Risk policy**")
    st.markdown("- Critical requests override the ML model.")
    st.markdown("- Safe education messages are separated from dangerous requests.")
    st.markdown("- OCR text should be reviewed before analysis.")
    st.markdown("- IR search retrieves similar scam patterns.")
    st.markdown("---")
    st.markdown("**Covered channels**")
    st.markdown("- SMS and chat")
    st.markdown("- Email phishing")
    st.markdown("- Screenshots")
    st.markdown("- Similarity search")
    st.markdown("- Batch CSV checks")
    st.markdown("---")
    st.caption("Portfolio-ready prototype for scam awareness and message triage.")

tabs = st.tabs(["Analyze", "Scam Type", "Risk Signals", "Next Steps", "Khmer Examples", "Feedback", "IR Explorer", "OCR", "Batch"])

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "demo_message" not in st.session_state:
    st.session_state.demo_message = "Dear customer, send your OTP, PIN, and account number to support-check@gmail.com to verify your account today."
if "ocr_text_area" not in st.session_state:
    st.session_state.ocr_text_area = ""
if "ocr_last_debug" not in st.session_state:
    st.session_state.ocr_last_debug = ""

with tabs[0]:
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        default_message = "សូមបញ្ចូលលេខគណនី និងពាក្យសម្ងាត់ ដើម្បីទទួលរង្វាន់"
        st.markdown("#### Message analysis")
        st.caption("Try a safe synthetic demo before pasting your own text.")
        demo_messages = {
            "OTP scam": "Dear customer, send your OTP, PIN, and account number to support-check@gmail.com to verify your account today.",
            "Safe OTP education": "Never share OTP, password, PIN, CVV, or bank account number with anyone, including bank staff.",
            "Khmer scam": "សូមផ្ញើលេខគណនី និង OTP ដើម្បីបញ្ជាក់រង្វាន់",
            "Delivery scam": "Your parcel is delayed. Pay $1.25 delivery fee now at http://tinyurl.example/parcel",
            "Job scam": "Remote job approved. Pay 15 USD registration fee and submit your bank details to start daily commission.",
            "Business email scam": "CEO request: urgently wire 18,750 USD to this new supplier account today and keep it confidential.",
        }
        demo_cols = st.columns(3)
        for idx, (label, sample) in enumerate(demo_messages.items()):
            if demo_cols[idx % 3].button(label, width="stretch"):
                st.session_state.demo_message = sample
        message = st.text_area(
            "Paste a suspicious message",
            value=st.session_state.demo_message or default_message,
            height=170,
            placeholder="Paste Khmer or English SMS, chat, or email text here...",
        )
        threshold = st.slider("Scam warning threshold", 0.10, 0.90, 0.50, 0.05)
        run = st.button("Analyze message", type="primary", width="stretch")
    with right:
        st.markdown(
            """
<div class="pro-card">
  <div class="result-title">Never share</div>
  <p class="muted">Keep these private, even if a message claims to be from support, bank staff, or an official service.</p>
  <ul>
    <li>Password / ពាក្យសម្ងាត់</li>
    <li>OTP / verification code</li>
    <li>PIN, CVV, card number</li>
    <li>Bank account number</li>
    <li>Seed phrase / private key</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    if run and message.strip():
        analysis_message = prepare_message_for_analysis(message)
        features = build_features(analysis_message, safe_words, scam_words)
        ml_probability = float(model.predict_proba(features)[0, 1])
        prediction, risk, rule_score, signals, level, category = hybrid_decision(analysis_message, ml_probability, threshold)
        st.session_state.last_result = {
            "message": analysis_message, "original_message": message, "prediction": prediction, "risk": risk, "rule_score": rule_score,
            "signals": signals, "level": level, "category": category, "ml_probability": ml_probability,
        }

    result = st.session_state.last_result
    if result:
        css_class = {"Critical": "critical", "High": "high", "Medium": "medium", "Low": "low"}[result["level"]]
        fill_color = {"Critical": "#991B1B", "High": "#DC2626", "Medium": "#F59E0B", "Low": "#16A34A"}[result["level"]]
        critical_override = any(signal.startswith("Critical trigger") for signal in result["signals"])
        similar_for_confidence = retrieve_similar_scam_tfidf(result["message"], top_k=1)
        top_similarity = float(similar_for_confidence["similarity"].iloc[0]) if not similar_for_confidence.empty else 0.0
        decision_label = f"{result['level']} Scam Risk" if result["prediction"] == "Scam" else "Lower Risk Message"
        decision_class = "critical-card" if result["prediction"] == "Scam" else "safe-card"
        st.markdown(
            f"""
<div class="decision-card {decision_class}">
  <div class="decision-kicker">Final decision</div>
  <div class="decision-title">{html.escape(decision_label)}</div>
  <div class="muted">{html.escape(risk_summary_text(result))}</div>
  <div class="risk-bar"><div class="risk-fill" style="width:{result['risk'] * 100:.1f}%; background:{fill_color};"></div></div>
  <div class="decision-meta">
    <span class="decision-pill">Type: {html.escape(result["category"])}</span>
    <span class="decision-pill">Risk score: {result["risk"]:.1%}</span>
    <span class="decision-pill">Critical override: {"Yes" if critical_override else "No"}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="metric-grid">
  <div class="metric-card"><div class="label">Prediction</div><div class="value">{html.escape(result["prediction"])}</div><div class="sub">Final assistant decision</div></div>
  <div class="metric-card"><div class="label">Risk level</div><div class="value">{html.escape(result["level"])}</div><div class="sub">Safety-first severity</div></div>
  <div class="metric-card"><div class="label">Hybrid scam risk</div><div class="value">{result["risk"]:.1%}</div><div class="sub">ML {result["ml_probability"]:.1%} + rules {result["rule_score"]}</div></div>
  <div class="metric-card"><div class="label">Scam type</div><div class="value">{html.escape(result["category"])}</div><div class="sub">Best matching scenario</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="status-card">
  <div class="result-title">{html.escape(risk_summary_text(result))}</div>
  <div class="muted">Hybrid score combines ML probability, critical scam rules, Khmer/English risk phrases, and safe-context checks.</div>
  <div class="risk-bar"><div class="risk-fill" style="width:{result['risk'] * 100:.1f}%; background:{fill_color};"></div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        if result["prediction"] == "Scam":
            st.markdown(f'<div class="{css_class}"><b>{result["level"]} risk:</b> This message may be dangerous. Follow the safety steps before taking action.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="low"><b>Lower risk:</b> This message looks safer, but still verify important requests through official channels.</div>', unsafe_allow_html=True)
        st.subheader("Highlighted message")
        st.markdown(f'<div class="khmer-box">{highlight_message(result["message"])}</div>', unsafe_allow_html=True)

        show_khmer = st.toggle("Show Khmer explanation", value=True)
        if show_khmer:
            st.markdown(f'<div class="khmer-box">{html.escape(khmer_explanation_text(result))}</div>', unsafe_allow_html=True)

        st.markdown("### Why risky?")
        if result["signals"]:
            for signal in result["signals"]:
                st.warning(signal)
        else:
            st.success("No strong scam trigger was detected. Continue to verify any request involving money or private information.")

        st.markdown("### Confidence explanation")
        st.markdown(
            f"""
<div class="pro-card">
  <b>How the assistant decided:</b>
  <ul>
    <li>ML probability: <b>{result["ml_probability"]:.1%}</b></li>
    <li>Rule score: <b>{result["rule_score"]}</b></li>
    <li>Critical override used: <b>{"Yes" if critical_override else "No"}</b></li>
    <li>Top TF-IDF retrieval similarity: <b>{top_similarity:.2f}</b></li>
  </ul>
  <p class="muted">The final decision is safety-first: direct requests for OTP, password, PIN, CVV, account number, seed phrase, or urgent payment can override the ML model.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### Similar scam examples")
        show_similar_scam_examples(result["message"], top_k=5)

        st.markdown("### What should I do?")
        actions = advice_for(result["level"], result["category"]) if result["prediction"] == "Scam" else [
            "Verify important requests through official channels.",
            "Never share OTP, password, PIN, CVV, seed phrase, or account number.",
            "If unsure, ask a trusted person or contact the organization directly."
        ]
        for item in actions:
            st.write(f"- {item}")

        st.markdown("### Report / save result")
        st.caption("Save this result for your own record or to report the sender. Do not include real OTP, password, or bank secrets.")
        st.download_button(
            "Download safety report",
            data=build_user_report(result),
            file_name="scam_safety_result.txt",
            mime="text/plain",
            width="stretch",
        )

with tabs[1]:
    result = st.session_state.last_result
    st.subheader("Scam Type Classification")
    if result:
        st.metric("Detected type", result["category"])
        st.write("This category is based on message content, risk rules, Khmer/English scam terms, and model signals.")
    else:
        st.info("Analyze a message first.")
    st.dataframe(pd.DataFrame([
        {"type": "Phishing scam", "description": "Fake email/SMS login, mailbox, bank, or website verification message."},
        {"type": "Relationship scam", "description": "Builds emotional trust, then asks for money, gift cards, or secrecy."},
        {"type": "Investment scam", "description": "Promises guaranteed returns, crypto profit, trading gains, or withdrawal after fees."},
        {"type": "Account or identity takeover scam", "description": "Tries to steal login codes, identity data, account access, or SIM/profile control."},
        {"type": "Buying and selling scam", "description": "Marketplace, buyer/seller, escrow, shipping, fee, or payment-release trick."},
        {"type": "Threat or extortion scam", "description": "Uses blackmail, legal pressure, arrest threats, hacked-phone claims, or penalties."},
        {"type": "Jobs and employment scam", "description": "Fake job/task offer requiring registration fee, deposit, documents, or bank details."},
        {"type": "Unexpected money scam", "description": "Inheritance, prize, refund, transfer, or government cash requiring fee/details."},
        {"type": "Business email compromise scam", "description": "CEO/vendor/invoice impersonation requesting wire transfer or bank-account change."},
        {"type": "Money recovery scam", "description": "Claims to recover stolen funds but asks for upfront fee, tax, or wallet validation."},
        {"type": "Donation scam", "description": "Fake charity/fundraiser/disaster request with suspicious payment pressure."},
        {"type": "Password / OTP theft", "description": "Requests password, OTP, PIN, CVV, account number, seed phrase."},
        {"type": "Fake bank support", "description": "Pretends to be bank/security/support and asks for action."},
        {"type": "Prize or reward scam", "description": "Promises money/reward and asks for details or payment."},
        {"type": "Delivery fee scam", "description": "Asks to pay delivery/customs fee through link."},
        {"type": "Loan scam", "description": "Promises loan approval and asks for fee or documents."},
        {"type": "Online gambling / deposit scam", "description": "Casino, betting, deposit bonus, app download, or gambling promotion."},
    ]), width="stretch")

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
    st.dataframe(SCAM_EXAMPLES, width="stretch")
    st.markdown("#### Safe vs scam comparison")
    st.write("- Safe: `កុំផ្ញើលេខសម្ងាត់ ឬ OTP ទៅអ្នកណាម្នាក់`")
    st.write("- Scam: `សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នកដើម្បីបញ្ជាក់`")

with tabs[5]:
    st.subheader("Report / Feedback")
    result = st.session_state.last_result
    if result:
        st.write("Help improve this project by marking the prediction.")
        f1, f2, f3 = st.columns(3)
        if f1.button("Correct", width="stretch"):
            save_feedback(result["message"], result["prediction"], result["risk"], result["category"], "correct")
            st.success("Saved feedback: correct.")
        if f2.button("Wrong, this is scam", width="stretch"):
            save_feedback(result["message"], result["prediction"], result["risk"], result["category"], "wrong_scam")
            st.success("Saved feedback: wrong, true label scam.")
        if f3.button("Wrong, this is safe", width="stretch"):
            save_feedback(result["message"], result["prediction"], result["risk"], result["category"], "wrong_safe")
            st.success("Saved feedback: wrong, true label safe.")
        if FEEDBACK_PATH.exists():
            st.download_button("Download feedback CSV", FEEDBACK_PATH.read_bytes(), "user_feedback.csv", "text/csv")
    else:
        st.info("Analyze a message first.")

with tabs[6]:
    st.subheader("Information Retrieval Explorer")
    st.markdown(
        "Compare a new Khmer/English message against known scam examples. "
        "TF-IDF cosine similarity finds lexical overlap, while multilingual embeddings can find semantic similarity."
    )
    default_ir_query = ""
    if st.session_state.last_result:
        default_ir_query = st.session_state.last_result.get("message", "")
    ir_query = st.text_area(
        "Message for similarity search",
        value=default_ir_query,
        height=130,
        placeholder="Paste a message to retrieve similar scam examples...",
        key="ir_query",
    )
    ir_top_k = st.slider("Number of similar scam examples", 3, 10, 5, 1)
    run_ir = st.button("Retrieve similar scam messages", type="primary", width="stretch")

    if run_ir and ir_query.strip():
        cleaned_query = prepare_message_for_analysis(ir_query)
        tfidf_results = retrieve_similar_scam_tfidf(cleaned_query, top_k=ir_top_k)
        st.markdown("#### TF-IDF Cosine Similarity")
        st.caption("Lexical retrieval: useful when the message shares words, phrases, URLs, numbers, or Khmer/English character patterns with known scams.")
        if tfidf_results.empty:
            st.info("No TF-IDF results found.")
        else:
            st.dataframe(tfidf_results, width="stretch")

        st.markdown("#### Multilingual Embedding Search")
        st.caption("Semantic retrieval: useful when the wording is different but the meaning is similar. Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` when installed.")
        embedding_results, embedding_error = retrieve_similar_scam_embeddings(cleaned_query, top_k=ir_top_k)
        if embedding_error:
            st.warning(
                "Embedding search is optional and is not ready in this Python environment. "
                "Install dependencies with `python -m pip install sentence-transformers torch`, then download/cache "
                "`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` once. "
                f"Details: {embedding_error}"
            )
        elif embedding_results.empty:
            st.info("No embedding results found.")
        else:
            st.dataframe(embedding_results, width="stretch")

        st.markdown("#### How to interpret the scores")
        st.write("- Higher similarity means the message is closer to known scam examples in the retrieval corpus.")
        st.write("- TF-IDF cosine similarity is strongest for shared words and character patterns.")
        st.write("- Embedding similarity is strongest for shared meaning, including paraphrases and mixed Khmer-English wording.")
        st.write("- Retrieval is not the final decision by itself; it supports the hybrid classifier and safety explanation.")
    else:
        st.info("Enter a message or analyze one first, then click the retrieval button.")

with tabs[7]:
    st.subheader("OCR Image Check")
    st.markdown(
        "Upload a screenshot of an SMS, Telegram/Facebook chat, or scam message image. "
        "The app will extract Khmer/English text, let you edit it, then analyze it."
    )
    st.markdown(
        """
<div class="pro-card">
  <div class="result-title">OCR quality warning</div>
  <ul>
    <li>OCR may be wrong, especially for Khmer screenshots.</li>
    <li>Crop only the message bubble whenever possible.</li>
    <li>Always correct the extracted text before analysis.</li>
  </ul>
  <p class="muted">Keyboard, status bar, sender names, dates, and faded old messages can reduce OCR accuracy.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="privacy"><b>Privacy warning:</b> Do not upload screenshots containing real OTP, password, bank account number, card number, or private code.</div>',
        unsafe_allow_html=True,
    )

    ocr_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], key="ocr_upload")
    lang_choice = st.multiselect("OCR languages", ["English", "Khmer"], default=["English", "Khmer"])
    lang_codes = []
    if "English" in lang_choice:
        lang_codes.append("en")
    if "Khmer" in lang_choice:
        lang_codes.append("km")
    if not lang_codes:
        lang_codes = ["en"]

    if ocr_file:
        from PIL import Image

        uploaded_image = Image.open(ocr_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded image", width="stretch")

        image_for_ocr = uploaded_image
        crop_mode = st.radio(
            "OCR focus mode",
            ["Auto detect message bubbles", "Manual crop", "Full image"],
            horizontal=True,
            help="Auto mode detects gray chat bubbles and removes keyboard/status/header noise. Use manual crop if auto misses a message.",
        )
        if crop_mode == "Auto detect message bubbles":
            image_for_ocr, bubble_count = auto_crop_message_bubbles(uploaded_image)
            if bubble_count:
                st.success(f"Auto detected {bubble_count} message bubble area(s). OCR will focus on these bubbles.")
                st.image(image_for_ocr, caption="Auto-detected message bubbles used for OCR", width="stretch")
            else:
                st.warning("Could not auto-detect message bubbles. Try Manual crop.")
        elif crop_mode == "Manual crop":
            st.caption("Adjust crop if OCR reads the keyboard, status bar, or old faded messages.")
            ctop, cbottom = st.columns(2)
            cleft, cright = st.columns(2)
            top_pct = ctop.slider("Crop top (%)", 0, 70, 12, 1)
            bottom_pct = cbottom.slider("Crop bottom (%)", 0, 70, 42, 1)
            left_pct = cleft.slider("Crop left (%)", 0, 40, 2, 1)
            right_pct = cright.slider("Crop right (%)", 0, 40, 4, 1)
            w, h = uploaded_image.size
            left = int(w * left_pct / 100)
            top = int(h * top_pct / 100)
            right = int(w * (100 - right_pct) / 100)
            bottom = int(h * (100 - bottom_pct) / 100)
            if right > left and bottom > top:
                image_for_ocr = uploaded_image.crop((left, top, right, bottom))
                st.image(image_for_ocr, caption="Cropped area used for OCR", width="stretch")

        if st.button("Extract text from image", type="primary", width="stretch"):
            image_buffer = io.BytesIO()
            image_for_ocr.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            extracted, error = extract_text_from_image(image_buffer, lang_codes)
            extracted = normalize_ocr_text(extracted)
            st.session_state.ocr_last_debug = extracted
            if error:
                st.warning(error)
            if extracted.strip():
                st.session_state.ocr_text_area = extracted
                st.success(f"OCR text extracted ({len(extracted)} characters). Please review and correct it before analysis.")
                quality_note = ocr_quality_warning(extracted, lang_codes)
                if quality_note:
                    st.warning(quality_note)
                else:
                    st.info("OCR quality looks usable, but manual review is still required before analysis.")
            else:
                st.info("No text was extracted. You can still type the Khmer/English message manually below and analyze it.")

    if st.session_state.ocr_last_debug:
        with st.expander("Show raw OCR result"):
            st.code(st.session_state.ocr_last_debug)

    ocr_text = st.text_area(
        "Extracted text / editable message",
        height=160,
        key="ocr_text_area",
    )
    if st.button("Analyze OCR text", width="stretch") and ocr_text.strip():
        analysis_text = prepare_message_for_analysis(ocr_text)
        features = build_features(analysis_text, safe_words, scam_words)
        ml_probability = float(model.predict_proba(features)[0, 1])
        prediction, risk, rule_score, signals, level, category = hybrid_decision(analysis_text, ml_probability, 0.50)
        st.session_state.last_result = {
            "message": analysis_text, "original_message": ocr_text, "prediction": prediction, "risk": risk, "rule_score": rule_score,
            "signals": signals, "level": level, "category": category, "ml_probability": ml_probability,
        }
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", prediction)
        c2.metric("Risk level", level)
        c3.metric("Hybrid scam risk", f"{risk:.1%}")
        c4.metric("Scam type", category)
        st.markdown(f'<div class="khmer-box">{highlight_message(analysis_text)}</div>', unsafe_allow_html=True)
        if analysis_text != normalize_ocr_text(ocr_text):
            with st.expander("Show original OCR text before cleanup"):
                st.code(ocr_text)
        if signals:
            for signal in signals:
                st.warning(signal)
        else:
            st.success("No strong risk signals found.")

with tabs[8]:
    st.subheader("Batch Check")
    uploaded = st.file_uploader("Upload CSV with a text column", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        if "text" not in batch.columns:
            st.error("CSV must contain a text column.")
        else:
            clean_texts = [prepare_message_for_analysis(str(t)) for t in batch["text"].fillna("")]
            rows = [build_features(text, safe_words, scam_words) for text in clean_texts]
            features = pd.concat(rows, ignore_index=True)
            probs = model.predict_proba(features)[:, 1]
            analyzed = [hybrid_decision(text, float(prob), 0.50) for text, prob in zip(clean_texts, probs)]
            result_df = batch.copy()
            result_df["cleaned_text"] = clean_texts
            result_df["ml_probability"] = probs
            result_df["hybrid_scam_risk"] = [a[1] for a in analyzed]
            result_df["risk_level"] = [a[4] for a in analyzed]
            result_df["scam_type"] = [a[5] for a in analyzed]
            result_df["prediction"] = [a[0].lower() for a in analyzed]
            st.dataframe(result_df, width="stretch")
            st.download_button("Download scored CSV", result_df.to_csv(index=False).encode("utf-8"), "scored_messages.csv", "text/csv")

st.markdown(
    """
<div class="footer">
  <b>SITHAN Sitana</b> | Master of Data Science | Information Retrieval and Analytics<br>
  GitHub / portfolio link: add your public repository URL before deployment.
</div>
""",
    unsafe_allow_html=True,
)
