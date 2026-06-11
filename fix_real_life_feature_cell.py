from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).resolve().parent / "Scam_or_Not_Scam_IR_Project.ipynb"

nb = nbformat.read(NOTEBOOK, as_version=4)

old_marker = "def add_simple_features(df):"
new_source = r'''
critical_terms = re.compile(
    r"\b(otp|password|pin|cvv|account number|account information|seed phrase|private key|verification code|login code)\b",
    re.I,
)
request_terms = re.compile(r"\b(send|give|share|provide|enter|submit|reply|verify|confirm|update|pay|click|claim|login|transfer)\b", re.I)
safe_context = re.compile(r"\b(never|do not|don't|dont|avoid|warning|education|awareness|protect|will never|will not|does not ask)\b", re.I)
scam_terms = re.compile(
    r"\b(locked|urgent|prize|reward|delivery fee|training fee|commission|guaranteed return|crypto|wire|vendor bank|recover your money|donation|gift card|blackmail|pay)\b",
    re.I,
)
urgent_terms_re = re.compile(r"\b(urgent|immediately|now|today|limited|final|expire|expired|asap|suspended|locked|verify now)\b", re.I)
action_terms_re = re.compile(r"\b(click|claim|verify|confirm|reply|call|text|login|update|send|transfer|provide|share|submit|enter|reset|pay)\b", re.I)
credential_terms_re = re.compile(r"\b(password|passcode|otp|pin|code|verification code|login code|username|account number|bank detail|card number|cvv|private key|seed phrase)\b", re.I)
social_terms_re = re.compile(r"\b(account|bank|wallet|paypal|telegram|whatsapp|facebook|email|security|support|admin|prize|reward|refund|delivery|parcel|loan|job|investment)\b", re.I)
no_re = re.compile(r"\bno\b", re.I)
pronoun_re = re.compile(r"\b(i|me|my|you|your)\b", re.I)
khmer_re = re.compile(r"[\u1780-\u17FF]")
khmer_critical = re.compile(r"(OTP|PIN|CVV|ពាក្យសម្ងាត់|លេខសម្ងាត់|លេខគណនី|គណនី|កូដ|លេខកូដ)")
khmer_request = re.compile(r"(ផ្ញើ|ផ្តល់|បញ្ចូល|បញ្ជាក់|ប្រាប់|ចែករំលែក|បង់|ផ្ទេរ|ចុច|ឆ្លើយតប)")
khmer_safe = re.compile(r"(កុំ|មិនត្រូវ|មិនគួរ|ប្រុងប្រយ័ត្ន|ការពារ|ព្រមាន)")
khmer_social = re.compile(r"(ធនាគារ|ABA|ACLEDA|Wing|Bakong|គណនី|សុវត្ថិភាព|រង្វាន់|ប្រាក់|កម្ចី|ការងារ|វិនិយោគ|ដឹកជញ្ជូន|កញ្ចប់|បន្ទាន់)")

def count_matches(pattern, text):
    return len(pattern.findall(str(text)))

def rule_only_predict(text):
    text = str(text)
    is_safe_context = bool(safe_context.search(text) or khmer_safe.search(text))
    critical_request = (
        bool(critical_terms.search(text) and request_terms.search(text))
        or bool(khmer_critical.search(text) and khmer_request.search(text))
    )
    if critical_request and not is_safe_context:
        return "scam"
    if bool(scam_terms.search(text) and request_terms.search(text)) and not is_safe_context:
        return "scam"
    return "safe"

def add_simple_features(df):
    out = pd.DataFrame({"text": df["text"].astype(str)})
    out["char_count"] = out["text"].str.len()
    out["word_count"] = out["text"].str.split().str.len()
    out["avg_word_len"] = out["text"].apply(lambda x: sum(len(t) for t in x.split()) / max(len(x.split()), 1))
    out["digit_count"] = out["text"].apply(lambda x: sum(ch.isdigit() for ch in x))
    out["uppercase_ratio"] = out["text"].apply(lambda x: sum(ch.isupper() for ch in x) / max(sum(ch.isalpha() for ch in x), 1))
    out["exclamation_count"] = out["text"].str.count("!")
    out["question_count"] = out["text"].str.count(r"\?")
    out["has_exclamation"] = (out["exclamation_count"] > 0).astype(int)
    out["contains_no"] = out["text"].apply(lambda x: int(bool(no_re.search(x))))
    out["first_second_pronoun_count"] = out["text"].apply(lambda x: count_matches(pronoun_re, x))
    out["log_length"] = out["char_count"].apply(lambda value: __import__("math").log1p(value))
    out["has_url"] = out["text"].str.contains(r"https?://|www\.|\.com|bit\.ly|tinyurl", case=False, regex=True).astype(int)
    out["has_phone"] = out["text"].str.contains(r"(?:\+?\d[\s\-().]*){7,}", regex=True).astype(int)
    out["has_money"] = out["text"].str.contains(r"\$|usd|cash|fee|payment|loan|prize|bonus|refund", case=False, regex=True).astype(int)
    out["has_urgency"] = out["text"].apply(lambda x: int(bool(urgent_terms_re.search(x))))
    out["has_action"] = out["text"].apply(lambda x: int(bool(action_terms_re.search(x))))
    out["urgent_terms"] = out["text"].apply(lambda x: count_matches(urgent_terms_re, x))
    out["action_terms"] = out["text"].apply(lambda x: count_matches(action_terms_re, x))
    out["request_terms"] = out["text"].apply(lambda x: count_matches(request_terms, x))
    out["credential_terms"] = out["text"].apply(lambda x: count_matches(credential_terms_re, x))
    out["credential_request"] = out["text"].apply(lambda x: int(bool(credential_terms_re.search(x) and request_terms.search(x))))
    out["has_credential_request"] = out["credential_request"]
    out["social_engineering_terms"] = out["text"].apply(lambda x: count_matches(social_terms_re, x))
    out["khmer_chars"] = out["text"].apply(lambda x: len(khmer_re.findall(str(x))))
    out["khmer_credential_terms"] = out["text"].apply(lambda x: count_matches(khmer_critical, x))
    out["khmer_request_terms"] = out["text"].apply(lambda x: count_matches(khmer_request, x))
    out["khmer_social_engineering_terms"] = out["text"].apply(lambda x: count_matches(khmer_social, x))
    out["scam_word_hits"] = out["text"].str.lower().str.count(r"otp|password|pin|cvv|fee|prize|verify|locked|account|urgent|reward|loan|job|investment")
    out["safe_word_hits"] = out["text"].str.lower().str.count(r"never|avoid|warning|protect|safe|education|reminder")
    out["scam_word_density"] = out["scam_word_hits"] / out["word_count"].clip(lower=1)
    out["safe_word_density"] = out["safe_word_hits"] / out["word_count"].clip(lower=1)
    return out

model = joblib.load(MODEL_PATH)
real_life_features = add_simple_features(real_life_tests)
ml_prob = model.predict_proba(real_life_features)[:, 1]
real_life_tests["ml_probability"] = ml_prob
real_life_tests["ml_only"] = ["scam" if p >= 0.50 else "safe" for p in ml_prob]
real_life_tests["rule_only"] = real_life_tests["text"].apply(rule_only_predict)
real_life_tests["hybrid"] = [
    "scam" if rule == "scam" or prob >= 0.50 else "safe"
    for rule, prob in zip(real_life_tests["rule_only"], real_life_tests["ml_probability"])
]
real_life_tests["hybrid_correct"] = real_life_tests["hybrid"].eq(real_life_tests["expected"])
real_life_tests[["case_type", "expected", "ml_probability", "ml_only", "rule_only", "hybrid", "hybrid_correct", "text"]]
'''.strip()

replaced = 0
for cell in nb.cells:
    if cell.cell_type == "code" and old_marker in cell.source and "real_life_features = add_simple_features(real_life_tests)" in cell.source:
        cell.source = new_source
        replaced += 1

nbformat.write(nb, NOTEBOOK)
print(f"Replaced {replaced} real-life feature evaluation cell(s).")
