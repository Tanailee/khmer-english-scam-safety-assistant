from pathlib import Path
import importlib.util


app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.load_model()
safe_words, scam_words = mod.load_wordlists()

examples = [
    "សូមបញ្ចូលលេខគណនី និងពាក្យសម្ងាត់ ដើម្បីទទួលរង្វាន់",
    "សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នកដើម្បីបញ្ជាក់",
    "គណនី ABA របស់អ្នកត្រូវបានចាក់សោ សូមផ្ញើ OTP ឥឡូវនេះ",
    "កុំផ្ញើលេខសម្ងាត់ ឬ OTP ទៅអ្នកណាម្នាក់",
    "យើងរៀនអំពីសុវត្ថិភាពគណនី និងការការពារពាក្យសម្ងាត់",
]

for text in examples:
    feats = mod.build_features(text, safe_words, scam_words)
    ml = float(model.predict_proba(feats)[0, 1])
    pred, hybrid, rule, signals, level, category = mod.hybrid_decision(text, ml, 0.5)
    print(f"{pred:4s} | {level:8s} | {category:38s} | hybrid={hybrid:.2%} | ml={ml:.2%} | rule={rule} | {text}")
    print("     " + "; ".join(signals))
