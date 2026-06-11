import runpy
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ns = runpy.run_path("app/streamlit_app.py", run_name="__test__")

cases = [
    "Security alert: your email account will be disabled today. Click the link and enter your password.",
    "I love you and want to visit you soon, but my card is blocked. Please send 300 USD.",
    "Guaranteed crypto profit: deposit 100 USD today and withdraw 1,000 USD tomorrow.",
    "Your Telegram account will be locked. Send the login code to verify ownership.",
    "I want to buy your phone. Pay the courier escrow verification fee to receive funds.",
    "We recorded your private video. Pay 500 USD in Bitcoin or we send it to your contacts.",
    "Remote job approved. Pay 15 USD registration fee and submit your bank account.",
    "You have received an inheritance fund. Pay clearance fee and submit bank details.",
    "CEO request: urgently wire 18,750 USD to this new supplier account today.",
    "We recovered your lost scam money. Pay a small processing fee.",
    "Urgent disaster donation: send money to this personal ABA account now.",
    "Dear valued customer, please send your account information such as account number, OTP, PIN to: sithansitana9899@gmail.com.",
    "ABA notice: please send account info, password, and OTP to the agent on WhatsApp for urgent verification.",
    "សូមផ្ញើលេខគណនី និង OTP ដើម្បីបញ្ជាក់គណនីរបស់អ្នក",
]

for text in cases:
    features = ns["build_features"](text, ns["safe_words"], ns["scam_words"])
    prob = float(ns["model"].predict_proba(features)[0, 1])
    decision = ns["hybrid_decision"](text, prob, 0.5)
    print(decision[0], decision[4], decision[5], f"{decision[1]:.2f}", "::", text[:70])

safe_cases = [
    "Never share your OTP, password, PIN, CVV, or bank account number with anyone.",
    "Security education: no real bank asks customers to send account information, OTP, or PIN to a Gmail address.",
    "សូមកុំផ្ញើ OTP ឬលេខសម្ងាត់ទៅអ្នកណាម្នាក់",
]

for text in safe_cases:
    analysis_text = ns["prepare_message_for_analysis"](text)
    features = ns["build_features"](analysis_text, ns["safe_words"], ns["scam_words"])
    prob = float(ns["model"].predict_proba(features)[0, 1])
    decision = ns["hybrid_decision"](analysis_text, prob, 0.5)
    print(decision[0], decision[4], decision[5], f"{decision[1]:.2f}", "::", text[:70])
