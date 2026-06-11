from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import make_realistic_model as trainer


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"


SCAM_TAXONOMY_EXAMPLES = {
    "Phishing scams": [
        "Security alert: your email account will be disabled today. Click the link and enter your password to keep access.",
        "Your bank profile is incomplete. Verify your identity at this secure link and submit your OTP.",
        "Microsoft 365 notice: mailbox storage is full. Login now to avoid losing incoming emails.",
        "We detected unusual activity on your wallet. Confirm your seed phrase to restore access.",
    ],
    "Relationship scams": [
        "I love you and want to visit you soon, but my card is blocked. Please send 300 USD for my ticket.",
        "My family has an emergency and I trust only you. Transfer money today and I will repay you after we meet.",
        "Please keep our relationship private. Buy gift cards and send the codes so I can solve my problem.",
        "I am stuck overseas and cannot video call. Send money to help me come to Cambodia.",
    ],
    "Investment scams": [
        "Guaranteed crypto profit: deposit 100 USD today and withdraw 1,000 USD tomorrow.",
        "Exclusive investment group. Transfer money now and earn 30 percent every week with no risk.",
        "Your trading account has profit waiting. Pay the release tax before withdrawal.",
        "Join our gold investment plan. Limited seats, send ABA transfer screenshot to activate.",
    ],
    "Account or identity takeover scams": [
        "Your Telegram account will be locked. Send the login code to verify ownership.",
        "Facebook support: your page violates policy. Provide password and verification code to appeal.",
        "Someone tried to access your bank account. Reply with OTP so we can secure it.",
        "Your SIM card registration failed. Submit ID number, date of birth, and PIN immediately.",
    ],
    "Buying and selling scams": [
        "I want to buy your phone. I sent extra payment by courier escrow; pay the delivery fee to receive funds.",
        "Your marketplace order is confirmed. Click the link and enter card details to release payment.",
        "Seller protection requires a refundable verification fee before we send the buyer money.",
        "The buyer paid through secure shipping. Pay customs first so the payment can appear in your account.",
    ],
    "Threat scams": [
        "We recorded your private video. Pay 500 USD in Bitcoin within 24 hours or we send it to your contacts.",
        "Final warning from police department: pay this fine now or your case will be sent to court.",
        "Your phone is hacked. Transfer money to this wallet or all photos will be leaked.",
        "You missed a legal notice. Click here and pay the penalty immediately to avoid arrest.",
    ],
    "Jobs and employment scams": [
        "Remote job approved. Pay 15 USD registration fee and submit your bank account to receive salary.",
        "Work from home: complete tasks and earn daily commission. Deposit first to unlock higher orders.",
        "HR department selected you. Send passport copy, bank account, and training payment today.",
        "Part-time job on Telegram. Transfer money to activate your employee account.",
    ],
    "Unexpected money scams": [
        "You have received an inheritance fund. Pay clearance fee and submit bank details to claim.",
        "Congratulations, you won a cash prize. Send your account number and OTP to receive money.",
        "Government refund available. Click the link and enter card number before midnight.",
        "International transfer pending. Pay anti-money-laundering certificate fee to release funds.",
    ],
    "Business email compromise scams": [
        "Hi finance team, I am in a meeting. Urgently wire 18,750 USD to this new supplier account today.",
        "Please update vendor banking details to the attached account and process payment before 3 PM.",
        "CEO request: buy gift cards for clients and send me the codes by email.",
        "Our invoice bank account changed. Kindly pay the outstanding balance to the new account below.",
    ],
    "Money recovery scams": [
        "We recovered your lost scam money. Pay a small processing fee so we can return the full amount.",
        "Anti-fraud agency notice: your stolen funds are ready. Send ID and transfer recovery tax today.",
        "Crypto recovery expert here. Pay upfront wallet validation fee and we restore your coins.",
        "Your previous investment loss can be refunded. Click the link and enter bank details.",
    ],
    "Donation scams": [
        "Urgent disaster donation: send money to this personal ABA account to help victims now.",
        "Charity for sick children. Transfer today; no receipt available because the hospital needs cash.",
        "Temple rebuilding fund: send your card details and OTP to complete donation.",
        "Emergency fundraiser. Click this shortened link and pay immediately before campaign closes.",
    ],
}


SAFE_TAXONOMY_EXAMPLES = [
    "This article explains phishing scams and reminds readers never to enter passwords from email links.",
    "Relationship scam awareness training: do not send money to someone you have never met in person.",
    "Investment safety lesson: guaranteed high returns with no risk are a warning sign.",
    "Account takeover prevention: enable two-factor authentication and never share login codes.",
    "Marketplace safety guide: meet in public places and do not pay verification fees.",
    "Threat scam warning: report blackmail messages and do not pay attackers.",
    "Job scam education: real employers do not ask applicants to pay a fee before starting work.",
    "Unexpected money scam warning: inheritance and prize messages asking for fees are suspicious.",
    "Business email compromise training: verify bank-account changes by phone before payment.",
    "Money recovery scam warning: recovery agents asking upfront fees are often scammers.",
    "Donation safety notice: donate only through official charity websites and request receipts.",
    "NoDrug public health message: download and use the app together for a Cambodia without drugs.",
    "HIV prevention awareness: stay informed and protect your health through official public campaigns.",
]


EXTRA_SCAM = [
    # English SMS, email, chat, and social-message scams.
    "Your ABA account is locked. Verify now and enter your OTP to restore access.",
    "ACLEDA security alert: click the link and confirm your password today.",
    "Wing support: send your verification code so we can protect your wallet.",
    "Bakong transfer pending. Pay 3 USD delivery fee to release the money.",
    "Congratulations, you won 500 USD. Send your bank account number and phone number.",
    "Your parcel cannot be delivered. Pay the customs fee through this link.",
    "Your loan is approved. Pay a processing fee before 5 PM.",
    "Work from home job available. Deposit 10 USD for training and receive salary tomorrow.",
    "Investment opportunity: guaranteed 30 percent profit today. Transfer money now.",
    "Casino bonus: 60 percent first deposit and 8.8 USD app bonus. Send ABA deposit screenshot.",
    "Telegram admin: your account will be suspended. Share login code to verify.",
    "Facebook page support: give us your password to recover your page.",
    "Email quota full. Login now and update your password to avoid deactivation.",
    "Final warning: your bank card is blocked. Enter CVV and OTP to unlock.",
    "Refund approved. Submit card number, expiry date, and CVV.",
    "I am bank staff. Tell me your PIN so I can check the failed transaction.",
    "Urgent, send your seed phrase to validate your wallet.",
    "You received money from abroad. Pay clearance fee before withdrawal.",
    "Click here to claim government cash support and provide account details.",
    "Your electricity bill has an error. Pay this small fee to avoid disconnection.",
    "Dear valued customer, please send your account information such as account number, OTP, and PIN to supportverify9899@gmail.com.",
    "Dear customer, provide your private information, bank details, and verification code to this email to keep your account active.",
    "Bank security review: reply with your account number, PIN, and OTP so our staff can unlock your account.",
    "Telegram support: send your login code and phone number to @secure_admin to prevent account suspension.",
    "Your identity verification failed. Submit ID card, date of birth, account number, and OTP by email today.",
    "ABA notice: please send account info, password, and OTP to the agent on WhatsApp for urgent verification.",
    "Facebook security team: provide your page password and login code to this email before midnight.",
    "Wallet verification required. Send seed phrase, private key, and phone number to restore access.",
    "Human resources selected you for a remote job. Submit passport, bank account, and pay training fee first.",
    "Invoice payment update: finance team must wire the balance to the new supplier account below today.",
    *[message for messages in SCAM_TAXONOMY_EXAMPLES.values() for message in messages],
    # Khmer realistic scam examples.
    "គណនី ABA របស់អ្នកត្រូវបានចាក់សោ សូមផ្ញើ OTP ដើម្បីបើកវិញភ្លាមៗ",
    "សូមផ្ញើលេខសម្ងាត់គណនីរបស់អ្នក ដើម្បីបញ្ជាក់ការទូទាត់",
    "អ្នកឈ្នះរង្វាន់ ៥០០ ដុល្លារ សូមផ្តល់លេខគណនីធនាគារ",
    "បន្ទាន់ កញ្ចប់របស់អ្នកមិនអាចដឹកជញ្ជូនបាន សូមបង់ថ្លៃសេវាតាមតំណនេះ",
    "កម្ចីរបស់អ្នកអនុម័តហើយ សូមផ្ទេរប្រាក់ថ្លៃសេវាមុនពេលទទួលលុយ",
    "ការងារអនឡាញ បង់ប្រាក់ចុះឈ្មោះ ១០ ដុល្លារ ហើយទទួលប្រាក់ខែ",
    "វិនិយោគថ្ងៃនេះបានចំណេញខ្ពស់ សូមផ្ទេរប្រាក់ឥឡូវនេះ",
    "បាការ៉ាត់ និងហ្គេមអនឡាញ ប្រាក់បន្ថែម ៦០ ភាគរយ សូមដាក់ប្រាក់តាម ABA",
    "គណនី Telegram របស់អ្នកនឹងត្រូវបិទ សូមផ្ញើលេខកូដចូលប្រើ",
    "ទំព័រ Facebook របស់អ្នកមានបញ្ហា សូមផ្ញើពាក្យសម្ងាត់ដើម្បីស្តារវិញ",
    "សូមចុចតំណនេះ ហើយបញ្ចូលលេខកាត និង CVV ដើម្បីទទួលប្រាក់សងវិញ",
    "សូមផ្ញើលេខកូដសម្ងាត់ ឬ OTP ដើម្បីទទួលប្រាក់ពីបរទេស",
]


EXTRA_SAFE = [
    # Hard negatives: safe educational messages that contain scam terms.
    "Never share your OTP, password, PIN, CVV, or bank account number with anyone.",
    "ABA staff will never ask for your password or OTP by SMS.",
    "If a message asks for money or OTP, contact the company using the official app.",
    "This training explains how delivery fee scams work and how to report them.",
    "The bank warning says do not click unknown links or share verification codes.",
    "My teacher asked us to write examples of phishing messages for class.",
    "The report describes fake job scams but does not ask users to send money.",
    "Please screenshot suspicious messages and report them, but never reply.",
    "Official public notice: heavy rain is expected in several provinces. Stay safe.",
    "Weather alert from the ministry: please be careful during flooding.",
    "Your parcel was delivered successfully. No fee is required.",
    "Meeting reminder: bring your laptop and assignment notebook tomorrow.",
    "Do not paste real private information into scam checking apps.",
    "A safe message may mention OTP only to warn people not to share it.",
    "Security education: no real bank asks customers to send account information, OTP, or PIN to a Gmail address.",
    "Class example: a message asking for account number, OTP, and PIN is dangerous and should be reported.",
    "Awareness notice: never send private information through email, Telegram, WhatsApp, or SMS.",
    "Portfolio demo note: test messages should use fake OTP and fake account numbers only.",
    "Safe banking reminder: official support will not ask for your password, PIN, CVV, or seed phrase.",
    "Training example: if someone asks for bank details by email, block and report the message.",
    *SAFE_TAXONOMY_EXAMPLES,
    "សូមកុំផ្ញើ OTP ពាក្យសម្ងាត់ PIN ឬលេខគណនីទៅអ្នកណាម្នាក់",
    "ធនាគារមិនសួរពាក្យសម្ងាត់ ឬ OTP តាមសារទេ",
    "បើមានសារសួរប្រាក់ ឬលេខកូដ សូមទាក់ទងតាមកម្មវិធីផ្លូវការ",
    "ការបណ្តុះបណ្តាលនេះពន្យល់ពីការបោកបញ្ឆោតតាមសារ",
    "សារជូនដំណឹងសាធារណៈអំពីភ្លៀងខ្លាំង សូមប្រុងប្រយ័ត្ន",
    "ក្រសួងជូនដំណឹងអំពីអាកាសធាតុ មិនសួរលេខសម្ងាត់ទេ",
    "សូមរាយការណ៍សារដែលសួរ OTP ឬប្រាក់ តែកុំឆ្លើយតប",
    "សារសុវត្ថិភាព៖ កុំចុចតំណមិនស្គាល់ និងកុំចែករំលែកលេខកូដ",
]


def write_real_life_seed_dataset() -> None:
    rows = []
    for i, text in enumerate(EXTRA_SCAM, 1):
        rows.append({"id": f"upgrade_scam_{i:03d}", "label": "scam", "text": text, "source": "real_life_upgrade"})
    for i, text in enumerate(EXTRA_SAFE, 1):
        rows.append({"id": f"upgrade_safe_{i:03d}", "label": "safe", "text": text, "source": "real_life_upgrade"})
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / "real_life_upgrade_seed_examples.csv", index=False)

    taxonomy_rows = []
    for category, messages in SCAM_TAXONOMY_EXAMPLES.items():
        for i, text in enumerate(messages, 1):
            taxonomy_rows.append({
                "category": category,
                "label": "scam",
                "text": text,
                "channel": "message_text_email",
            })
    for i, text in enumerate(SAFE_TAXONOMY_EXAMPLES, 1):
        taxonomy_rows.append({
            "category": "Safe education / public awareness",
            "label": "safe",
            "text": text,
            "channel": "message_text_email",
        })
    pd.DataFrame(taxonomy_rows).to_csv(DATA / "scam_taxonomy_message_email_examples.csv", index=False)


def main() -> None:
    trainer.SCAM_EXAMPLES.extend(EXTRA_SCAM)
    trainer.SAFE_EXAMPLES.extend(EXTRA_SAFE)
    write_real_life_seed_dataset()
    trainer.main()

    metrics_path = REPORTS / "realistic_model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["real_life_upgrade"] = {
        "extra_scam_examples": len(EXTRA_SCAM),
        "extra_safe_examples": len(EXTRA_SAFE),
        "focus": [
            "Khmer-English SMS, email, chat, and social-message scams",
            "Critical credential/OTP/password triggers",
            "Phishing, relationship, investment, account takeover, marketplace, threat, job, unexpected money, BEC, recovery, donation, bank, delivery, loan, and gambling/deposit scams",
            "Safe educational and public-warning messages as hard negatives",
        ],
        "taxonomy_categories": list(SCAM_TAXONOMY_EXAMPLES.keys()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics["real_life_upgrade"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
