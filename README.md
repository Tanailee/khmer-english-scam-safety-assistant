# Khmer-English Scam Safety Assistant

Portfolio-ready Information Retrieval and text analytics mini project for detecting suspicious SMS, chat, email, and screenshot-based messages.

This project is no longer only a Safe/Scam classifier. It is designed as a scam-prevention assistant that explains risk, retrieves similar scam examples, gives safety advice, supports Khmer-English text, and keeps OCR results editable because Khmer OCR is still imperfect.

## Main Deliverables

- `Scam_or_Not_Scam_IR_Project.ipynb` - complete Jupyter notebook with preprocessing, EDA, modeling, realistic testing, ablation study, TF-IDF retrieval, and embedding retrieval discussion.
- `reports/Scam_Detection_Report.md` - academic report with methodology, ethics, evaluation, limitations, and future work.
- `slides/Scam_Detection_Presentation.pptx` - presentation deck.
- `app/streamlit_app.py` - Streamlit Scam Safety Assistant.
- `models/realistic_scam_detector_pipeline.joblib` - trained hybrid-ready model pipeline.
- `data/realistic_labeled_messages_with_features.csv` - final labeled dataset used by the app/model.
- `MODEL_CARD.md` - intended use, limitations, risks, and deployment notes.

## Screenshots

Add screenshots here before publishing the GitHub repository:

| Screen | Suggested file |
|---|---|
| Home and privacy warning | `assets/screenshots/home.png` |
| Analyze result with critical risk | `assets/screenshots/analyze-critical.png` |
| Similar scam examples / IR Explorer | `assets/screenshots/ir-explorer.png` |
| OCR workflow with manual correction | `assets/screenshots/ocr-workflow.png` |

Recommended demo screenshot message:

```text
Dear customer, send your OTP, PIN, and account number to support-check@gmail.com to verify your account today.
```

## Dataset Summary

Final dataset: **45,579 labeled messages**.

| Source | Count | Purpose | Limitation |
|---|---:|---|---|
| Original assignment dataset | 40,000 | Main supervised safe/scam corpus | Highly separable, so accuracy can look unrealistically perfect |
| UCI SMS Spam-style public dataset | 5,158 | Adds public SMS spam/ham patterns | Older and mostly English |
| Realistic augmented examples | 180 | Adds modern scam scenarios | Synthetic, not a substitute for live reports |
| Synthetic scam templates | 175 | Covers broad scam taxonomy | Template wording may be predictable |
| Hard-negative safe education examples | 24 | Teaches the model that "Never share OTP" is safe education | Needs more real public-awareness examples |
| Public report pattern synthesis | 18 | Adds public scam report patterns without private data | Manually summarized patterns |
| Manual safe educational contrast | 14 | Adds safe messages containing risky words | Small sample |
| Manual Khmer scam examples | 10 | Adds Khmer and Khmer-English local examples | Still not enough for production Khmer deployment |

**Production warning:** this dataset is suitable for coursework and portfolio demonstration, not production fraud prevention. Real scams change quickly and require continuous data collection, drift monitoring, reviewed user feedback, and privacy-preserving evaluation.

## Real-Life Features

- Hybrid decision logic: ML probability + critical safety rules.
- Critical override for OTP, password, PIN, CVV, account number, seed phrase, private key, payment fee, and suspicious link + urgency.
- Scam category detection: phishing, relationship scam, investment scam, account/identity takeover, buying/selling scam, threat/extortion, job scam, unexpected money, business email compromise, money recovery, donation scam, OTP theft, fake bank support, delivery fee, loan, gambling/deposit scam.
- TF-IDF cosine similarity retrieval: shows top 5 similar scam examples.
- Optional multilingual embedding retrieval with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- OCR workflow for screenshots with auto message-bubble crop, manual crop, raw OCR review, manual correction, and OCR quality warnings.
- Khmer explanation toggle, highlighted risk terms, next-step safety advice, feedback saving, batch CSV scanning, and downloadable safety report.

## Run Notebook

Open `Scam_or_Not_Scam_IR_Project.ipynb` and run cells from top to bottom.

The defense-ready section includes:

- dataset quality table and source limitations
- real-life hard test set
- false positive / false negative review
- ablation study: ML only vs rules only vs hybrid
- TF-IDF cosine retrieval and precision@5
- optional multilingual embedding retrieval
- research limitations and future work

## Run Streamlit App

```powershell
cd "D:\M-DAS\Semerster 2\Information Retrieval and Analytic\Scam or not scam\professional_scam_detection_project"
python -m streamlit run app\streamlit_app.py --server.port 8529
```

Then open:

```text
http://localhost:8529
```

If a port is busy, change `8529` to another port such as `8530`.

## OCR Setup

Basic OCR packages:

```powershell
python -m pip install easyocr pillow pytesseract
```

Khmer OCR requires Tesseract OCR plus Khmer trained data (`khm.traineddata`). The app does not overclaim OCR accuracy. It always asks the user to review and correct OCR text before analysis.

## Demo Examples

Scam:

```text
Dear valued customer, please send your account information such as account number, OTP, PIN to: support-check@gmail.com.
```

Safe education:

```text
Never share OTP, password, PIN, or bank account number with anyone.
```

Khmer scam:

```text
សូមផ្ញើលេខគណនី និង OTP ដើម្បីបញ្ជាក់រង្វាន់
```

Khmer safe education:

```text
សូមកុំផ្ញើ OTP ឬពាក្យសម្ងាត់ទៅអ្នកណាម្នាក់។
```

## Privacy Warning

This app does not need real passwords, OTPs, bank account numbers, card numbers, seed phrases, private keys, or identity documents. Do not paste or upload real sensitive information.

## Streamlit Deployment Guide

1. Push the project to GitHub.
2. Add `requirements.txt`.
3. Deploy with Streamlit Community Cloud.
4. Set the app file to `app/streamlit_app.py`.
5. For deployment, keep OCR optional because Tesseract Khmer data may require extra system configuration.

## Future Work

- Larger Cambodian scam corpus from verified public sources.
- Reviewed active learning from user feedback.
- Transformer fine-tuning for Khmer-English scam text.
- Production multilingual embedding index.
- Better Khmer OCR and mobile screenshot parsing.
- Privacy-preserving browser/email/SMS integration.
- Human-in-the-loop review for high-risk or uncertain cases.
