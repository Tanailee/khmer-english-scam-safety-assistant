# Model Card: Khmer-English Scam Safety Assistant

## Model Overview

This model is a coursework and portfolio prototype for detecting suspicious SMS, chat, email, and screenshot-extracted messages. It combines a supervised text classifier with rule-based safety logic and Information Retrieval similarity search.

## Intended Use

- Educational Information Retrieval and text analytics project.
- Scam-awareness assistant for pasted Khmer/English messages.
- Portfolio demo for Streamlit deployment.
- Support tool that explains risk and suggests safe next steps.

## Not Intended For

- Fully automated financial fraud prevention.
- Legal, banking, or law-enforcement decisions.
- Analysis of real sensitive credentials such as OTP, passwords, PINs, CVV, seed phrases, or bank account numbers.
- Production use without larger validation data and human review.

## Training and Reference Data

The working dataset contains 45,579 labeled messages:

| Source | Count |
|---|---:|
| Original assignment dataset | 40,000 |
| UCI SMS Spam-style public dataset | 5,158 |
| Realistic augmented examples | 180 |
| Synthetic realistic scam templates | 175 |
| Hard-negative safe education examples | 24 |
| Public report pattern synthesis | 18 |
| Manual safe educational contrast examples | 14 |
| Manual Khmer scam examples | 10 |

## Features

- TF-IDF text representation.
- Engineered message features: length, URL, phone, money terms, urgency, action verbs, credential terms, safe-context terms.
- Khmer/English rule patterns.
- Critical override rules for credential and payment-risk requests.
- TF-IDF cosine retrieval against known scam examples.
- Optional multilingual embedding retrieval.

## Safety Logic

Messages asking users to send, share, enter, verify, or provide OTP, password, PIN, CVV, account number, seed phrase, private key, or payment fee are treated as high risk unless they are clearly educational safety messages.

## Evaluation

The original dataset gives very high classification metrics because it is highly separable. The notebook includes additional realistic hard tests, false-positive/false-negative review, ablation study, and IR retrieval evaluation to make the defense more credible.

## Limitations

- Khmer scam data is still small.
- OCR quality for Khmer screenshots can be poor and must be manually reviewed.
- Synthetic examples may not represent all real attacker language.
- Real scams change over time.
- The model may still produce false positives or false negatives.

## Privacy

The app does not need real private information. Users should not paste or upload real OTPs, passwords, bank account numbers, card numbers, private keys, seed phrases, or identity documents.

For public deployment, show the privacy warning before users upload screenshots or paste messages. Store feedback only when the user intentionally submits it, and review feedback before using it for retraining.

## Deployment Notes

Streamlit deployment should keep OCR optional unless Tesseract Khmer data is configured on the server. User feedback should be stored only with consent and reviewed before retraining.

## Future Work

- Larger Cambodian scam corpus.
- Active learning with reviewed feedback.
- Transformer fine-tuning.
- Production multilingual embedding index.
- Better Khmer OCR and message-bubble segmentation.
- Real-time browser/email/SMS integration with privacy controls.
