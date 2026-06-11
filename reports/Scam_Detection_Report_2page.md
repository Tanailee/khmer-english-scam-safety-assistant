# Scam or Not Scam Text Classification Report

**Student:** [Your Name]  
**Student ID:** [Your Student ID]  
**Course:** Information Web Retrieval Analysis / Natural Language Processing  
**Project:** Mini Project 2 - Binary Text Classification  

## 1. Objective

This project adapts the assigned positive/negative review classification workflow to the supplied `safe` and `scam` message corpus. The goal is to classify a new message as **safe** or **scam** using an Information Retrieval pipeline: text preprocessing, dictionary construction, feature extraction, vector-space representation, supervised classification, evaluation, and a deployable Streamlit prototype.

## 2. Literature Review

| Author | Title | Methodology | Key Insight | Research Gap |
|---|---|---|---|---|
| T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami | [A Contribution to the Study of SMS Spam Filtering: New Collection and Results](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) | Created a public SMS spam corpus and benchmarked traditional text-classification methods for ham/spam detection. | Short-message classification can work well with carefully cleaned text, word dictionaries, and supervised classifiers. | The dataset is older and mostly spam/ham, so modern scam and smishing messages require newer, domain-specific validation. |
| N. Al Moubayed, T. Breckon, P. Matthews, and A. S. McGough | [SMS Spam Filtering using Probabilistic Topic Modelling and Stacked Denoising Autoencoder](https://arxiv.org/abs/1606.05554) | Combined topic modelling with stacked denoising autoencoders to classify SMS spam with limited manual feature engineering. | Topic representations can improve interpretability, while learned representations can achieve strong spam-filtering accuracy. | Deep models are less transparent for beginner users and may be unnecessary when a small labelled corpus is linearly separable. |
| Y. Li, R. Zhang, W. Rong, and X. Mi | [SpamDam: Towards Privacy-Preserving and Adversary-Resistant SMS Spam Detection](https://arxiv.org/abs/2404.09481) | Built a large SMS spam collection pipeline, studied campaign patterns, and evaluated centralized and federated spam detectors. | Real-world SMS spam changes over time and privacy-preserving learning is important for deployment. | Portfolio projects should add drift monitoring, feedback loops, and privacy controls before use with personal messages. |
| M. Salman, M. Ikram, N. Basta, and M. A. Kaafar | [SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs](https://arxiv.org/abs/2501.04985) | Compared zero-shot, few-shot, fine-tuned, and chain-of-thought LLM strategies for SMS spam detection. | Fine-tuned language models can be robust, but zero-shot prompting alone is unreliable for production spam detection. | LLMs are powerful but heavier than classical models; small TF-IDF/logistic models remain attractive for transparent, low-cost deployments. |

The literature shows that short-message scam detection benefits from a balance between **interpretable engineered features** and **statistical text representations**. Therefore, this project uses both assignment-required features and additional real-world scam indicators.

## 3. Dataset and Preprocessing

The original assignment dataset contains 40,000 labelled messages: 20,000 safe and 20,000 scam. To make the project more realistic, an additional real-world dataset layer was collected and generated. It includes the public UCI SMS Spam Collection, paraphrased public-warning scam patterns, Khmer manual scam examples, synthetic Khmer/English scam templates, and safe educational contrast messages. The combined training resource now contains more than 45,000 messages, including realistic cases for OTP theft, fake bank support, delivery-fee scams, prize scams, fake loans, fake jobs, and Khmer scam wording.

Following the project instruction, the original corpus still uses the top 80% of each class for training and the final 20% for testing. External and realistic examples are included in training to improve real-world behavior. Preprocessing includes lowercasing, English tokenization, Khmer Unicode detection, URL/phone/money detection, punctuation cues, pronoun matching, dictionary matching, and Khmer credential/action/service pattern matching.

## 4. Features

Required assignment features:

- Count of safe/positive words.
- Count of scam/negative words.
- Binary indicator for the word `no`.
- Count of first/second person pronouns: `I`, `me`, `my`, `you`, `your`.
- Binary indicator for exclamation mark `!`.
- Log length of the message.

Additional professional features:

- URL, phone-number, and money/prize indicators.
- Urgency and action-word counts.
- Digit count, uppercase ratio, average word length, question count.
- Safe/scam word density.
- TF-IDF unigram and bigram representation for vector-space modeling.

## 5. Model Architecture and Experiments

Three models were compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Required engineered features + Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF + Multinomial Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF + engineered features + Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The selected model is **Required engineered features + Logistic Regression** because it achieved the highest test performance and remains transparent enough for a portfolio-ready safety application.

## 6. Realism Improvement and Results

The final model achieved **1.0000 accuracy**, **1.0000 precision**, **1.0000 recall**, and **1.0000 F1-score**. The confusion matrix is `[[4000, 0], [0, 4000]]` in the order `[[safe-safe, safe-scam], [scam-safe, scam-scam]]`.

The perfect result suggests that the provided corpus is highly separable and not fully realistic. A practical scam detector must catch messages such as `Give me your account password`, even when they are short and do not contain links or money terms. To improve real-life usefulness, the project adds a realistic augmentation set with credential theft, OTP requests, delivery-fee scams, refund scams, account-lock scams, and hard safe examples such as `Never give anyone your password`. The final deployed model uses a hybrid approach: word TF-IDF, character TF-IDF, engineered features, and a rule-based credential-risk layer.

The realistic demo test now correctly flags credential requests as scam while treating educational safety warnings as safe. Khmer support was also added using Khmer Unicode detection, Khmer credential terms, Khmer request/action terms, and Khmer trusted-service scam patterns. For example, messages asking users to enter or send account numbers, passwords, OTP codes, or payment details in Khmer are now treated as high-risk. This is more applicable than relying only on the original clean English-oriented dataset accuracy.

## 7. Real-Life Assistant and Deployment

The saved pipeline is connected to a Streamlit app upgraded into a real-life **Khmer-English Scam Safety Assistant**. Users can paste a suspicious SMS, email, Telegram/Facebook chat, or upload a screenshot. The app returns a prediction, risk level, scam type, highlighted suspicious terms, risk signals, and next-step safety advice.

The deployed assistant uses critical rule overrides for OTP, password, PIN, CVV, seed phrase, account number, urgent verification, suspicious links, and payment/deposit pressure. It classifies common scam categories such as password/OTP theft, fake bank support, delivery-fee scam, prize scam, loan scam, job scam, online gambling/deposit scam, investment/crypto scam, email phishing, and account locked verification scam.

For screenshots, the OCR pipeline uses Tesseract Khmer OCR and English OCR fallback. The extracted text is cleaned before analysis to reduce phone UI headers, dates, contact names, duplicated lines, and symbol-heavy OCR noise. The app also includes user feedback buttons, batch CSV checking, and a downloadable safety report.

## 8. Future Work

Future improvements include continuous user-feedback retraining, larger Khmer datasets from consented public reports, drift monitoring, privacy-first deployment, and comparison with transformer embeddings or fine-tuned language models.
