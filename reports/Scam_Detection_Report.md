# Scam or Not Scam: Information Retrieval and Text Analytics Mini Project

## Executive Summary

This project builds a practical scam-message detection system using the Information Retrieval concepts covered in Chapters 1-5: text preprocessing, tokenization, dictionary construction, term weighting, vector-space representation, ranking/classification evaluation, and user-oriented retrieval applications. The final solution is designed as both an academic submission and a portfolio-ready prototype that can later be deployed with Streamlit.

The best model in this reproducible package is **Required engineered features + Logistic Regression**. On the held-out evaluation split it achieved:

| Metric | Score |
|---|---:|
| Accuracy | 1.0000 |
| Precision (scam) | 1.0000 |
| Recall (scam) | 1.0000 |
| F1-score (scam) | 1.0000 |
| ROC-AUC | 1.0 |

Confusion matrix format is `[[safe predicted safe, safe predicted scam], [scam predicted safe, scam predicted scam]]`: `[[4000, 0], [0, 4000]]`.

The assignment describes product-review sentiment classification with positive and negative review files. The supplied project files follow the same binary text-classification structure but use `safe` and `scam` labels. Therefore, `safe_word_hits` is used as the positive-word count and `scam_word_hits` is used as the negative/scam-indicator word count.

## Literature Review

| Author | Title | Methodology | Key Insight | Research Gap |
|---|---|---|---|---|
| T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami | [A Contribution to the Study of SMS Spam Filtering: New Collection and Results](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) | Created a public SMS spam corpus and benchmarked traditional text-classification methods for ham/spam detection. | Short-message classification can work well with carefully cleaned text, word dictionaries, and supervised classifiers. | The dataset is older and mostly spam/ham, so modern scam and smishing messages require newer, domain-specific validation. |
| N. Al Moubayed, T. Breckon, P. Matthews, and A. S. McGough | [SMS Spam Filtering using Probabilistic Topic Modelling and Stacked Denoising Autoencoder](https://arxiv.org/abs/1606.05554) | Combined topic modelling with stacked denoising autoencoders to classify SMS spam with limited manual feature engineering. | Topic representations can improve interpretability, while learned representations can achieve strong spam-filtering accuracy. | Deep models are less transparent for beginner users and may be unnecessary when a small labelled corpus is linearly separable. |
| Y. Li, R. Zhang, W. Rong, and X. Mi | [SpamDam: Towards Privacy-Preserving and Adversary-Resistant SMS Spam Detection](https://arxiv.org/abs/2404.09481) | Built a large SMS spam collection pipeline, studied campaign patterns, and evaluated centralized and federated spam detectors. | Real-world SMS spam changes over time and privacy-preserving learning is important for deployment. | Portfolio projects should add drift monitoring, feedback loops, and privacy controls before use with personal messages. |
| M. Salman, M. Ikram, N. Basta, and M. A. Kaafar | [SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs](https://arxiv.org/abs/2501.04985) | Compared zero-shot, few-shot, fine-tuned, and chain-of-thought LLM strategies for SMS spam detection. | Fine-tuned language models can be robust, but zero-shot prompting alone is unreliable for production spam detection. | LLMs are powerful but heavier than classical models; small TF-IDF/logistic models remain attractive for transparent, low-cost deployments. |
| D. Goel, H. Ahmad, A. K. Jain, and N. K. Goel | [Machine Learning Driven Smishing Detection Framework for Mobile Security](https://arxiv.org/abs/2412.09641) | Used content-based smishing detection with text normalization and machine-learning classifiers. | Normalizing slang, abbreviations, and short forms improves mobile-message threat detection. | The current project should later add Khmer-English normalization and multilingual scam examples. |

## Problem Statement

Online and SMS scams are a real information filtering problem. Users receive short, noisy, high-risk messages and need a system that can separate safe messages from suspicious messages quickly. The project objective is to design an IR-based text analytics pipeline that:

1. Reads labeled scam and safe text corpora.
2. Cleans, tokenizes, and represents messages as weighted terms.
3. Uses scam-indicator and safe word lists as domain features.
4. Trains and evaluates a classification model.
5. Produces interpretable outputs suitable for a real user-facing warning tool.

## Course Alignment

The project uses the chapter themes as follows:

| Course concept | Project implementation |
|---|---|
| Text preprocessing | Lowercasing, tokenization, punctuation/digit/URL/phone/money detection |
| Dictionary and vocabulary | Scam-indicator and safe word lists plus TF-IDF vocabulary |
| Term weighting | TF-IDF with unigram and bigram features |
| Vector-space model | Each message becomes a sparse vector plus engineered numeric features |
| Classification and evaluation | Logistic regression, confusion matrix, precision, recall, F1, ROC-AUC |
| Practical retrieval system | Streamlit-ready interface for checking real messages |

## Dataset

The project uses four provided dataset files:

- `safe-texts.txt`
- `scam-texts.txt`
- `safe-words.txt`
- `scam-indicator-words.txt`

Dataset summary:

| Label | Documents | Average words | Median words |
|---|---:|---:|---:|
| Safe | 20000 | 8.17 | 11.00 |
| Scam | 20000 | 8.90 | 6.00 |

Top safe terms: the, for, thanks, let, know, have, you, best, regards, great, day, next.

Top scam terms: ref, your, bit, link, login, verify, now, claim, reward, via, secure, web.

## Methodology

The pipeline follows a professional machine learning workflow:

1. **Data ingestion:** Each non-empty line or paragraph is treated as one message.
2. **Labeling:** Messages from `safe-texts.txt` are labeled `safe`; messages from `scam-texts.txt` are labeled `scam`.
3. **Required feature engineering:** The system computes safe/positive word count, scam/negative word count, whether the text contains `no`, first/second pronoun count (`I`, `me`, `my`, `you`, `your`), whether `!` appears, and log review length.
4. **Additional feature engineering:** It also computes digit count, uppercase ratio, URL presence, phone presence, money cue presence, urgency terms, action terms, scam word density, and safe word density.
5. **Text representation:** TF-IDF converts message text into weighted unigram and bigram vectors.
6. **Modeling:** Multiple models are compared: required engineered features with Logistic Regression, TF-IDF with Naive Bayes, and TF-IDF plus engineered features with Logistic Regression.
7. **Evaluation:** Performance is measured using the assignment-required accuracy metric plus precision, recall, F1-score, ROC-AUC, and the confusion matrix.
8. **Deployment readiness:** The best model pipeline is saved as a reusable `joblib` file and a Streamlit app template is included.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Required engineered features + Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF + Multinomial Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF + engineered features + Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Results

```text
              precision    recall  f1-score   support

        safe       1.00      1.00      1.00      4000
        scam       1.00      1.00      1.00      4000

    accuracy                           1.00      8000
   macro avg       1.00      1.00      1.00      8000
weighted avg       1.00      1.00      1.00      8000

```

The project emphasizes **recall for scam detection** because missing a scam can be more harmful than showing a cautious warning for a safe message. Precision still matters because too many false alarms reduce user trust.

Because the provided corpora are highly separable, the results may appear perfect. This should be reported honestly as a property of the given dataset and should be validated on newer, messier real-world scam messages before deployment.

## Interpretation

Typical scam indicators in the dataset include money/prize terms, urgent action requests, verification language, links, phone numbers, and unusually promotional wording. Safe messages tend to contain more ordinary conversational terms and fewer pressure cues. The combined TF-IDF plus engineered-feature approach is stronger than a pure keyword filter because it learns phrase patterns while still preserving interpretable risk signals.

## Real-Life Application

The included Streamlit prototype can be turned into a portfolio application where users paste a suspicious message and receive:

- Scam probability.
- Safe/scam prediction.
- Human-readable risk signals.
- Practical safety advice.

This can be expanded for SMS screening, email triage, consumer education, or multilingual scam-awareness tools for local communities.

## Real-Life Safety Assistant Upgrade

The project was upgraded from a simple binary classifier into a **Khmer-English Scam Safety Assistant** for SMS, email, chat, and screenshot-based messages. This improves practical usefulness because real users do not only need a `safe` or `scam` label; they need to understand the risk and know what to do next.

The upgraded system uses a hybrid approach:

1. **Machine learning prediction:** TF-IDF and engineered message features estimate scam probability.
2. **Rule-based critical triggers:** requests for OTP, password, PIN, CVV, seed phrase, account number, or urgent verification are treated as high-risk even when the ML probability is uncertain.
3. **Scam type classification:** the app identifies likely scam categories such as password/OTP theft, fake bank support, delivery fee scam, loan scam, job scam, prize/reward scam, online gambling/deposit scam, investment/crypto scam, email phishing, and account locked verification scam.
4. **Risk explanation:** suspicious terms and behavioral signals are shown to the user.
5. **Action guidance:** the app recommends not replying, not clicking links, not sending money, not sharing secrets, verifying through official apps or phone numbers, saving evidence, and blocking/reporting the sender.

The data layer was also improved with realistic Khmer-English examples, including SMS scams, Telegram/Facebook-style messages, bank impersonation, delivery-fee scams, job scams, fake loans, casino/deposit promotions, investment scams, email phishing, and safe educational messages that mention scam words without being scams. This hard-negative design helps the model distinguish between a real scam request and a safety warning such as "Never share your OTP."

The newest upgrade adds a real-life scam taxonomy covering phishing scams, relationship scams, investment scams, account or identity takeover scams, buying and selling scams, threat/extortion scams, jobs and employment scams, unexpected money scams, business email compromise scams, money recovery scams, and donation scams. Each category is represented as realistic message, chat, or email text and saved in `data/scam_taxonomy_message_email_examples.csv` for transparency and future expansion.

For usability, the Streamlit app now includes Khmer-English OCR for screenshots. Khmer OCR uses Tesseract with Khmer language data, while English/header extraction uses EasyOCR or Tesseract fallback. OCR output is cleaned before prediction to reduce noise from phone UI headers, dates, contact names, and symbol-heavy recognition errors.

## Ethical Considerations

The model should support human judgment, not replace it. False negatives are dangerous because they may allow a scam to pass as safe. False positives are also costly because they may create unnecessary fear. A deployed version should show confidence, explain risk signals, avoid collecting private messages unnecessarily, and provide clear safety guidance.

## Future Work

1. Add more recent scam examples from real public datasets.
2. Include multilingual support for Khmer and English messages.
3. Add explainability with highlighted scam terms.
4. Add active learning so user feedback improves the model.
5. Deploy the Streamlit app with a privacy-first design.
6. Compare Logistic Regression with Naive Bayes, SVM, Random Forest, and transformer embeddings.

## Defense-Ready Methodology Addendum

### Data Collection Ethics

The project uses the original assignment files, public SMS spam-style data, manually written Khmer examples, synthetic realistic scam examples, and safe educational contrast examples. No private user messages are required for the current prototype. Public scam patterns are summarized or synthesized rather than copied with personal identifiers. A real deployment should obtain user consent, avoid storing sensitive content by default, and review user feedback before adding it to the training corpus.

### Dataset Quality and Limitations

The final working dataset contains 45,579 labeled messages. The largest portion is the original assignment corpus, which is balanced and useful for learning classification but highly separable. This explains why standard test accuracy can be extremely high. To make the evaluation more credible, the notebook now includes a separate real-life test set with short messages, safe educational messages, Khmer-English mixed content, direct credential requests, public notices, and common scam categories.

The current dataset is still not production-ready. Real scams change over time, attackers deliberately change wording, and Khmer-English OCR adds noise. A production-grade system would need a larger Cambodian scam corpus, ongoing drift monitoring, and reviewed human feedback.

### Preprocessing and Feature Engineering

The pipeline normalizes message text, extracts token-level features, detects URLs, phone-like numbers, money terms, urgency, action verbs, credential terms, and Khmer/English risk patterns. Safe-context detection helps separate educational warnings such as "Never share OTP" from actual scam requests such as "Send OTP now."

### Model Design

The system uses a hybrid design:

- Machine learning estimates scam probability from TF-IDF and engineered features.
- Rule-based critical triggers override the model for dangerous requests involving OTP, password, PIN, CVV, account number, seed phrase, private key, payment fee, or suspicious link plus urgency.
- Scam type classification translates the result into user-understandable categories such as phishing, OTP theft, fake bank support, delivery fee scam, job scam, investment scam, relationship scam, BEC, and donation scam.

This design is intentionally safety-first. In scam prevention, a false "Safe" result can cause financial or identity harm, so critical credential requests must be treated as high risk even when the ML model is uncertain.

### Retrieval Design

The notebook and Streamlit app now include Information Retrieval experiments:

- TF-IDF cosine similarity ranks known scam examples against a new input.
- Top-k retrieval shows similar scam messages and supports explainability.
- Precision@5/qualitative top-5 review is included for retrieval evaluation.
- Optional multilingual sentence embeddings support semantic retrieval when the model is installed locally.

This connects the project directly to IR concepts: document representation, vector-space ranking, cosine similarity, top-k retrieval, and evaluation.

### Evaluation

The report still presents the assignment-required accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix. The notebook now adds stronger validation:

- Real-life hard test set.
- False positive and false negative inspection.
- Ablation study comparing ML only, rule-based only, and hybrid ML + rules.
- TF-IDF retrieval experiment and top-5 similarity analysis.
- Optional comparison with multilingual embedding retrieval.

### Practical Usability

The Streamlit prototype is designed as a scam safety assistant. It shows the final decision first, then explains why the message is risky, retrieves similar scam examples, suggests what to do next, supports Khmer explanation, allows screenshot OCR with manual correction, saves feedback, and exports a safety report.

### Research-Level Future Work

Future research should build a larger Cambodian scam corpus, add active learning from reviewed user feedback, fine-tune multilingual transformers, improve Khmer OCR, deploy multilingual embedding search at scale, integrate privacy-preserving browser/email/SMS scanning, and use human-in-the-loop verification for uncertain or high-risk cases.

## Appendix: Extracted Course Notes Summary

{
  "assignment": "NATURAL LANGUAGE PRO CESSING M - DAS Mini Project 2 \u2013 Text Classification Given a corpus of product reviews, build a text classifier to predict if an input review is positive or negative. You can implement a ny classification model of your choice. Write a report (max 2 pages) about the features use d, the architecture of the models implemented as well as the results obtained from the experiments. The corpus includes 2 text files: \uf0b7 positive - reviews.txt contains 20,000 lines of reviews (one review per line) considered to be positive \uf0b7 negative - reviews.txt contains 20,000 lines of negative reviews You can use the top 80% lines of each file as your training set. The rest will be used as a test set.",
  "chapter_1": "\u2022 People doing IR work with different media (image retrieval: Google Lens), different types of search applications (Web search engines), and different tasks (recommending content) Dimension of IR Content Applications Tasks T ext Web search Ad hoc search Images Vertical search Filtering Video Enterprise search Classification Scanned docs Desktop search \u01eauestion answering Audio Forum search Music Literature search Dimension of IR Dimension of IR IR Tasks Ad-hoc search Filtering Classification Question Answering \u2022 Find relevant documents for a text query \u2022 Identify relevant user profiles for a new document \u2022 Identify relevant labels for documents \u2022 Give a specific answer to a question The Comparing of IR and Search Engines Information Retrieval Search Engines Relevance \u2022 Effective ranking Evaluation \u2022 Testing and measuring Information needs \u2022 User interaction Performance \u2022 Efficient search and indexing Incorporating new data \u2022 Coverage and freshness Scalability \u2022 Growing with data and use",
  "chapter_2": "Slide 1: Chapter 2 \u2013 Web Scraping | Theory and Lab session | Department of Applied Mathematics and Statistics (AMS) | Institute of Technology of Cambodia | Course: Information Retrieval Web Analytics | Lecturer: Khean Vesal Slide 2: Introduction | to Web Scraping | Web Scraping Techniques | Data Cleaning and Processing | Challenges in Web Scraping | Future of Web Scraping | References Slide 3: Introduction | to Web Scraping | Definition and Importance | Understanding Web Scraping | Web scraping involves automatically extracting large amounts of data from website for analysis and decision making. Slide 8: Introduction | to Web Scraping | c) Implementing Responsible Scraping | Use techniques like rate limiting and respectful crawling to prevent server overloads. | c) Cloud-Based Scraping Platforms | Services like | Octoparse | and | Import.io | provide scalable solutions for large-scale data extraction Slide 10: Introduction | to Web Scraping | Tools and Technologies Used Slide 11: Web S",
  "chapter_3": "Chapter 3 \u2013 Web Crawling Theory and Lab session Department of Applied Mathematics and Statistics (AMS) Institute of Technology of Cambodia Course: Information Retrieval Web Analytics Lecturer: Khean Vesal Web Crawling vs Scraping \u2022 Web crawling is the process by which we gather pages from the web, in order to index them and support a search engine. The comparing of Web Crawling vs Scraping Web Scraping Web Crawling The tool used is Web Scraper The tool used Web Crawler or Spiders It is used for downloading information It is used for indexing of Web pages It need not visit all the pages of website for information It visits each and every page, until the last line for information A Web Scraper doesn\u2019t obey robots.txt in most of cases Not all web crawlers obey robots.txt It is done on both small and large scale It is mostly employed in large scale Application areas include Retail Marketing, Equity search, and Machine learning Used in search engines to give search results to the user \u2022 Inf",
  "chapter_4": "Chapter 4 \u2013 Document Representation and Processing Theory and Lab session Department of Applied Mathematics and Statistics (AMS) Institute of Technology of Cambodia Course: Information Retrieval Web Analytics Lecturer: Khean Vesal Content 1) Tokenization 2) Stop-word removal 3) Stemming and lemmatization 4) Normalization and case folding 5) Handling special characters, numbers, and punctuation 1). Tokenization \u2751 Tokenization Use Cases: \u2022 Information Retrieval: Tokenization is essential for indexing and searching in systems that store and retrieve information efficiently based on words or phrases. Tasks that benefit from stopword removal: \u2022 Text classification and sentiment analysis \u2022 Information retrieval and search engines \u2022 Topic modelling and clustering \u2022 Keyword extraction 2). \u2751 Purpose: Ensures consistent representation of text for matching and retrieval. Tokenization 1).",
  "chapter_5": "Chapter 5 \u2013 Bag of Words model Theory and Lab session Department of Applied Mathematics and Statistics (AMS) Institute of Technology of Cambodia Course: Information Retrieval Web Analytics Lecturer: Khean Vesal Content \u2756 Understanding BoW models \u2756 Workflow of BoW \u2756 The importance of cleaning and preprocessing our data \u2756 BoW model Vs CBoW \u2756 Decoding the Essence of Word Representations \u2756 Workflow of CBoW \u2756 Application and Limitations of CBoW Bag of Words (BoW) models \u2022 The Bag of Words (BoW) model is a fundamental and representation technique in Natural Language Processing (NLP) and Information Retrieval (IR) \u2022 The BoW model represents a document as an unordered set or \u201cbag\u201d of its words, ignore grammar and word order but considering the frequency of each word. \u2022 The resulting representation is a vector where each element corresponds to a unique word in the vocabulary, and the value in each element reflects the frequency of that word in the document. Each unique word becomes a feature in",
  "chapter_2_assignment": "Assignment 1. Scrap data for Khmer Job Data Source Ideas: ex. BongThom, CamHR\u2026 Task: + Scrape job postings: o Job title o Company name o Location o Description + Save to CSV file. 2. Scrap data for Khmer News Data Source: ex."
}
