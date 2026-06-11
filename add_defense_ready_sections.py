from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Scam_or_Not_Scam_IR_Project.ipynb"


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    return nbformat.v4.new_code_cell(text.strip())


nb = nbformat.read(NOTEBOOK, as_version=4)

marker = "Defense-Ready Evaluation, Ablation Study, and IR Analysis"
if any(marker in cell.get("source", "") for cell in nb.cells):
    raise SystemExit("Defense-ready section already exists; no duplicate cells added.")

cells = [
    md(
        """
        # Defense-Ready Evaluation, Ablation Study, and IR Analysis

        This section strengthens the project for presentation/defense. The original classroom dataset is useful for learning text classification, but its perfect or near-perfect accuracy can be misleading because real scam messages are noisy, multilingual, short, adversarial, and constantly changing. Therefore, this section adds dataset transparency, realistic hard tests, ablation analysis, and Information Retrieval experiments.
        """
    ),
    md(
        """
        ## Dataset Quality and Source Limitations

        | Dataset source | Role in project | Strength | Limitation |
        |---|---|---|---|
        | Original assignment dataset (`safe-texts.txt`, `scam-texts.txt`) | Main supervised binary classification corpus | Balanced classes and follows the assignment structure | Highly separable, so accuracy can look unrealistically perfect |
        | UCI SMS Spam collection / public SMS spam-style examples | Adds classic SMS spam/smishing patterns | Well-known benchmark style for short messages | Older English-heavy data; not enough for modern Cambodian scams |
        | Synthetic realistic examples | Adds scam categories such as phishing, job, investment, delivery, OTP theft, donation, relationship, BEC, and recovery scams | Covers real-life risk types that the original data lacks | Synthetic text may not fully match real attacker language |
        | Manual Khmer examples | Adds Khmer and Khmer-English mixed scam/safe messages | Important for Cambodian users and local deployment | Still small; Khmer spelling and OCR noise vary widely |
        | Safe educational examples | Hard-negative examples such as "Never share OTP" | Reduces false alarms for safety-awareness messages | Needs more variety from schools, banks, NGOs, and public notices |

        **Production warning:** this dataset is not enough for production deployment. Real scams change over time, attackers adapt wording, and Khmer-English OCR can be noisy. A real system needs continuous data collection, feedback review, drift monitoring, privacy protection, and human-in-the-loop validation.
        """
    ),
    code(
        """
        import re
        import joblib
        import pandas as pd
        from pathlib import Path
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        ROOT = Path.cwd()
        if not (ROOT / "data").exists() and (ROOT / "professional_scam_detection_project").exists():
            ROOT = ROOT / "professional_scam_detection_project"

        DATA_PATH = ROOT / "data" / "realistic_labeled_messages_with_features.csv"
        MODEL_PATH = ROOT / "models" / "realistic_scam_detector_pipeline.joblib"

        dataset_df = pd.read_csv(DATA_PATH)
        dataset_df[["label", "source"]].value_counts().reset_index(name="count").head(20)
        """
    ),
    md(
        """
        ## Real-Life Test Set

        The test set below is intentionally small but difficult. It includes short scam messages, safe educational warnings, Khmer-English mixed messages, public notices, and direct credential requests. This is not a replacement for a large validation set; it is a credibility check showing whether the system handles cases that matter in real life.
        """
    ),
    code(
        """
        real_life_tests = pd.DataFrame([
            {"text": "Send OTP now", "expected": "scam", "case_type": "short critical request"},
            {"text": "Never share OTP with anyone, including bank staff.", "expected": "safe", "case_type": "safe education"},
            {"text": "Dear valued customer, please send your account information such as account number, OTP, PIN to: support-check@gmail.com.", "expected": "scam", "case_type": "credential theft"},
            {"text": "Your parcel is delayed. Pay $1.25 delivery fee now at http://tinyurl.example/parcel", "expected": "scam", "case_type": "delivery fee"},
            {"text": "Meeting reminder: please submit the Python assignment before midnight.", "expected": "safe", "case_type": "normal reminder"},
            {"text": "ABA security: your account is locked. Verify now and enter OTP.", "expected": "scam", "case_type": "bank impersonation"},
            {"text": "សូមកុំផ្ញើ OTP ឬពាក្យសម្ងាត់ទៅអ្នកណាម្នាក់។", "expected": "safe", "case_type": "Khmer safe education"},
            {"text": "សូមផ្ញើលេខគណនី និង OTP ដើម្បីបញ្ជាក់រង្វាន់", "expected": "scam", "case_type": "Khmer OTP theft"},
            {"text": "Public notice: heavy rain is expected today. Please stay safe.", "expected": "safe", "case_type": "public notice"},
            {"text": "Remote job approved. Pay training fee first, then earn daily commission.", "expected": "scam", "case_type": "job scam"},
            {"text": "I love you and trust only you. I am stuck overseas; send money for my ticket.", "expected": "scam", "case_type": "relationship scam"},
            {"text": "Finance team, update the vendor bank account below and process payment today.", "expected": "scam", "case_type": "business email compromise"},
        ])
        real_life_tests
        """
    ),
    code(
        """
        critical_terms = re.compile(
            r"\\b(otp|password|pin|cvv|account number|account information|seed phrase|private key|verification code|login code)\\b",
            re.I,
        )
        request_terms = re.compile(r"\\b(send|give|share|provide|enter|submit|reply|verify|confirm|update|pay)\\b", re.I)
        safe_context = re.compile(r"\\b(never|do not|don't|dont|avoid|warning|education|awareness|protect|will never)\\b", re.I)
        scam_terms = re.compile(
            r"\\b(locked|urgent|prize|reward|delivery fee|training fee|commission|guaranteed return|crypto|wire|vendor bank|recover your money|donation|gift card|blackmail|pay)\\b",
            re.I,
        )

        khmer_critical = re.compile(r"(OTP|PIN|CVV|ពាក្យសម្ងាត់|លេខសម្ងាត់|លេខគណនី|កូដ)")
        khmer_request = re.compile(r"(ផ្ញើ|ផ្តល់|បញ្ចូល|បញ្ជាក់|ប្រាប់|ចែករំលែក|បង់|ផ្ទេរ)")
        khmer_safe = re.compile(r"(កុំ|មិនត្រូវ|ប្រុងប្រយ័ត្ន|ការពារ|ព្រមាន)")

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
            out["has_exclamation"] = (out["exclamation_count"] > 0).astype(int)
            out["has_url"] = out["text"].str.contains(r"https?://|www\\.|\\.com", case=False, regex=True).astype(int)
            out["has_phone"] = out["text"].str.contains(r"(?:\\+?\\d[\\s\\-().]*){7,}", regex=True).astype(int)
            out["has_money"] = out["text"].str.contains(r"\\$|fee|payment|loan|prize|bonus|refund", case=False, regex=True).astype(int)
            out["has_urgency"] = out["text"].str.contains(r"urgent|now|today|locked|suspended|expire|limited", case=False, regex=True).astype(int)
            out["has_action"] = out["text"].str.contains(r"send|give|share|provide|click|verify|pay|reply|submit|enter", case=False, regex=True).astype(int)
            out["has_credential_request"] = out["text"].apply(lambda x: int(bool(critical_terms.search(x) and request_terms.search(x))))
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
        """
    ),
    code(
        """
        def metrics_for(column):
            return {
                "approach": column,
                "accuracy": accuracy_score(real_life_tests["expected"], real_life_tests[column]),
                "precision_scam": precision_score(real_life_tests["expected"], real_life_tests[column], pos_label="scam", zero_division=0),
                "recall_scam": recall_score(real_life_tests["expected"], real_life_tests[column], pos_label="scam", zero_division=0),
                "f1_scam": f1_score(real_life_tests["expected"], real_life_tests[column], pos_label="scam", zero_division=0),
                "confusion_matrix": confusion_matrix(real_life_tests["expected"], real_life_tests[column], labels=["safe", "scam"]).tolist(),
            }

        ablation_metrics = pd.DataFrame([metrics_for("ml_only"), metrics_for("rule_only"), metrics_for("hybrid")])
        ablation_metrics
        """
    ),
    code(
        """
        errors = real_life_tests[~real_life_tests["hybrid_correct"]].copy()
        false_positives = errors[(errors["expected"] == "safe") & (errors["hybrid"] == "scam")]
        false_negatives = errors[(errors["expected"] == "scam") & (errors["hybrid"] == "safe")]

        print("False positives")
        display(false_positives[["case_type", "text", "ml_probability", "rule_only", "hybrid"]])

        print("False negatives")
        display(false_negatives[["case_type", "text", "ml_probability", "rule_only", "hybrid"]])
        """
    ),
    md(
        """
        ## Ablation Study Interpretation

        - **ML only** tests whether the trained classifier can generalize from the dataset.
        - **Rule-based only** tests whether critical safety triggers can catch dangerous cases even when the model is unsure.
        - **Hybrid ML + rules** is preferred for scam prevention because false-safe outputs are more dangerous than cautious warnings.
        - **TF-IDF retrieval** adds Information Retrieval ranking by lexical similarity to known scam examples.
        - **Embedding retrieval** adds semantic similarity for paraphrases and multilingual messages, when the multilingual embedding model is available.
        """
    ),
    md(
        """
        ## Information Retrieval Experiment: TF-IDF Cosine Similarity

        This applies the vector-space model from Information Retrieval. A user message is converted into a TF-IDF vector and compared with known scam-message vectors using cosine similarity:

        $$\\cos(\\theta)=\\frac{A\\cdot B}{\\|A\\|\\|B\\|}$$

        The top-k retrieved scam examples help explain *which known scam patterns are closest* to the user's message.
        """
    ),
    code(
        """
        scam_corpus = dataset_df[dataset_df["label"].astype(str).str.lower().eq("scam")].copy()
        scam_corpus = scam_corpus[scam_corpus["text"].astype(str).str.len() > 8].sample(min(3500, len(scam_corpus)), random_state=42)

        tfidf_retriever = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)
        scam_matrix = tfidf_retriever.fit_transform(scam_corpus["text"].astype(str))

        def top_k_scam_tfidf(query, k=5):
            query_vec = tfidf_retriever.transform([query])
            scores = cosine_similarity(query_vec, scam_matrix).ravel()
            top_idx = scores.argsort()[::-1][:k]
            out = scam_corpus.iloc[top_idx][["label", "source", "text"]].copy()
            out.insert(0, "similarity", [round(float(scores[i]), 4) for i in top_idx])
            return out

        sample_query = "Dear customer, send OTP PIN and account number to verify your bank account today."
        top_k_scam_tfidf(sample_query, k=5)
        """
    ),
    code(
        """
        ir_eval_queries = pd.DataFrame([
            {"query": "Send OTP PIN and account number to verify your account", "expected_terms": ["otp", "pin", "account"]},
            {"query": "Pay delivery fee now to release your parcel", "expected_terms": ["delivery", "fee", "parcel"]},
            {"query": "Remote job approved pay registration fee for commission", "expected_terms": ["job", "fee", "commission"]},
            {"query": "CEO request update vendor bank account and wire today", "expected_terms": ["vendor", "bank", "wire"]},
        ])

        rows = []
        for _, row in ir_eval_queries.iterrows():
            retrieved = top_k_scam_tfidf(row["query"], k=5)
            terms = [term.lower() for term in row["expected_terms"]]
            relevant = retrieved["text"].astype(str).str.lower().apply(lambda text: any(term in text for term in terms))
            rows.append({
                "query": row["query"],
                "precision_at_5": relevant.mean(),
                "top_similarity": retrieved["similarity"].iloc[0],
                "top_result": retrieved["text"].iloc[0],
            })

        tfidf_ir_eval = pd.DataFrame(rows)
        tfidf_ir_eval
        """
    ),
    code(
        """
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
            sample_scam = scam_corpus.head(1200).copy()
            scam_embeddings = embedding_model.encode(sample_scam["text"].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=False)

            def top_k_scam_embedding(query, k=5):
                q = embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
                scores = cosine_similarity(q, scam_embeddings).ravel()
                top_idx = scores.argsort()[::-1][:k]
                out = sample_scam.iloc[top_idx][["label", "source", "text"]].copy()
                out.insert(0, "similarity", [round(float(scores[i]), 4) for i in top_idx])
                return out

            embedding_demo = top_k_scam_embedding("Someone asks me to confirm my secret login code for a reward", k=5)
            display(embedding_demo)
        except Exception as exc:
            print("Embedding retrieval skipped because the multilingual model is not installed/cached locally.")
            print(exc)
        """
    ),
    md(
        """
        ## TF-IDF Retrieval vs Embedding Retrieval

        | Method | Best at | Weakness |
        |---|---|---|
        | TF-IDF cosine retrieval | Exact/near-exact words, URLs, numbers, repeated scam phrases, character-level Khmer-English patterns | Misses paraphrases when vocabulary is different |
        | Multilingual embedding retrieval | Semantic similarity, paraphrases, mixed Khmer-English meanings | Requires larger model dependency and can be slower |
        | Hybrid classifier + retrieval | Gives a decision, explanation, and supporting similar cases | Still needs more real Cambodian scam data for production reliability |
        """
    ),
    md(
        """
        ## Research-Level Limitations and Future Work

        The project is suitable as an academic prototype and portfolio demo, but not a production fraud-prevention system yet. Future work should include:

        - A larger Cambodian scam corpus from verified public reports and responsible user feedback.
        - Active learning where user feedback is reviewed and added to the training data.
        - Transformer fine-tuning for Khmer-English scam text.
        - Multilingual embedding retrieval as a standard feature, not optional.
        - Better Khmer OCR and message-bubble detection for screenshots.
        - Real-time browser, email, or SMS integration with privacy controls.
        - Human-in-the-loop verification for high-risk or uncertain cases.
        """
    ),
]

nb.cells.extend(cells)
nbformat.write(nb, NOTEBOOK)
print(f"Added {len(cells)} defense-ready notebook cells to {NOTEBOOK.name}")
