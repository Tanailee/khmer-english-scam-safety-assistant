from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("Scam_or_Not_Scam_IR_Project.ipynb")


def main() -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)

    marker = "## Information Retrieval Extension: Similarity Search and Embeddings"
    if any(marker in "".join(cell.get("source", "")) for cell in nb.cells):
        print("IR retrieval notebook section already exists.")
        return

    cells = [
        nbf.v4.new_markdown_cell(
            marker
            + "\n\n"
            "The earlier model treats scam detection as a supervised text-classification task. "
            "To connect the project more directly to Information Retrieval, this section adds a retrieval layer. "
            "Given a new user message, the system searches the scam corpus and returns the most similar known scam examples. "
            "This is useful for explainability because users can see which previous scam patterns are closest to the new message."
        ),
        nbf.v4.new_markdown_cell(
            "### TF-IDF Vectorization\n\n"
            "TF-IDF converts each message into a vector of weighted terms. A term receives a high weight when it appears often in one message "
            "but is not common across the whole corpus. In this project, TF-IDF is useful for detecting repeated scam language such as "
            "`OTP`, `account locked`, `verify`, `delivery fee`, `loan approved`, and Khmer scam phrases."
        ),
        nbf.v4.new_code_cell(
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "from sklearn.metrics.pairwise import cosine_similarity\n\n"
            "retrieval_df = realistic_df[realistic_df['label'].eq('scam')][['id', 'label', 'text', 'source']].copy()\n"
            "retrieval_df = retrieval_df[retrieval_df['text'].astype(str).str.len() > 8].reset_index(drop=True)\n"
            "retrieval_df['scam_type'] = retrieval_df['text'].apply(lambda x: simple_scam_type(str(x)) if 'simple_scam_type' in globals() else 'scam pattern')\n\n"
            "tfidf_retriever = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=50000, lowercase=True)\n"
            "tfidf_matrix = tfidf_retriever.fit_transform(retrieval_df['text'].astype(str))\n"
            "print('Scam retrieval corpus:', retrieval_df.shape)\n"
            "print('TF-IDF matrix:', tfidf_matrix.shape)"
        ),
        nbf.v4.new_markdown_cell(
            "### Cosine Similarity Formula\n\n"
            "Cosine similarity measures the angle between two vectors. It is common in vector-space information retrieval:\n\n"
            "$$\\cos(\\theta)=\\frac{A \\cdot B}{\\lVert A \\rVert \\lVert B \\rVert}$$\n\n"
            "A score closer to 1 means the new message is more similar to a known scam example."
        ),
        nbf.v4.new_code_cell(
            "def retrieve_top_tfidf_scam_messages(query, top_k=5):\n"
            "    query_vec = tfidf_retriever.transform([query])\n"
            "    scores = cosine_similarity(query_vec, tfidf_matrix).ravel()\n"
            "    top_idx = scores.argsort()[::-1][:top_k]\n"
            "    result = retrieval_df.iloc[top_idx][['scam_type', 'text', 'source']].copy()\n"
            "    result.insert(0, 'cosine_similarity', [round(float(scores[i]), 4) for i in top_idx])\n"
            "    return result\n\n"
            "sample_query = 'Dear customer, send your account number, OTP, and PIN to this email for verification.'\n"
            "retrieve_top_tfidf_scam_messages(sample_query, top_k=5)"
        ),
        nbf.v4.new_markdown_cell(
            "### Top-k Retrieval\n\n"
            "The previous cell returns the top 5 most similar scam messages. This is an IR-style result list: instead of only giving a class label, "
            "the system retrieves nearest examples from the corpus. This helps explain *why* a message is suspicious."
        ),
        nbf.v4.new_markdown_cell(
            "### Embedding-Based Semantic Retrieval\n\n"
            "TF-IDF is lexical: it works best when messages share similar words or character patterns. "
            "Embeddings are semantic: they can retrieve similar meaning even when the exact words are different. "
            "For Khmer-English text, a multilingual sentence-transformer model is appropriate, for example "
            "`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`."
        ),
        nbf.v4.new_code_cell(
            "try:\n"
            "    from sentence_transformers import SentenceTransformer\n"
            "    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')\n"
            "    embedding_sample = retrieval_df.sample(min(1800, len(retrieval_df)), random_state=42).reset_index(drop=True)\n"
            "    scam_embeddings = embedding_model.encode(\n"
            "        embedding_sample['text'].astype(str).tolist(),\n"
            "        normalize_embeddings=True,\n"
            "        show_progress_bar=False,\n"
            "    )\n"
            "    print('Embedding matrix:', scam_embeddings.shape)\n"
            "except Exception as exc:\n"
            "    embedding_model = None\n"
            "    embedding_sample = None\n"
            "    scam_embeddings = None\n"
            "    print('Embedding retrieval is optional. Install with: python -m pip install sentence-transformers torch')\n"
            "    print('Reason:', exc)"
        ),
        nbf.v4.new_code_cell(
            "def retrieve_top_embedding_scam_messages(query, top_k=5):\n"
            "    if embedding_model is None or scam_embeddings is None:\n"
            "        return pd.DataFrame({'message': ['Embedding model is not available in this environment.']})\n"
            "    query_embedding = embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)\n"
            "    scores = cosine_similarity(query_embedding, scam_embeddings).ravel()\n"
            "    top_idx = scores.argsort()[::-1][:top_k]\n"
            "    result = embedding_sample.iloc[top_idx][['scam_type', 'text', 'source']].copy()\n"
            "    result.insert(0, 'embedding_similarity', [round(float(scores[i]), 4) for i in top_idx])\n"
            "    return result\n\n"
            "retrieve_top_embedding_scam_messages(sample_query, top_k=5)"
        ),
        nbf.v4.new_markdown_cell(
            "### Lexical Similarity vs Semantic Similarity\n\n"
            "| Approach | Representation | Strength | Limitation | Use in this project |\n"
            "|---|---|---|---|---|\n"
            "| TF-IDF + cosine similarity | Sparse term/character vectors | Fast, explainable, strong for exact scam phrases and Khmer character patterns | May miss paraphrases | Shows top 5 lexically similar scam messages |\n"
            "| Multilingual embeddings | Dense semantic vectors | Finds similar meaning across different wording and mixed Khmer-English text | Requires larger dependency/model download | Embedding Explorer in Streamlit retrieves semantically similar scam patterns |\n"
            "| Hybrid classifier | TF-IDF, engineered features, and rules | Gives final Safe/Scam decision and risk level | Less transparent alone | Combined with retrieval results for explainability |\n\n"
            "In the final Streamlit app, the **IR Explorer** tab uses both retrieval approaches. The classifier answers *Is this risky?*, while retrieval answers *Which known scam patterns does it resemble?*"
        ),
    ]

    nb.cells.extend(cells)
    nbf.write(nb, NOTEBOOK)
    print(f"Added IR retrieval section to {NOTEBOOK}")


if __name__ == "__main__":
    main()
