from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("Scam_or_Not_Scam_IR_Project.ipynb")


OLD = """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

retrieval_df = realistic_df[realistic_df['label'].eq('scam')][['id', 'label', 'text', 'source']].copy()
retrieval_df = retrieval_df[retrieval_df['text'].astype(str).str.len() > 8].reset_index(drop=True)
retrieval_df['scam_type'] = retrieval_df['text'].apply(lambda x: simple_scam_type(str(x)) if 'simple_scam_type' in globals() else 'scam pattern')

tfidf_retriever = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=50000, lowercase=True)
tfidf_matrix = tfidf_retriever.fit_transform(retrieval_df['text'].astype(str))
print('Scam retrieval corpus:', retrieval_df.shape)
print('TF-IDF matrix:', tfidf_matrix.shape)"""


NEW = """from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if 'realistic_df' not in globals():
    realistic_path = Path('data/realistic_labeled_messages_with_features.csv')
    realistic_df = pd.read_csv(realistic_path)

if 'source' not in realistic_df.columns:
    realistic_df['source'] = 'unknown'

retrieval_df = realistic_df[realistic_df['label'].eq('scam')][['id', 'label', 'text', 'source']].copy()
retrieval_df = retrieval_df[retrieval_df['text'].astype(str).str.len() > 8].reset_index(drop=True)
retrieval_df['scam_type'] = 'Known scam example'

tfidf_retriever = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=50000, lowercase=True)
tfidf_matrix = tfidf_retriever.fit_transform(retrieval_df['text'].astype(str))
print('Scam retrieval corpus:', retrieval_df.shape)
print('TF-IDF matrix:', tfidf_matrix.shape)"""


def main() -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source == OLD:
            cell.source = NEW
            changed = True
    if changed:
        nbf.write(nb, NOTEBOOK)
        print("Fixed IR notebook retrieval dataset cell.")
    else:
        print("No matching cell found or already fixed.")


if __name__ == "__main__":
    main()
