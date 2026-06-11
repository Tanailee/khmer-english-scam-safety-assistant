from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    print(f"Downloaded and cached {MODEL_NAME}")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")


if __name__ == "__main__":
    main()
