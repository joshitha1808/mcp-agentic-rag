from app.retrieval.embeddings import OllamaEmbeddingService


def main():
    service = OllamaEmbeddingService()

    text = (
        "Retrieval augmented generation combines "
        "information retrieval with language models."
    )

    embedding = service.embed_text(text)

    print("Embedding generated successfully.")
    print("Model:", service.model)
    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])


if __name__ == "__main__":
    main()