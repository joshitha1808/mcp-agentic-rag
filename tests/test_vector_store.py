from app.retrieval.vector_store import ChromaVectorStore


def main():
    print("=" * 60)
    print("CHROMADB TEST")
    print("=" * 60)

    store = ChromaVectorStore()

    print("Stored chunks:", store.count())

    print()
    print(
        "ChromaDB persistent store initialized successfully."
    )


if __name__ == "__main__":
    main()