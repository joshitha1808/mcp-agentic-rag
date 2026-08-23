from app.ingestion.pdf_loader import load_all_pdfs
from app.ingestion.chunker import chunk_pages
from app.retrieval.embeddings import OllamaEmbeddingService
from app.retrieval.vector_store import ChromaVectorStore


BATCH_SIZE = 32


def main():

    print("=" * 60)
    print("DOCUMENT INDEXING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load PDF pages
    # --------------------------------------------------

    print()
    print("Step 1: Extracting PDF pages...")

    pages = load_all_pdfs()

    print(
        f"Total pages extracted: {len(pages)}"
    )

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    print()
    print("Step 2: Creating chunks...")

    chunks = chunk_pages(pages)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    if not chunks:
        raise RuntimeError(
            "No chunks were created."
        )

    # --------------------------------------------------
    # 3. Initialize Ollama
    # --------------------------------------------------

    print()
    print("Step 3: Initializing Ollama...")

    embedding_service = OllamaEmbeddingService(
        model="nomic-embed-text"
    )

    print(
        f"Embedding model: "
        f"{embedding_service.model}"
    )

    # --------------------------------------------------
    # 4. Initialize ChromaDB
    # --------------------------------------------------

    print()
    print("Step 4: Initializing ChromaDB...")

    vector_store = ChromaVectorStore()

    existing_count = vector_store.count()

    print(
        f"Existing vectors: {existing_count}"
    )

    # --------------------------------------------------
    # 5. Generate embeddings and store them
    # --------------------------------------------------

    print()
    print("Step 5: Generating embeddings...")

    total = len(chunks)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = chunks[start:end]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        print(
            f"Processing chunks "
            f"{start + 1}-{end} "
            f"of {total}..."
        )

        try:

            embeddings = (
                embedding_service.embed_documents(
                    texts
                )
            )

            vector_store.add_chunks(
                batch,
                embeddings,
            )

            print(
                f"Successfully indexed "
                f"{end}/{total}"
            )

        except Exception as error:

            print()
            print(
                "ERROR while processing batch:"
            )

            print(error)

            print()
            print(
                f"Failed batch: "
                f"{start + 1}-{end}"
            )

            raise

    # --------------------------------------------------
    # 6. Final statistics
    # --------------------------------------------------

    final_count = vector_store.count()

    print()
    print("=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    print(
        f"Total chunks: {total}"
    )

    print(
        f"Total vectors in ChromaDB: "
        f"{final_count}"
    )

    print()
    print(
        "Persistent vector index created successfully."
    )


if __name__ == "__main__":
    main()