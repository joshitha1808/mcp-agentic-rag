from app.ingestion.pdf_loader import (
    load_all_pdfs,
)

from app.ingestion.chunker import (
    chunk_pages,
)

from app.retrieval.bm25 import (
    BM25Retriever,
)


def main():

    print("=" * 60)
    print("BM25 INDEX BUILD")
    print("=" * 60)

    print()
    print("Loading PDF pages...")

    pages = load_all_pdfs()

    print(
        f"Total pages: {len(pages)}"
    )

    print()
    print("Creating chunks...")

    chunks = chunk_pages(
        pages
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    print()
    print("Building BM25 index...")

    retriever = BM25Retriever()

    retriever.build(
        chunks
    )

    print()
    print("=" * 60)
    print("BM25 INDEX CREATED")
    print("=" * 60)

    print(
        f"Indexed chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Index location: "
        f"{retriever.index_path}"
    )


if __name__ == "__main__":
    main()