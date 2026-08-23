from app.retrieval.bm25 import (
    BM25Retriever,
)


def main():

    print("=" * 60)
    print("BM25 RETRIEVAL TEST")
    print("=" * 60)

    query = (
        "geopolitical tensions "
        "economic growth"
    )

    print()
    print("Query:")
    print(query)

    retriever = BM25Retriever()

    print()
    print("Loading BM25 index...")

    retriever.load()

    print(
        f"Indexed chunks: "
        f"{len(retriever.chunks)}"
    )

    print()
    print("Searching...")

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print()
    print(
        f"Retrieved {len(results)} results"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()
        print("-" * 60)

        print(
            f"Rank: {rank}"
        )

        print(
            f"BM25 Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Document: "
            f"{result['document']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        text = result["text"]

        if len(text) > 400:
            text = text[:400] + "..."

        print()
        print(text)

    print()
    print("=" * 60)
    print("BM25 TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()