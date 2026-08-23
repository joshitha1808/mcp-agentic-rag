from app.retrieval.dense_search import (
    DenseRetriever,
)


def main():

    print("=" * 60)
    print("DENSE RETRIEVAL TEST")
    print("=" * 60)

    query = (
        "What are the major risks to global "
        "economic growth?"
    )

    print()
    print("Query:")
    print(query)

    print()
    print("Searching...")

    retriever = DenseRetriever()

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print()
    print(
        f"Retrieved {len(results)} results"
    )

    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print("-" * 60)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: {result['score']}"
        )

        print(
            f"Distance: "
            f"{result['distance']}"
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

        print()

        text = result["text"]

        if len(text) > 500:
            text = text[:500] + "..."

        print(text)

    print()
    print("=" * 60)
    print("DENSE RETRIEVAL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()