from app.retrieval.hybrid_search import (
    HybridRetriever,
)


def main():

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    query = (
        "What are the major risks "
        "to global economic growth?"
    )

    print()
    print("Query:")
    print(query)

    print()
    print("Initializing hybrid retriever...")

    retriever = HybridRetriever()

    print()
    print("Running Dense + BM25 retrieval...")

    results = retriever.search(
        query=query,
        top_k=10,
        candidate_k=20,
    )

    print()
    print(
        f"Final results: {len(results)}"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()
        print("-" * 70)

        print(
            f"Final Rank: {rank}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Dense Rank: "
            f"{result['dense_rank']}"
        )

        print(
            f"BM25 Rank: "
            f"{result['bm25_rank']}"
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

        if len(text) > 450:
            text = text[:450] + "..."

        print(text)

    print()
    print("=" * 70)
    print("HYBRID RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()