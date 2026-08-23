from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


def main():

    print("=" * 70)
    print("CROSS-ENCODER RERANKING TEST")
    print("=" * 70)

    query = (
        "What are the major risks to "
        "global economic growth?"
    )

    print("\nQuery:")
    print(query)

    # --------------------------------------------------
    # STEP 1: Hybrid retrieval
    # --------------------------------------------------

    print("\nInitializing hybrid retriever...")

    hybrid = HybridRetriever()

    print("\nRunning hybrid retrieval...")

    candidates = hybrid.search(
        query,
        top_k=10,
    )

    print(
        f"\nHybrid candidates: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------
    # STEP 2: Cross Encoder
    # --------------------------------------------------

    print("\nInitializing Cross Encoder...")

    reranker = CrossEncoderReranker()

    print("\nReranking candidates...")

    results = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=5,
    )

    # --------------------------------------------------
    # STEP 3: Display results
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("FINAL RERANKED RESULTS")
    print("-" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
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

        print(
            "\nText:"
        )

        print(
            result["text"][:700]
        )

    print("\n" + "=" * 70)
    print("RERANKING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()