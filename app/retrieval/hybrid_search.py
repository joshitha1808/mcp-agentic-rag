from typing import Dict, List

from app.retrieval.dense_search import (
    DenseRetriever,
)

from app.retrieval.bm25 import (
    BM25Retriever,
)


class HybridRetriever:
    """
    Hybrid retrieval using:

    1. Dense semantic retrieval
    2. BM25 lexical retrieval
    3. Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        dense_model: str = "nomic-embed-text",
        rrf_k: int = 60,
    ):
        self.dense_retriever = (
            DenseRetriever(
                embedding_model=dense_model
            )
        )

        self.bm25_retriever = (
            BM25Retriever()
        )

        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 20,
    ) -> List[Dict]:
        """
        Perform hybrid retrieval.

        Parameters
        ----------
        query:
            User's natural-language query.

        top_k:
            Number of final hybrid results.

        candidate_k:
            Number of candidates retrieved from
            each retrieval method before fusion.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # ------------------------------------------------
        # 1. Dense retrieval
        # ------------------------------------------------

        dense_results = (
            self.dense_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        # ------------------------------------------------
        # 2. BM25 retrieval
        # ------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        # ------------------------------------------------
        # 3. Create lookup table
        # ------------------------------------------------

        combined = {}

        # ------------------------------------------------
        # 4. Add Dense rankings
        # ------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id not in combined:

                combined[chunk_id] = {
                    **result,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "rrf_score": 0.0,
                }

            combined[
                chunk_id
            ]["dense_rank"] = rank

            combined[
                chunk_id
            ]["rrf_score"] += (
                1
                / (
                    self.rrf_k
                    + rank
                )
            )

        # ------------------------------------------------
        # 5. Add BM25 rankings
        # ------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id not in combined:

                combined[chunk_id] = {
                    **result,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "rrf_score": 0.0,
                }

            combined[
                chunk_id
            ]["bm25_rank"] = rank

            combined[
                chunk_id
            ]["rrf_score"] += (
                1
                / (
                    self.rrf_k
                    + rank
                )
            )

        # ------------------------------------------------
        # 6. Sort by RRF score
        # ------------------------------------------------

        results = sorted(
            combined.values(),
            key=lambda x: x[
                "rrf_score"
            ],
            reverse=True,
        )

        # ------------------------------------------------
        # 7. Return top results
        # ------------------------------------------------

        return results[:top_k]