import sys
from typing import Dict, List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a Cross Encoder.

    The Cross Encoder receives:
        (query, document)

    and directly predicts a relevance score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        # Send diagnostic messages to stderr.
        # MCP stdio uses stdout for JSON-RPC messages,
        # so normal print statements must not go to stdout.
        print(
            f"Loading Cross Encoder model: {model_name}",
            file=sys.stderr,
        )

        self.model = CrossEncoder(
            model_name
        )

        self.model_name = model_name

        print(
            "Cross Encoder loaded successfully.",
            file=sys.stderr,
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank retrieved documents against the query.

        Parameters
        ----------
        query:
            User's question.

        documents:
            Documents returned by hybrid retrieval.

        top_k:
            Number of final documents to return.

        Returns
        -------
        List of reranked documents.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            return []

        # ----------------------------------------------------
        # Create query-document pairs
        # ----------------------------------------------------

        pairs = []

        for document in documents:

            text = document.get(
                "text",
                ""
            )

            pairs.append(
                (
                    query,
                    text,
                )
            )

        # ----------------------------------------------------
        # Get Cross Encoder scores
        # ----------------------------------------------------

        scores = self.model.predict(
            pairs
        )

        # ----------------------------------------------------
        # Attach reranking score
        # ----------------------------------------------------

        reranked_documents = []

        for document, score in zip(
            documents,
            scores,
        ):

            result = document.copy()

            result["rerank_score"] = float(
                score
            )

            reranked_documents.append(
                result
            )

        # ----------------------------------------------------
        # Sort from highest relevance
        # ----------------------------------------------------

        reranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Return only top K
        # ----------------------------------------------------

        return reranked_documents[:top_k]