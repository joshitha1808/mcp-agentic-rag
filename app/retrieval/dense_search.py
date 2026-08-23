from typing import Dict, List

from app.retrieval.embeddings import (
    OllamaEmbeddingService,
)
from app.retrieval.vector_store import (
    ChromaVectorStore,
)


class DenseRetriever:
    """
    Semantic retriever using:

    Query
      ↓
    Ollama embedding
      ↓
    ChromaDB vector search
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
    ):

        self.embedding_service = (
            OllamaEmbeddingService(
                model=embedding_model
            )
        )

        self.vector_store = (
            ChromaVectorStore()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Perform semantic vector search.

        Returns a list of dictionaries containing:

        - chunk_id
        - text
        - document
        - document_id
        - page
        - score
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # ------------------------------------------
        # Generate query embedding
        # ------------------------------------------

        query_embedding = (
            self.embedding_service.embed_text(
                query
            )
        )

        # ------------------------------------------
        # Search ChromaDB
        # ------------------------------------------

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # ------------------------------------------
        # Convert Chroma response
        # into clean result format
        # ------------------------------------------

        formatted_results = []

        ids = results.get("ids", [[]])[0]
        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        for i in range(len(ids)):

            metadata = (
                metadatas[i]
                if i < len(metadatas)
                else {}
            )

            distance = (
                distances[i]
                if i < len(distances)
                else None
            )

            # Chroma returns distance.
            # Smaller distance = more similar.
            #
            # Convert it into an intuitive score.
            if distance is not None:
                score = 1 / (
                    1 + distance
                )
            else:
                score = 0.0

            formatted_results.append(
                {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "document_id": metadata.get(
                        "document_id"
                    ),
                    "document": metadata.get(
                        "document"
                    ),
                    "page": metadata.get(
                        "page"
                    ),
                    "source_path": metadata.get(
                        "source_path"
                    ),
                    "score": round(
                        score,
                        6,
                    ),
                    "distance": distance,
                }
            )

        return formatted_results