from pathlib import Path
from typing import Dict, List

import chromadb


DEFAULT_CHROMA_PATH = Path("storage/chroma")

COLLECTION_NAME = "document_chunks"


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store for document chunks.
    """

    def __init__(
        self,
        persist_directory: Path = DEFAULT_CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_directory = Path(
            persist_directory
        )

        # Create storage directory if it doesn't exist
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Create persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Create or load collection
        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": (
                        "Document chunk embeddings "
                        "for MCP Agentic RAG"
                    )
                },
            )
        )

    def count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
    ) -> None:
        """
        Store document chunks and their embeddings.
        """

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match "
                "number of embeddings."
            )

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": chunk["document_id"],
                "document": chunk["document"],
                "source_path": chunk["source_path"],
                "page": chunk["page"],
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict:
        """
        Perform vector similarity search.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        return self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )