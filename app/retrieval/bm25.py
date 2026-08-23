import pickle
import re

from pathlib import Path
from typing import Dict, List


DEFAULT_INDEX_PATH = Path(
    "storage/indexes/bm25.pkl"
)


class BM25Retriever:
    """
    BM25 keyword-based document retriever.

    BM25 focuses on lexical matching between
    the query and document chunks.
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
    ):

        self.index_path = Path(
            index_path
        )

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.bm25 = None
        self.chunks = []

    @staticmethod
    def tokenize(
        text: str,
    ) -> List[str]:
        """
        Convert text into simple lowercase tokens.
        """

        if not text:
            return []

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def build(
        self,
        chunks: List[Dict],
    ) -> None:
        """
        Build the BM25 index from document chunks.
        """

        if not chunks:
            raise ValueError(
                "Cannot build BM25 index "
                "from empty chunks."
            )

        from rank_bm25 import (
            BM25Okapi
        )

        self.chunks = chunks

        tokenized_documents = [
            self.tokenize(
                chunk["text"]
            )
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        self.save()

    def save(self) -> None:
        """
        Persist the BM25 index to disk.
        """

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        with open(
            self.index_path,
            "wb",
        ) as file:

            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunks": self.chunks,
                },
                file,
            )

    def load(self) -> None:
        """
        Load a previously saved BM25 index.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: "
                f"{self.index_path}"
            )

        with open(
            self.index_path,
            "rb",
        ) as file:

            data = pickle.load(file)

        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search the BM25 index.

        Returns ranked chunks with BM25 scores.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if self.bm25 is None:
            self.load()

        query_tokens = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:

            chunk = self.chunks[index]

            results.append(
                {
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "text": chunk[
                        "text"
                    ],
                    "document_id": chunk[
                        "document_id"
                    ],
                    "document": chunk[
                        "document"
                    ],
                    "page": chunk[
                        "page"
                    ],
                    "source_path": chunk[
                        "source_path"
                    ],
                    "score": float(
                        scores[index]
                    ),
                }
            )

        return results