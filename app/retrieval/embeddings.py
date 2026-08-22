from typing import List

import requests

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
)


class OllamaEmbeddingService:
    """
    Generates text embeddings using a locally running Ollama model.
    """

    def __init__(
        self,
        model: str = OLLAMA_EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        """

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        return embeddings[0]

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        return embeddings