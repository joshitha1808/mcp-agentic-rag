import requests
from typing import Any


class RAGGenerator:

    def __init__(
        self,
        model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.ollama_url = ollama_url

    # ============================================================
    # BUILD CONTEXT
    # ============================================================

    def build_context(
        self,
        documents: list[dict[str, Any]],
    ) -> str:

        context_parts = []

        for doc in documents:

            text = doc.get("text", "").strip()

            if not text:
                continue

            document = doc.get(
                "document",
                "Unknown document"
            )

            page = doc.get(
                "page",
                "Unknown page"
            )

            context_parts.append(
                f"""
SOURCE:
Document: {document}
Page: {page}

{text}
"""
            )

        return "\n\n".join(context_parts)

    # ============================================================
    # GENERATE ANSWER
    # ============================================================

    def generate(
        self,
        question: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not question.strip():

            return {
                "answer": "Question cannot be empty.",
                "sources": [],
            }

        if not documents:

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the available documents."
                ),
                "sources": [],
            }

        context = self.build_context(documents)

        # --------------------------------------------------------
        # PROMPT
        # --------------------------------------------------------

        prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided
document context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not present in the context,
   clearly say that the information was not found.
4. Give a clear and concise answer.
5. Mention the relevant document and page when useful.
6. Do not mention that you are an AI model.
7. Do not mention internal retrieval or ranking systems.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        try:

            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            answer = data.get(
                "response",
                ""
            ).strip()

            if not answer:

                answer = (
                    "The model did not return an answer."
                )

        except requests.exceptions.ConnectionError:

            answer = (
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )

        except Exception as e:

            answer = (
                f"Error while generating answer: {str(e)}"
            )

        # --------------------------------------------------------
        # SOURCES
        # --------------------------------------------------------

        sources = []

        for doc in documents:

            sources.append(
                {
                    "document": doc.get(
                        "document"
                    ),
                    "page": doc.get(
                        "page"
                    ),
                    "chunk_id": doc.get(
                        "chunk_id"
                    ),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }