from typing import Any

from app.mcp.server import (
    search_documents,
    hybrid_search,
    get_document_page,
    evaluate_answer,
)


class RAGOrchestrator:

    def __init__(self):
        self.max_iterations = 2

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        results = search_documents(
            query=query,
            top_k=top_k,
        )

        return results

    def build_context(
        self,
        documents: list[dict[str, Any]],
    ) -> str:

        context_parts = []

        for i, document in enumerate(documents, start=1):

            context_parts.append(
                f"""
SOURCE {i}

Document: {document.get("document")}
Page: {document.get("page")}

Content:
{document.get("text", "")}
"""
            )

        return "\n".join(context_parts)

    def run(
        self,
        question: str,
    ) -> dict[str, Any]:

        if not question.strip():

            return {
                "answer": "",
                "sources": [],
                "iterations": 0,
                "error": "Question cannot be empty.",
            }

        documents = self.retrieve(
            question,
            top_k=5,
        )

        context = self.build_context(
            documents
        )

        return {
            "question": question,
            "context": context,
            "sources": documents,
            "iterations": 1,
        }