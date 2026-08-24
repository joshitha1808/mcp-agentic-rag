from typing import Any

from app.mcp.server import (
    search_documents,
    hybrid_search,
    rerank_results,
    get_document_page,
    evaluate_answer,
    ask_question,
    ask_documents,
    summarize_document,
)


class MCPClient:
    """
    Client-side wrapper around MCP server tools.
    """

    # ============================================================
    # SEARCH DOCUMENTS
    # ============================================================

    def search_documents(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        return search_documents(
            query=query,
            top_k=top_k,
        )

    # ============================================================
    # HYBRID SEARCH
    # ============================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:

        return hybrid_search(
            query=query,
            top_k=top_k,
        )

    # ============================================================
    # RERANK
    # ============================================================

    def rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        return rerank_results(
            query=query,
            results=results,
            top_k=top_k,
        )

    # ============================================================
    # GET DOCUMENT PAGE
    # ============================================================

    def get_document_page(
        self,
        document_id: str,
        page_number: int,
    ) -> dict[str, Any]:

        return get_document_page(
            document_id=document_id,
            page_number=page_number,
        )

    # ============================================================
    # EVALUATE ANSWER
    # ============================================================

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        context: list[dict[str, Any]],
    ) -> dict[str, Any]:

        return evaluate_answer(
            question=question,
            answer=answer,
            context=context,
        )

    # ============================================================
    # ASK QUESTION
    # ============================================================

    def ask_question(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:

        return ask_question(
            question=question,
            top_k=top_k,
        )

    # ============================================================
    # ASK DOCUMENTS
    # ============================================================

    def ask_documents(
        self,
        question: str,
    ) -> dict[str, Any]:

        return ask_documents(
            question=question,
        )

    # ============================================================
    # SUMMARIZE DOCUMENT
    # ============================================================

    def summarize_document(
        self,
        document_id: str,
    ) -> dict[str, Any]:

        return summarize_document(
            document_id=document_id,
        )