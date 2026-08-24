from typing import Any

from app.mcp.client import MCPClient


# ============================================================
# MCP CLIENT
# ============================================================

mcp_client = MCPClient()


# ============================================================
# ASK RAG
# ============================================================

def ask_rag(
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:

    if not question or not question.strip():

        return {
            "error": "Question cannot be empty."
        }

    try:

        return mcp_client.ask_question(
            question=question.strip(),
            top_k=top_k,
        )

    except Exception as e:

        return {
            "error": f"ASK_RAG failed: {str(e)}"
        }


# ============================================================
# SEARCH RAG
# ============================================================

def search_rag(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:

    if not query or not query.strip():

        return []

    try:

        return mcp_client.search_documents(
            query=query.strip(),
            top_k=top_k,
        )

    except Exception as e:

        print(
            f"SEARCH failed: {e}"
        )

        return []


# ============================================================
# GET PAGE
# ============================================================

def get_page(
    document_id: str,
    page_number: int,
) -> dict[str, Any]:
    """
    Get a specific page from a specific PDF.

    Example:

    document_id =
        world_development_report_2024

    page_number =
        13
    """

    if not document_id:

        return {
            "error": "Document ID cannot be empty."
        }

    if page_number < 1:

        return {
            "error": "Page number must be greater than 0."
        }

    try:

        return mcp_client.get_document_page(
            document_id=document_id,
            page_number=page_number,
        )

    except Exception as e:

        return {
            "error": (
                f"GET_PAGE failed: {str(e)}"
            )
        }


# ============================================================
# GET DOCUMENT PAGE
# ============================================================

def get_document_page(
    document_id: str,
    page_number: int,
) -> dict[str, Any]:

    return get_page(
        document_id=document_id,
        page_number=page_number,
    )


# ============================================================
# SUMMARIZE DOCUMENT
# ============================================================

def summarize_document(
    document_id: str,
) -> dict[str, Any]:
    """
    Summarize a document by ID using the MCP server summarize_document tool.
    """

    if not document_id or not document_id.strip():
        return {
            "error": "Document ID cannot be empty."
        }

    try:
        return mcp_client.summarize_document(
            document_id=document_id.strip(),
        )

    except Exception as e:
        return {
            "error": f"SUMMARIZE_DOCUMENT failed: {e}"
        }


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:

    if not query or not query.strip():

        return []

    try:

        return mcp_client.hybrid_search(
            query=query.strip(),
            top_k=top_k,
        )

    except Exception as e:

        print(
            f"HYBRID_SEARCH failed: {e}"
        )

        return []


# ============================================================
# RERANK
# ============================================================

def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:

    if not query or not query.strip():

        return []

    if not results:

        return []

    try:

        return mcp_client.rerank_results(
            query=query.strip(),
            results=results,
            top_k=top_k,
        )

    except Exception as e:

        print(
            f"RERANK failed: {e}"
        )

        return []


# ============================================================
# EVALUATE ANSWER
# ============================================================

def evaluate_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:

    if not question or not question.strip():

        return {
            "supported": False,
            "reason": "Question is empty.",
        }

    if not answer or not answer.strip():

        return {
            "supported": False,
            "reason": "Answer is empty.",
        }

    try:

        return mcp_client.evaluate_answer(
            question=question.strip(),
            answer=answer.strip(),
            context=sources,
        )

    except Exception as e:

        return {
            "supported": False,
            "reason": str(e),
        }