from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.ingestion.pdf_loader import extract_pdf_pages
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.rag.pipeline import RAGPipeline
   
# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP(
    "mcp-agentic-rag"
)

# ============================================================
# SHARED SERVICES
# ============================================================

hybrid_retriever = HybridRetriever(
    dense_model="nomic-embed-text",
    rrf_k=60,
)

reranker = CrossEncoderReranker()

rag_pipeline = RAGPipeline(
    retriever=hybrid_retriever,
    reranker=reranker,
)


# ============================================================
# CONTEXT CONSTRUCTION
# ============================================================

def build_context(
    reranked_results: list[dict[str, Any]]
) -> str:
    """
    Combine the text from reranked document chunks
    into a single context string for the LLM.
    """

    context_parts = []

    for result in reranked_results:
        text = result.get("text", "")

        if text.strip():
            context_parts.append(text)

    return "\n\n".join(context_parts)


# ============================================================
# TOOL 1 — SEARCH DOCUMENTS
# ============================================================

@mcp.tool()
def search_documents(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Search the document collection using hybrid retrieval
    followed by cross-encoder reranking.
    """

    if not query.strip():
        return []

    candidates = hybrid_retriever.search(
        query=query,
        top_k=10,
        candidate_k=20,
    )

    results = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=top_k,
    )

    return results


# ============================================================
# TOOL 2 — HYBRID SEARCH
# ============================================================

@mcp.tool()
def hybrid_search(
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Search documents using hybrid BM25 + dense retrieval.
    """

    if not query.strip():
        return []

    return hybrid_retriever.search(
        query=query,
        top_k=top_k,
        candidate_k=20,
    )


# ============================================================
# TOOL 3 — RERANK RESULTS
# ============================================================

@mcp.tool()
def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved chunks using a cross-encoder.
    """

    if not query.strip():
        return []

    if not results:
        return []

    return reranker.rerank(
        query=query,
        documents=results,
        top_k=top_k,
    )


# ============================================================
# TOOL 4 — GET DOCUMENT PAGE
# ============================================================

@mcp.tool()
def get_document_page(
    document_id: str,
    page_number: int,
) -> dict[str, Any]:
    """
    Retrieve the complete text of a specific PDF page.
    """

    documents_dir = Path("data/documents")

    if page_number < 1:
        return {
            "error": "Page number must be greater than 0."
        }

    pdf_path = None

    for file in documents_dir.glob("*.pdf"):

        current_id = (
            file.stem
            .lower()
            .replace(" ", "_")
        )

        if current_id == document_id.lower():
            pdf_path = file
            break

    if pdf_path is None:
        return {
            "error": f"Document not found: {document_id}"
        }

    pages = extract_pdf_pages(pdf_path)

    for page in pages:

        if page["page"] == page_number:

            return {
                "document_id": document_id,
                "document": page["document"],
                "page": page_number,
                "text": page["text"],
                "source_path": page["source_path"],
            }

    return {
        "error": (
            f"Page {page_number} "
            f"not found in {document_id}"
        )
    }


# ============================================================
# TOOL 5 — EVALUATE ANSWER
# ============================================================

@mcp.tool()
def evaluate_answer(
    question: str,
    answer: str,
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Perform a lightweight groundedness check.
    """

    if not answer.strip():
        return {
            "supported": False,
            "reason": "Answer is empty.",
        }

    if not context:
        return {
            "supported": False,
            "reason": "No context was provided.",
        }

    context_text = " ".join(
        item.get("text", "")
        for item in context
    ).lower()

    answer_words = {
        word.strip(".,!?;:()[]{}\"'")
        for word in answer.lower().split()
        if len(word.strip(".,!?;:()[]{}\"'")) > 4
    }

    context_words = set(
        context_text.split()
    )

    overlap = answer_words.intersection(
        context_words
    )

    overlap_ratio = (
        len(overlap) / len(answer_words)
        if answer_words
        else 0
    )

    supported = overlap_ratio >= 0.20

    return {
        "supported": supported,
        "overlap_ratio": round(overlap_ratio, 3),
        "answer_word_count": len(answer_words),
        "matched_word_count": len(overlap),
        "reason": (
            "Answer has sufficient overlap "
            "with retrieved context."
            if supported
            else
            "Answer has insufficient overlap "
            "with retrieved context."
        ),
    }


# ============================================================
# TOOL 6 — ASK QUESTION
# ============================================================

@mcp.tool()
def ask_question(
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Answer a question using the complete RAG pipeline.

    Pipeline:

    1. Hybrid retrieval
    2. Cross-encoder reranking
    3. Context construction
    4. LLM generation
    5. Source attribution
    """

    if not question.strip():
        return {
            "error": "Question cannot be empty."
        }

    return rag_pipeline.ask(
        question=question,
        top_k=top_k,
    )


# ============================================================
# TOOL 7 — ASK DOCUMENTS
# ============================================================

@mcp.tool()
def ask_documents(
    question: str,
) -> dict[str, Any]:
    """
    Ask a question about the indexed PDF documents
    using the complete RAG pipeline.
    """

    if not question.strip():
        return {
            "error": "Question cannot be empty."
        }

    return rag_pipeline.ask(
        question=question,
    )

# ============================================================
# TOOL 8 — SUMMARIZE DOCUMENT
# ============================================================

@mcp.tool()
def summarize_document(
    document_id: str,
) -> dict[str, Any]:
    """
    Summarize an indexed PDF/document.

    Steps:
    1. Locate the PDF in data/documents by normalized ID.
    2. Extract its pages using extract_pdf_pages.
    3. Call the RAGGenerator.summarize_document method.
    4. Return {"document": <name>, "summary": <text>}.
    """

    documents_dir = Path("data/documents")

    pdf_path = None

    for file in documents_dir.glob("*.pdf"):

        current_id = (
            file.stem
            .lower()
            .replace(" ", "_")
        )

        if current_id == document_id.lower():
            pdf_path = file
            break

    if pdf_path is None:
        return {
            "error": f"Document not found: {document_id}"
        }

    try:
        pages = extract_pdf_pages(pdf_path)
    except Exception as e:
        return {
            "error": f"Failed to extract PDF pages: {e}"
        }

    if not pages:
        return {
            "error": "Document contains no extractable text."
        }

    try:
        summary = rag_pipeline.generator.summarize_document(
            pages
        )
    except Exception as e:
        return {
            "error": f"Summarization failed: {e}"
        }

    return {
        "document": pdf_path.name,
        "summary": summary,
    }

# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":
    mcp.run()