from typing import Any
import re

from app.mcp.tools import (
    ask_rag,
    search_rag,
    get_page,
    hybrid_search,
    rerank_results,
)


class RAGAgent:

    # ============================================================
    # STEP 1 — DECIDE ACTION
    # ============================================================

    def decide_action(
        self,
        question: str,
    ) -> str:

        q = question.lower().strip()

        # --------------------------------------------------------
        # PAGE LOOKUP
        # --------------------------------------------------------

        if re.search(
            r"\bpage\s*(?:number\s*)?\d+\b",
            q,
        ):
            return "PAGE_LOOKUP"

        # --------------------------------------------------------
        # RERANK
        # --------------------------------------------------------

        rerank_keywords = [
            "rerank",
            "re-rank",
            "re rank",
        ]

        if any(
            keyword in q
            for keyword in rerank_keywords
        ):
            return "RERANK"

        # --------------------------------------------------------
        # HYBRID SEARCH
        # --------------------------------------------------------

        hybrid_keywords = [
            "hybrid search",
            "hybrid retrieval",
            "bm25 and dense",
            "bm25 + dense",
        ]

        if any(
            keyword in q
            for keyword in hybrid_keywords
        ):
            return "HYBRID_SEARCH"

        # --------------------------------------------------------
        # SEARCH
        # --------------------------------------------------------

        search_keywords = [
            "search for",
            "search information",
            "search documents",
            "find information",
            "find passages",
            "find documents",
            "locate information",
            "show passages",
        ]

        if any(
            keyword in q
            for keyword in search_keywords
        ):
            return "SEARCH"

        # --------------------------------------------------------
        # DEFAULT
        # --------------------------------------------------------

        return "ASK_RAG"

    # ============================================================
    # STEP 2 — EXTRACT PAGE NUMBER
    # ============================================================

    def extract_page_number(
        self,
        question: str,
    ) -> int | None:

        match = re.search(
            r"\bpage\s*(?:number\s*)?(\d+)\b",
            question,
            re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

        return None

    # ============================================================
    # STEP 3 — EXTRACT DOCUMENT
    # ============================================================

    def extract_document_name(
        self,
        question: str,
    ) -> str | None:

        documents = [
            "World Development Report 2024.pdf",
            "Global Economic Prospects 2024.pdf",
        ]

        question_lower = question.lower()

        for document in documents:

            name_without_extension = document[:-4]

            if (
                name_without_extension.lower()
                in question_lower
            ):
                return document

        return None

    # ============================================================
    # STEP 4 — DOCUMENT ID
    # ============================================================

    def document_to_id(
        self,
        document: str,
    ) -> str:

        return (
            document
            .replace(".pdf", "")
            .lower()
            .replace(" ", "_")
        )

    # ============================================================
    # STEP 5 — PAGE LOOKUP
    # ============================================================

    def run_page_lookup(
        self,
        question: str,
    ) -> dict[str, Any]:

        page_number = self.extract_page_number(
            question
        )

        if page_number is None:

            return {
                "answer": "Please specify a page number.",
                "sources": [],
            }

        document = self.extract_document_name(
            question
        )

        if document is None:

            return {
                "answer": (
                    f"I found page {page_number}, but I need "
                    "the document name.\n\n"
                    "Example:\n"
                    f"What is on page {page_number} of "
                    "Global Economic Prospects 2024?"
                ),
                "sources": [],
            }

        document_id = self.document_to_id(
            document
        )

        result = get_page(
            document_id=document_id,
            page_number=page_number,
        )

        if result.get("error"):

            return {
                "answer": result["error"],
                "sources": [],
            }

        return {
            "answer": result.get("text", ""),
            "sources": [
                {
                    "document": document,
                    "page": page_number,
                }
            ],
        }

    # ============================================================
    # STEP 6 — MAIN AGENT
    # ============================================================

    def run(
        self,
        question: str,
    ) -> dict[str, Any]:

        if not question or not question.strip():

            return {
                "answer": "",
                "action": None,
                "sources": [],
                "error": "Question cannot be empty.",
            }

        question = question.strip()

        # --------------------------------------------------------
        # AGENT DECIDES ACTION
        # --------------------------------------------------------

        action = self.decide_action(question)

        # ========================================================
        # ASK RAG
        # ========================================================

        if action == "ASK_RAG":

            result = ask_rag(
                question=question,
                top_k=5,
            )

            return {
                "answer": result.get("answer", ""),
                "action": action,
                "sources": result.get("sources", []),
            }

        # ========================================================
        # SEARCH
        # ========================================================

        if action == "SEARCH":

            results = search_rag(
                query=question,
                top_k=5,
            )

            return {
                "answer": (
                    "I found the following relevant "
                    "document passages."
                ),
                "action": action,
                "results": results,
                "sources": results,
            }

        # ========================================================
        # PAGE LOOKUP
        # ========================================================

        if action == "PAGE_LOOKUP":

            result = self.run_page_lookup(
                question
            )

            return {
                "answer": result.get("answer", ""),
                "action": action,
                "sources": result.get("sources", []),
            }

        # ========================================================
        # HYBRID SEARCH
        # ========================================================

        if action == "HYBRID_SEARCH":

            results = hybrid_search(
                query=question,
                top_k=10,
            )

            return {
                "answer": (
                    "Hybrid retrieval returned the "
                    "following relevant passages."
                ),
                "action": action,
                "results": results,
                "sources": results,
            }

        # ========================================================
        # RERANK
        # ========================================================

        if action == "RERANK":

            # First retrieve candidates
            candidates = hybrid_search(
                query=question,
                top_k=10,
            )

            # Then rerank candidates
            results = rerank_results(
                query=question,
                results=candidates,
                top_k=5,
            )

            return {
                "answer": (
                    "The retrieved passages were "
                    "reranked using the cross-encoder."
                ),
                "action": action,
                "results": results,
                "sources": results,
            }

        # ========================================================
        # FALLBACK
        # ========================================================

        return {
            "answer": "Unable to determine the required action.",
            "action": action,
            "sources": [],
        }


# ================================================================
# TERMINAL TEST
# ================================================================

if __name__ == "__main__":

    agent = RAGAgent()

    print("=" * 60)
    print("AGENTIC RAG")
    print("=" * 60)

    while True:

        try:

            question = input(
                "\nEnter your question: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print("\nExiting...")
            break

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print("Exiting...")
            break

        if not question:

            print("Please enter a question.")
            continue

        result = agent.run(question)

        print("\n" + "=" * 60)
        print("ACTION")
        print("=" * 60)

        print(result.get("action"))

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(result.get("answer"))

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        print(result.get("sources", []))