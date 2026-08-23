from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.rag.generator import RAGGenerator


class RAGPipeline:

    def __init__(
        self,
        retriever=None,
        reranker=None,
        generator=None,
    ):

        # ========================================================
        # RETRIEVER
        # ========================================================

        self.retriever = (
            retriever
            if retriever is not None
            else HybridRetriever(
                dense_model="nomic-embed-text",
                rrf_k=60,
            )
        )

        # ========================================================
        # RERANKER
        # ========================================================

        self.reranker = (
            reranker
            if reranker is not None
            else CrossEncoderReranker()
        )

        # ========================================================
        # LLM GENERATOR
        # ========================================================

        self.generator = (
            generator
            if generator is not None
            else RAGGenerator(
                model="llama3.2"
            )
        )

    # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ):

        if not question or not question.strip():

            return {
                "question": question,
                "answer": "Question cannot be empty.",
                "sources": [],
                "retrieved_documents": [],
            }

        question = question.strip()

        # ========================================================
        # STEP 1 — HYBRID RETRIEVAL
        # ========================================================

        candidates = self.retriever.search(
            query=question,
            top_k=10,
            candidate_k=20,
        )

        # ========================================================
        # STEP 2 — RERANK
        # ========================================================

        reranked = self.reranker.rerank(
            query=question,
            documents=candidates,
            top_k=top_k,
        )

        # ========================================================
        # STEP 3 — OLLAMA GENERATION
        # ========================================================

        result = self.generator.generate(
            question=question,
            documents=reranked,
        )

        # ========================================================
        # RETURN
        # ========================================================

        return {
            "question": question,
            "answer": result.get(
                "answer",
                ""
            ),
            "sources": result.get(
                "sources",
                []
            ),
            "retrieved_documents": reranked,
        }