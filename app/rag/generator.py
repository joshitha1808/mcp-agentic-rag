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

    # ============================================================
    # SUMMARIZE DOCUMENT
    # ============================================================

    def summarize_document(
        self,
        pages: list[dict[str, Any]],
    ) -> str:
        """
        Generate a concise professional summary for the supplied
        document pages. Uses a simple two-stage chunk-summary
        approach when the document is large:

        1. Split pages into chunks (based on configured CHUNK_SIZE).
        2. Summarize each chunk individually.
        3. Combine chunk summaries and ask the model for a final
           concise summary that identifies the main topic,
           important points/findings, and conclusions.

        Returns the final summary string. Raises RuntimeError on
        Ollama/generation failures.
        """

        from app.llm.ollama_client import OllamaClient
        from app.config import (
            OLLAMA_BASE_URL,
            OLLAMA_LLM_MODEL,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        def chunk_text(text: str) -> list[str]:
            if not text:
                return []

            chunks = []
            start = 0
            text_len = len(text)

            while start < text_len:
                end = start + CHUNK_SIZE
                chunk = text[start:end]

                # If possible, extend to end of sentence to avoid chopping
                if end < text_len:
                    # try to extend until next period/newline within 100 chars
                    extend_to = min(text_len, end + 100)
                    sep_pos = None
                    for sep in ['.\n', '\n', '.']:
                        pos = text.find(sep, end, extend_to)
                        if pos != -1:
                            sep_pos = pos + (1 if sep == '.' else len(sep))
                            break
                    if sep_pos:
                        chunk = text[start:sep_pos]
                        end = sep_pos

                chunks.append(chunk.strip())

                # advance with overlap
                start = max(end - CHUNK_OVERLAP, end)

            return [c for c in chunks if c]

        # Build raw chunks from pages
        raw_chunks: list[str] = []

        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue
            # If page is small, keep as-is to preserve page boundaries
            if len(text) <= CHUNK_SIZE:
                raw_chunks.append(text)
            else:
                raw_chunks.extend(chunk_text(text))

        if not raw_chunks:
            raise RuntimeError("No extractable text found in document pages.")

        ollama = OllamaClient(
            model=getattr(self, "model", OLLAMA_LLM_MODEL),
            base_url=OLLAMA_BASE_URL,
        )

        # If only one chunk, summarize directly
        if len(raw_chunks) == 1:
            prompt = f"""
Summarize ONLY the following document content. Produce a concise, professional summary (3-6 sentences) that:
- Identifies the main topic
- Highlights the most important points/findings
- Mentions important conclusions
- Avoids adding information not present in the text

DOCUMENT CONTENT:

{raw_chunks[0]}

SUMMARY:
"""
            try:
                return ollama.generate(prompt=prompt, temperature=0.2)
            except Exception as e:
                raise RuntimeError(f"Summarization failed: {e}")

        # Multi-chunk: summarize each chunk, then combine
        chunk_summaries = []

        for idx, chunk in enumerate(raw_chunks, start=1):
            prompt_chunk = f"""
Summarize ONLY the following document chunk. Keep it very brief (2-4 concise bullet points and a one-sentence conclusion). Do not add outside information.

CHUNK {idx} CONTENT:

{chunk}

CHUNK {idx} SUMMARY:
"""
            try:
                summary = ollama.generate(
                    prompt=prompt_chunk,
                    temperature=0.2,
                )
            except Exception as e:
                raise RuntimeError(f"Chunk summarization failed: {e}")

            chunk_summaries.append(f"Chunk {idx}: {summary}")

        # Combine chunk summaries into final prompt
        combined = "\n\n".join(chunk_summaries)

        final_prompt = f"""
Combine the following chunk summaries into a single concise, professional document summary. The final summary should:
- Identify the main topic
- Identify the most important points/findings
- Mention important conclusions
- Not introduce any information not present in the chunk summaries
- Be suitable for an executive summary (3-6 sentences)

CHUNK SUMMARIES:

{combined}

FINAL SUMMARY:
"""

        try:
            final_summary = ollama.generate(
                prompt=final_prompt,
                temperature=0.2,
            )
        except Exception as e:
            raise RuntimeError(f"Final summarization failed: {e}")

        return final_summary.strip()