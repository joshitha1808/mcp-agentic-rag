import streamlit as st
from pathlib import Path

from app.agent.agent import RAGAgent
from app.mcp.tools import summarize_document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Agentic RAG")
st.caption(
    "PDF Question Answering using Hybrid Search, "
    "Reranking and Ollama"
)


# ============================================================
# INITIALIZE AGENT
# ============================================================

if "agent" not in st.session_state:

    st.session_state.agent = RAGAgent()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This system uses:

        • PDF document ingestion  
        • BM25 search  
        • Dense vector search  
        • Reciprocal Rank Fusion  
        • Cross Encoder reranking  
        • Ollama LLM  
        • Agentic routing  
        """
    )

    st.divider()

    st.write(
        "**Supported documents:**"
    )

    st.write(
        "📄 World Development Report 2024"
    )

    st.write(
        "📄 Global Economic Prospects 2024"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

# -----------------------------
# DOCUMENT SUMMARY
# -----------------------------

with st.expander("📄 Document Summary", expanded=False):

    documents_dir = Path("data/documents")

    pdf_files = sorted(
        documents_dir.glob("*.pdf")
    )

    options = [p.name for p in pdf_files]

    if options:
        selected_doc = st.selectbox(
            "Select a document to summarize",
            options,
        )

        if st.button("Generate Summary"):

            doc_id = (
                selected_doc.replace(".pdf", "")
                .lower()
                .replace(" ", "_")
            )

            with st.spinner("Generating summary..."):

                result = summarize_document(
                    document_id=doc_id
                )

            if result.get("error"):
                st.error(result["error"])
            else:
                st.subheader("Summary")
                st.markdown(result.get("summary", ""))

    else:
        st.write("No PDF documents found in data/documents.")


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your PDFs..."
)


if question:

    # --------------------------------------------------------
    # DISPLAY USER
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            result = st.session_state.agent.run(
                question
            )

        answer = result.get(
            "answer",
            ""
        )

        action = result.get(
            "action"
        )

        sources = result.get(
            "sources",
            []
        )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(answer)

        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        with st.expander(
            "🔍 Agent details"
        ):

            st.write(
                "Action:"
            )

            st.code(
                action or "None"
            )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    document = source.get(
                        "document",
                        "Unknown document"
                    )

                    page = source.get(
                        "page",
                        "Unknown page"
                    )

                    st.write(
                        f"📄 {document} — Page {page}"
                    )

    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )