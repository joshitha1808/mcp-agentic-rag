import streamlit as st

from app.agent.agent import RAGAgent


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