from app.agent.agent import RAGAgent


def test_ask_rag():
    agent = RAGAgent()

    assert agent.decide_action(
        "What are the major risks to global economic growth?"
    ) == "ASK_RAG"


def test_search():
    agent = RAGAgent()

    assert agent.decide_action(
        "Search for geopolitical tensions."
    ) == "SEARCH"


def test_page_lookup():
    agent = RAGAgent()

    assert agent.decide_action(
        "Show me page 110 of Global Economic Prospects 2024."
    ) == "PAGE_LOOKUP"


def test_hybrid_search():
    agent = RAGAgent()

    assert agent.decide_action(
        "Perform hybrid search for climate risks."
    ) == "HYBRID_SEARCH"


def test_rerank():
    agent = RAGAgent()

    assert agent.decide_action(
        "Rerank the results for geopolitical tensions."
    ) == "RERANK"