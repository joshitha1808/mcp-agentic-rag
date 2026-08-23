from typing import Any, Dict


class AgentRouter:
    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools

    def route_query(self, question: str) -> str:
        question_lower = question.lower()

        if "document" in question_lower:
            return "ask_documents"
        elif "search" in question_lower:
            return "search_documents"
        elif "evaluate" in question_lower:
            return "evaluate_answer"
        elif "rerank" in question_lower:
            return "rerank_results"
        elif "hybrid" in question_lower:
            return "hybrid_search"
        else:
            return "ask_question"

    def handle_query(self, question: str):
        tool_name = self.route_query(question)

        if tool_name in self.tools:
            return self.tools[tool_name](question)

        return {"error": "No suitable tool found for the query."}


# Backward compatibility
QueryRouter = AgentRouter


def handle_query(question: str, tools: Dict[str, Any]):
    router = AgentRouter(tools)
    return router.handle_query(question)