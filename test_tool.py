import asyncio

from app.mcp.server import mcp


async def main():
    result = await mcp.call_tool(
        "ask_documents",
        {
            "question": "What is the World Development Report 2025 about?"
        }
    )

    print("\n========== MCP RESULT ==========\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())