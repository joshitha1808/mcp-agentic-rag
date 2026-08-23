import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    server_params = StdioServerParameters(
        command=".venv\\Scripts\\python.exe",
        args=["-m", "app.mcp.server"],
    )

    async with stdio_client(server_params) as (
        read,
        write,
    ):

        async with ClientSession(
            read,
            write,
        ) as session:

            # Initialize MCP connection
            await session.initialize()

            # -----------------------------------------
            # List available tools
            # -----------------------------------------

            tools = await session.list_tools()

            print("\n=== MCP TOOLS ===")

            for tool in tools.tools:
                print("-", tool.name)

            # -----------------------------------------
            # Call ask_question
            # -----------------------------------------

            print("\n=== ASK QUESTION ===")

            result = await session.call_tool(
                "ask_question",
                {
                    "question": (
                        "What is the World Development "
                        "Report 2024 about?"
                    ),
                    "top_k": 5,
                },
            )

            print("\n=== RESULT ===")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())