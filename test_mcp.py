import asyncio

from app.mcp.server import mcp


async def main():
    print("MCP SERVER LOADED:", mcp.name)

    tools = await mcp.list_tools()

    print("\nTOOLS:")
    for tool in tools:
        print("-", tool.name)


if __name__ == "__main__":
    asyncio.run(main())