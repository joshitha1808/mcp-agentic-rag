import asyncio

from app.mcp.server import mcp


if __name__ == "__main__":
    asyncio.run(
        mcp.run_stdio_async()
    )