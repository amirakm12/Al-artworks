import os
import asyncio
import mcp
from dotenv import load_dotenv
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()
API_URL = os.getenv("CONTEXT7_API_URL")

async def main():
    async with streamablehttp_client(API_URL) as session:
        await session.initialize()
        tools = await session.list_tools()
        print(f"Available tools: {', '.join(t.name for t in tools.tools)}")

if __name__ == "__main__":
    asyncio.run(main())
