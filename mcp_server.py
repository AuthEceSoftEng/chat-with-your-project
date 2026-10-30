from fastmcp import FastMCP
from tool_registry import TOOL_REGISTRY
from dotenv import load_dotenv
load_dotenv()
mcp = FastMCP("MyTools")
for name,handler in TOOL_REGISTRY.items():
    mcp.tool(name=name)(handler)

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)