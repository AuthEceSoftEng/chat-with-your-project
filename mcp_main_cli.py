import argparse
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from model import get_model

MCP_SERVER_CONFIG = {
    "my_mcp_server": {
        "url": "http://localhost:8000/mcp",
        "transport": "http"
    }
}

async def run_agent(repo_name: str):
    _, aimodel = get_model()
    mcp_client = MultiServerMCPClient(MCP_SERVER_CONFIG)
    tools = await mcp_client.get_tools()
    tool_map = {t.name: t for t in tools}
    
    bound_model = aimodel.bind_tools(tools)

    try:
        with open("system_prompt.txt", "r") as infile:
            system_prompt_text = infile.read().strip().replace("repo_name", repo_name)
    except FileNotFoundError:
        system_prompt_text = f"You are an AI assistant for {repo_name}."
    
    messages = [SystemMessage(content=system_prompt_text)]
    if repo_name != "none":
        print(f"AI Assistant for '{repo_name}' is ready! (Type 'exit' to stop)")
    else:
        print("Use --repo to select a repository.")
        return

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        messages.append(HumanMessage(content=user_input))
        
        response = bound_model.invoke(messages)
        messages.append(response)
        
        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"  [Executing {tool_name} with args: {tool_args}]")
                
                try:
                    result = await tool_map[tool_name].ainvoke(tool_args)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(f"  [!] {error_msg}")
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
            
            response = bound_model.invoke(messages)
            messages.append(response)
            
        print(f"\nAI: {response.content}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP AI Agent for a specific repository.")
    parser.add_argument(
        "--repo", 
        type=str, 
        default="none", 
        help="The name of the repository (see gddata.repos.csv) for the names"
    )
    args = parser.parse_args()
    asyncio.run(run_agent(args.repo))