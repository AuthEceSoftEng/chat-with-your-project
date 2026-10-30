import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_ariadne import ChatAriadne
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
load_dotenv(override=True)

def get_model():
    providername = os.getenv("LLM_PROVIDER")
    modelname = os.getenv("LLM_MODEL")
    apikey = os.getenv("LLM_API_KEY")
    baseurl = os.getenv("LLM_BASE_URL")
    if providername == "google":
        return modelname, ChatGoogleGenerativeAI(model=modelname, google_api_key=apikey)
    elif providername == "ariadne":
        return modelname, ChatAriadne(model=modelname, api_key=apikey, base_url=baseurl, provider=os.getenv("LLM_PROVIDER_NAME"))
    elif providername == "openai":
        if baseurl is None or baseurl == "":
            return modelname, ChatOpenAI(model=modelname, api_key=apikey)
        return modelname, ChatOpenAI(model=modelname, api_key=apikey, base_url=baseurl)
    elif providername == "ollama":
        return modelname, ChatOllama(model=modelname, base_url=baseurl) 
    else:
        raise ValueError(f"Unsupported LLM provider: {providername}")
        exit()

def connect_to_mcp(model, server_config):
    # Configuration for your ALREADY RUNNING MCP server
    async def fetch_tools():
        client = MultiServerMCPClient(server_config)
        return await client.get_tools()
    
    # Pull tools from the server and bind them to the LLM
    try:
        tools = asyncio.run(fetch_tools())
    except Exception as e:
        print("Could not connect to MCP server:", e)
        exit()
    model_with_tools = model.bind_tools(tools)
    return model_with_tools, {t.name: t for t in tools}