import os

import streamlit as st
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from model import get_model, connect_to_mcp


@st.cache_resource
def get_mcp_model():
    aimodelname, aimodel = get_model()

    server_config = {
        "MyTools": {
            "url": "http://localhost:8000/mcp",
            "transport": "http"
        }
    }

    aimodel, tool_map = connect_to_mcp(aimodel, server_config)

    return aimodelname, aimodel, tool_map


aimodelname, aimodel, tool_map = get_mcp_model()


if "repo_name" not in st.session_state:
    st.session_state.repo_name = "default_repo"

if "messages" not in st.session_state:
    try:
        with open("system_prompt.txt", "r") as infile:
            system_prompt_text = infile.read().strip().replace(
                "repo_name",
                st.session_state.repo_name
            )
    except FileNotFoundError:
        system_prompt_text = (
            f"You are an AI assistant for {st.session_state.repo_name}."
        )

    st.session_state.messages = [
        SystemMessage(content=system_prompt_text)
    ]

st.sidebar.header("Project Configuration")

csv_path = "gddata.repos.csv"
repo_options = []

if os.path.exists(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        repo_options = [line.strip() for line in f if line.strip()]

if not repo_options:
    repo_options = ["No repositories found"]

try:
    default_index = repo_options.index(st.session_state.repo_name)
except ValueError:
    default_index = 0

st.session_state.repo_name = st.sidebar.selectbox(
    "Select Repository",
    options=repo_options,
    index=default_index
)


def reset_project():
    st.cache_resource.clear()

    try:
        with open("system_prompt.txt", "r") as infile:
            system_prompt_text = infile.read().strip().replace(
                "repo_name",
                st.session_state.repo_name
            )
    except FileNotFoundError:
        system_prompt_text = (
            f"You are an AI assistant for {st.session_state.repo_name}."
        )

    st.session_state.messages = [
        SystemMessage(content=system_prompt_text)
    ]

    st.rerun()


if st.sidebar.button("Apply Repository Change"):
    reset_project()

st.title("Chat with your project AI Assistant")


for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content or msg.tool_calls:
            with st.chat_message("assistant"):
                if msg.content:
                    st.write(msg.content)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        st.status(
                            f"Used tool: {tc['name']}",
                            state="complete"
                        )
    elif isinstance(msg, ToolMessage):
        with st.expander(f"Tool Data ({msg.tool_call_id})"):
            st.code(msg.content)

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = aimodel.invoke(
            st.session_state.messages
        )
        while response.tool_calls:
            st.session_state.messages.append(response)
            for tc in response.tool_calls:
                with st.status(
                    f"Running {tc['name']}...",
                    expanded=False
                ) as status:
                    tool_args = dict(tc["args"])
                    tool_args["repo_name"] = st.session_state.repo_name
                    tool_result = asyncio.run(
                        tool_map[tc["name"]].ainvoke(tool_args)
                    )
                    st.session_state.messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tc["id"]
                        )
                    )
                    status.update(
                        label=f"Completed {tc['name']}",
                        state="complete"
                    )
            response = aimodel.invoke(
                st.session_state.messages
            )
        response_placeholder = st.empty()
        full_response = ""
        for chunk in aimodel.stream(
            st.session_state.messages
        ):
            if isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and "text" in part:
                        full_response += part["text"]
                    elif isinstance(part, str):
                        full_response += part
            else:
                full_response += chunk.content
            response_placeholder.markdown(
                full_response + "|"
            )
        response_placeholder.markdown(full_response)
        st.session_state.messages.append(
            AIMessage(content=full_response)
        )