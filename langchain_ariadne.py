import json
import requests
from langchain_core.messages import AIMessage

class ChatAriadne:
    def __init__(self, model, api_key, base_url, provider):
        self.PROV_NAME = provider
        self.MODEL = model
        self.API_KEY = api_key
        self.BASE_URL = base_url
        self._tools = None

    def bind_tools(self, tools):
        """Return a copy of this model with tools bound (LangChain compatible)."""
        bound = ChatAriadne(self.MODEL, self.API_KEY, self.BASE_URL, self.PROV_NAME)
        bound._tools = [self._convert_tool(t) for t in tools]
        return bound

    @staticmethod
    def _convert_tool(tool):
        """Convert a LangChain StructuredTool/BaseTool to Ariadne's Anthropic-style JSON format."""
        s = tool.args_schema
        if s is None:
            schema = {}
        elif isinstance(s, dict):
            schema = s
        else:
            schema = s.schema()  # Pydantic model class
        return {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        }

    def _to_api_msgs(self, messages):
        result = []
        for m in messages:
            if m.type == "system":
                result.append({"role": "system", "content": m.content})
            elif m.type == "human":
                result.append({"role": "user", "content": m.content})
            elif m.type == "tool":
                # LangChain ToolMessage > Anthropic tool_result block
                result.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": str(m.content)}
                ]})
            elif m.type == "ai":
                tool_calls = getattr(m, "tool_calls", None)
                if tool_calls:
                    content = []
                    if m.content:
                        content.append({"type": "text", "text": m.content})
                    for tc in tool_calls:
                        content.append({"type": "tool_use", "id": tc.get("id"), "name": tc.get("name"),
                                        "input": tc.get("args", tc.get("input", {}))})
                    result.append({"role": "assistant", "content": content})
                else:
                    result.append({"role": "assistant", "content": m.content})
        return result

    def _payload(self, messages):
        p = {"provider": self.PROV_NAME, "model": self.MODEL, "messages": self._to_api_msgs(messages), "max_tokens": 4096}
        if self._tools:
            p["tools"] = self._tools
        return p

    def invoke(self, messages):
        data = requests.post(f"{self.BASE_URL}/v1/messages",
                             headers={"Authorization": f"Bearer {self.API_KEY}", "Content-Type": "application/json"},
                             json=self._payload(messages)).json()
        content_items = data.get("content", [])
        text = next((i["text"] for i in content_items if i.get("type") == "text"), "")
        tool_calls = [
            {"id": i.get("id"), "name": i.get("name"), "args": i.get("input", {})}
            for i in content_items if i.get("type") == "tool_use"
        ]
        return AIMessage(content=text, tool_calls=tool_calls)

    def stream(self, messages):
        resp = requests.post(f"{self.BASE_URL}/v1/messages/stream",
                             headers={"Authorization": f"Bearer {self.API_KEY}", "Content-Type": "application/json", "Accept": "text/event-stream"},
                             json=self._payload(messages), stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                line_text = line.decode("utf-8")
                if line_text.startswith("data: "):
                    data_str = line_text[6:]
                    if data_str == "[DONE]": break
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content" and data.get("content"):
                            class _C:
                                content = data["content"]
                            yield _C()
                        elif data.get("type") == "done": break
                    except json.JSONDecodeError: pass
