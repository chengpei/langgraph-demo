import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain.messages import ToolMessage

from agent import graph, HumanMessage


app = FastAPI(title="LangGraph Agent Web")

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # 每个浏览器会话对应一个 thread_id，checkpointer 自动维护该会话的历史
    thread_id = req.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=req.message)]}

    async def event_stream():
        try:
            # stream_mode="messages" 逐条产出 (消息块, 元数据)，
            # 可同时拿到 LLM 的 token 流和工具节点的执行结果
            async for message, metadata in graph.astream(
                inputs, config=config, stream_mode="messages"
            ):
                node = metadata.get("langgraph_node")

                if isinstance(message, ToolMessage):
                    yield _sse({
                        "type": "tool",
                        "name": message.name,
                        "content": message.content,
                    })
                elif node == "llm" and isinstance(message.content, str) and message.content:
                    yield _sse({"type": "token", "content": message.content})

            yield _sse({"type": "done", "thread_id": thread_id})
        except Exception as e:
            yield _sse({"type": "error", "content": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
