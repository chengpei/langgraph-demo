import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain.messages import ToolMessage
from langchain_core.messages import AIMessageChunk

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

    # 工具执行结果是次要信息，可作为进度日志（前端以弱化卡片展示）
    async def event_stream():
        try:
            # stream_mode="messages" 逐条产出 (消息块, 元数据)。
            # 主回复想表达的原则：
            #   - 子 agent（data/sms/weather）的中间输出只是日志，绝不进主回复；
            #   - supervisor 也要区分「中间派发话术」与「最终总结」——只有无 tool_call
            #     的最终总结才算主回复，带路由 tool_call 的中间话术归入日志。
            # 因为 tool_call 通常出现在同一条消息的末尾，supervisor 的文本要先按
            # 消息缓冲，等该消息结束（节点切换 / 工具事件）再决定下发方式。
            pending = None  # 缓冲中的 supervisor 消息：{id, handoff, text}

            def flush(p):
                """落定一条缓冲的 supervisor 消息，返回本应下发的 SSE 事件列表。
                handoff=带路由 tool_call 的中间话术 → 日志；否则 → 主回复 token。
                空文本直接丢弃，避免产生空的日志卡。"""
                if not p["text"].strip():
                    return []
                if p["handoff"]:
                    return [_sse({"type": "agent_log", "agent": "supervisor",
                                  "id": p["id"], "content": p["text"]})]
                return [_sse({"type": "token", "content": p["text"]})]

            # async 事件循环内不能用 `yield from`，落到 supervisor 的缓冲文本统一走 flush()，
            # 调用点用 `for evt in flush(p): yield evt` 转发。

            async for message, metadata in graph.astream(
                inputs, config=config, stream_mode="messages"
            ):
                node = metadata.get("langgraph_node")

                if isinstance(message, ToolMessage):
                    if pending:
                        for evt in flush(pending):
                            yield evt
                        pending = None
                    yield _sse({
                        "type": "tool",
                        "name": message.name,
                        "content": message.content,
                    })
                    continue

                if not (
                    isinstance(message, AIMessageChunk)
                    and isinstance(message.content, str)
                    and message.content
                ):
                    continue

                if node == "supervisor":
                    if pending is None:
                        pending = {"id": message.id, "handoff": False, "text": ""}
                    elif pending["id"] != message.id:
                        # 新一条 supervisor 消息：先落定上一条
                        for evt in flush(pending):
                            yield evt
                        pending = {"id": message.id, "handoff": False, "text": ""}
                    if message.tool_call_chunks:
                        pending["handoff"] = True
                    pending["text"] += message.content
                else:
                    # 子 agent 的中间回复 → 日志
                    if pending:
                        for evt in flush(pending):
                            yield evt
                        pending = None
                    yield _sse({"type": "agent_log", "agent": node,
                                "id": message.id, "content": message.content})

            if pending:
                for evt in flush(pending):
                    yield evt
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
