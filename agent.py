import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from weather import get_weather
from db import query_database, list_tables, get_table_schema
from sms import send_sms


# 加载环境变量
load_dotenv()

# 初始化大模型（通过 OpenAI 兼容协议接入 DeepSeek 等第三方模型）
model = init_chat_model(
    os.getenv("MODEL_NAME", "deepseek-chat"),
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

# 工具注册表：bind_tools 和 ToolNode 共用这一个列表
tools = [get_weather, list_tables, get_table_schema, query_database, send_sms]
model_with_tools = model.bind_tools(tools)


# LLM 节点：把完整消息历史交给模型推理
def llm_node(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 条件路由：模型请求工具则去 tools 节点，否则结束
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


def build_graph():
    """构建并编译 ReAct Agent 图。

    使用 InMemorySaver 按 thread_id 保存对话历史，
    同一 thread_id 内多轮对话自动携带上下文（进程重启后丢失）。
    """
    builder = StateGraph(MessagesState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    # 工具执行完毕后重新回到 LLM
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=InMemorySaver())


# 全局共享的编译后的图（checkpointer 必须是同一个实例才能共享历史）
graph = build_graph()

__all__ = ["graph", "HumanMessage"]
