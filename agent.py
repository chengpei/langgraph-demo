import os
from typing import Literal

import deepseek_compat  # noqa: F401  在构建模型前启用 DeepSeek 思考模式兼容补丁
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from weather import get_weather
from db import query_database, list_tables, get_table_schema
from sms import send_sms


# 加载环境变量
load_dotenv()


def _make_model():
    return init_chat_model(
        os.getenv("MODEL_NAME", "deepseek-chat"),
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )


base_model = _make_model()

# =====================================================================
# 一、把工具按职责拆成三组，每个子 Agent 只绑定自己那一组
# =====================================================================
data_tools = [list_tables, get_table_schema, query_database]   # 数据查询
# 短信助手额外附带只读查询工具：用户常说"给某老师发消息"，它需要数据源去查老师的
# 手机号再发。这是职责拆分的务实取舍——否则它只能向用户要手机号。
sms_tools = [send_sms, *data_tools]                              # 发短信（+ 查手机号）
weather_tools = [get_weather]                                   # 天气

data_model = base_model.bind_tools(data_tools)
sms_model = base_model.bind_tools(sms_tools)
weather_model = base_model.bind_tools(weather_tools)

# =====================================================================
# 二、supervisor：把这些子 Agent 当作"可用工具"暴露给它
#     模型通过 tool_call 点选要把任务派给谁；路由节点再把
#     tool_call 翻译成"进入哪个 Agent 节点"。这就是多 Agent 的分派机制。
# =====================================================================
AGENT_OPTIONS = [
    Tool(
        name="data_agent",
        description="数据查询助手：回答需要查数据库的问题（学生/老师/成绩库的总分、平均分、"
                    "最高分、排名、老师手机号等）。它只负责「查询」并给出数字答案，"
                    "绝不发送短信——任何发短信/通知请求必须派给 sms_agent。",
        func=lambda: "data_agent",
    ),
    Tool(
        name="sms_agent",
        description="短信助手：负责「给某人发送短信/通知/提醒」这类请求，例如通知老师开会、"
                    "交作业、提醒事项。如果知道老师姓名但不知道手机号，sms_agent 会自己"
                    "先查库拿到手机号再发送。只要用户表达了发送/通知/提醒的意思就归它。",
        func=lambda: "sms_agent",
    ),
    Tool(
        name="weather_agent",
        description="天气助手：负责查询某个城市的天气情况。",
        func=lambda: "weather_agent",
    ),
]
supervisor_model = base_model.bind_tools(AGENT_OPTIONS)

SUPERVISOR_PROMPT = (
    "你是任务调度 supervisor。阅读用户的请求，判断要做的任务，然后【一次只派一个助手】"
    "（调用一个路由工具）；该助手处理完会回到你，你再根据它的结果决定是否派下一个，"
    "直到所有子任务都完成，才输出面向用户的最终总结。\n"
    "\n"
    "意图与派单规则：\n"
    "1. 查/统计数据库数据（总分、最高分、名次、老师手机号等）→ data_agent。\n"
    "2. 「给某人发短信/发消息/通知/提醒/祝贺」等任何要『把话发给对方』的诉求 → sms_agent，"
    "一律不交给 data_agent，即使需要先查手机号。（不管问句里有没有“查”字。）\n"
    "3. 查天气 → weather_agent。\n"
    "\n"
    "要点：如果用户既想查数据、又要给某人发短信（例如“查王五最高分，再给王丹老师发短信祝贺”），"
    "必须分两步依次派：先派 data_agent 查数据，再派 sms_agent 发短信。"
    "绝不允许在用户想『发短信』时，只派 data_agent 把话写出来就收场——data_agent 没有"
    "发送能力，那样等于没完成用户的请求。"
)


def _ensure_system(messages, prompt: str):
    """往消息头注入一条系统提示（每个 agent 只注入一次，防止循环中重复）。"""
    if any(isinstance(m, SystemMessage) and prompt in m.content for m in messages):
        return messages
    return [SystemMessage(content=prompt), *messages]


# supervisor 的分派工具名：它们只是"路由指令"而非真实工具，有没有对应 ToolMessage 回复无关紧要
_ROUTING_TOOLS = {"data_agent", "sms_agent", "weather_agent"}


def _strip_routing(messages):
    """把纯分派性质的 AI 消息从历史里去掉。

    若不剥掉，supervisor 的 tool_call(认 data_agent 等) 会以
    "有 tool_calls 却无对应 ToolMessage" 的非法形态混进后续模型调用，
    被 OpenAI 拒收。真正的工具链（query_database/send_sms 等）不受影响。
    """
    out = []
    for m in messages:
        if getattr(m, "tool_calls", None):
            names = [tc["name"] for tc in m.tool_calls]
            if names and all(n in _ROUTING_TOOLS for n in names):
                continue
        out.append(m)
    return out


# =====================================================================
# 三、节点定义
# =====================================================================
def supervisor_node(state: MessagesState):
    messages = _ensure_system(_strip_routing(state["messages"]), SUPERVISOR_PROMPT)
    response = supervisor_model.invoke(messages)
    return {"messages": [response]}


def data_agent(state: MessagesState):
    messages = _ensure_system(
        _strip_routing(state["messages"]),
        "你是数据查询助手，只查成绩数据库。先用 list_tables / get_table_schema 看表结构，"
        "再生成 SQL 用 query_database 查询，最后把结果转成自然语言。",
    )
    response = data_model.invoke(messages)
    return {"messages": [response]}


def sms_agent(state: MessagesState):
    messages = _ensure_system(
        _strip_routing(state["messages"]),
        "你是短信助手，负责用 send_sms 给指定手机号发送短信。如需给某位老师发，"
        "可先用数据查询拿到该老师的 phone，再调用 send_sms。",
    )
    response = sms_model.invoke(messages)
    return {"messages": [response]}


def weather_agent(state: MessagesState):
    messages = _ensure_system(
        _strip_routing(state["messages"]),
        "你是天气助手，用 get_weather 查询城市天气并回答。",
    )
    response = weather_model.invoke(messages)
    return {"messages": [response]}


# =====================================================================
# 四、路由与循环
# =====================================================================
def route_supervisor(state: MessagesState) -> str:
    """supervisor 的 tool_call 名 -> 进入哪个子 Agent 节点。"""
    last = state["messages"][-1]
    if not last.tool_calls:
        return END
    return last.tool_calls[0]["name"]   # data_agent / sms_agent / weather_agent


def _make_continue(tool_node: str):
    """每个子 Agent 自己的小 ReAct：要工具→去自己的工具节点；否则回 supervisor。"""
    def continue_fn(state: MessagesState) -> Literal[tool_node, "supervisor"]:
        if state["messages"][-1].tool_calls:
            return tool_node
        return "supervisor"
    return continue_fn


# =====================================================================
# 五、搭图
# =====================================================================
builder = StateGraph(MessagesState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("data_agent", data_agent)
builder.add_node("data_tools", ToolNode(data_tools))
builder.add_node("sms_agent", sms_agent)
builder.add_node("sms_tools", ToolNode(sms_tools))
builder.add_node("weather_agent", weather_agent)
builder.add_node("weather_tools", ToolNode(weather_tools))

# 入口 -> supervisor
builder.add_edge(START, "supervisor")

# supervisor 分派 -> 各子 Agent / 结束
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "data_agent": "data_agent",
        "sms_agent": "sms_agent",
        "weather_agent": "weather_agent",
        END: END,
    },
)

# 每个子 Agent：自己那组工具循环，完成后回 supervisor
builder.add_conditional_edges("data_agent", _make_continue("data_tools"),
                              {"data_tools": "data_tools", "supervisor": "supervisor"})
builder.add_edge("data_tools", "data_agent")

builder.add_conditional_edges("sms_agent", _make_continue("sms_tools"),
                              {"sms_tools": "sms_tools", "supervisor": "supervisor"})
builder.add_edge("sms_tools", "sms_agent")

builder.add_conditional_edges("weather_agent", _make_continue("weather_tools"),
                              {"weather_tools": "weather_tools", "supervisor": "supervisor"})
builder.add_edge("weather_tools", "weather_agent")


graph = builder.compile(checkpointer=InMemorySaver())

__all__ = ["graph", "HumanMessage"]