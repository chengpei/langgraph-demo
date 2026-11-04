import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from weather import get_weather


# 1. 加载环境变量
load_dotenv()


# 2. 初始化大模型
model = init_chat_model(
    os.getenv("MODEL_NAME", "deepseek-chat"),
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

# 3. 注册工具，并让模型知道有哪些工具可用
tools = [get_weather]
model_with_tools = model.bind_tools(tools)


# 4. 定义 LLM 节点
def llm_node(state: MessagesState):
    print("\n========== 进入 LLM 节点 ==========")

    print("当前 State 中的消息数量：", len(state["messages"]))

    for message in state["messages"]:
        print(
            type(message).__name__,
            "=>",
            message.content
        )

    response = model_with_tools.invoke(state["messages"])

    print("\n[LLM节点] 大模型完成一次推理")

    if response.tool_calls:
        print("[LLM节点] 模型决定调用工具：", response.tool_calls)

    return {"messages": [response]}


# 5. 定义条件路由函数
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]

    print("\n========== 条件路由 ==========")
    print("最后一条消息：", type(last_message).__name__)
    print("工具调用数量：", len(last_message.tool_calls))

    if last_message.tool_calls:
        return "tools"

    return END


# 6. 创建工具执行节点
tool_node = ToolNode(tools)


# 7. 构建 StateGraph
builder = StateGraph(MessagesState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

# 定义图的入口
builder.add_edge(START, "llm")

# LLM 执行后，根据条件决定下一步
builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# 工具执行完毕后，重新回到 LLM
builder.add_edge("tools", "llm")


# 8. 编译图
graph = builder.compile()


# 9. 命令行交互
while True:
    question = input("\n请输入问题（输入 exit 退出）：")

    if question.lower() == "exit":
        break

    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })

    print("\n========== 执行结果 ==========")

    for i, message in enumerate(result["messages"]):
        print(f"\n--- 第 {i + 1} 条消息 ---")
        print("类型：", type(message).__name__)
        print("内容：", message.content)

    print("\nAI：", result["messages"][-1].content)
