# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

这是一个基于 LangGraph 的 AI Agent 开发脚手架，后续所有 Agent 都在此模式上扩展。当前示例是一个最小 ReAct 循环：大模型通过工具调用获取信息，工具结果返回大模型，直到模型不再请求工具。

## 常用命令

```bash
# 激活虚拟环境（Python 3.11）
source .venv/bin/activate

# 运行 Agent（交互式命令行，输入 exit 退出）
python main.py

# 运行 Web 版对话界面
python -m uvicorn app:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

依赖固化在 `requirements.txt`，重建环境用 `python -m venv .venv && pip install -r requirements.txt`。注意：venv 内写死绝对路径，**项目目录改名后必须重建虚拟环境**（activate 脚本和 bin 下命令的 shebang 都会失效）。

项目没有配置测试、lint 和格式化工具。

## 环境变量

`.env` 必须提供（`main.py` 启动时通过 `load_dotenv()` 加载）：

- `OPENAI_API_KEY` — API 密钥
- `OPENAI_BASE_URL` — OpenAI 兼容接口地址
- `MODEL_NAME` — 模型名，默认 `deepseek-chat`（即通过 OpenAI 协议接入 DeepSeek 等第三方模型）

## 架构

图的定义和编译在 `agent.py`（`build_graph()` + 模块级共享实例 `graph`，带 `InMemorySaver` checkpointer），两个入口共用：`main.py` 是控制台 REPL，`app.py` 是 FastAPI Web 服务（SSE 流式输出，页面在 `static/index.html`，按浏览器会话的 `thread_id` 保持多轮上下文）。图结构：

```
START → llm ──should_continue──→ tools → llm（循环）
                   │
                   └──────────→ END（无 tool_calls 时）
```

- **State**：使用 LangGraph 预置的 `MessagesState`，唯一状态是 `messages` 列表；每个节点返回 `{"messages": [...]}` 追加消息。
- **llm 节点**：把完整消息历史交给 `model.bind_tools(tools)` 绑定后的模型推理；模型返回 `tool_calls` 时由条件路由转发。
- **条件路由 `should_continue`**：检查最后一条消息的 `tool_calls`，有则去 `tools` 节点，否则返回 `END`。新增分支（如人工审核、子图跳转）在此函数和 `add_conditional_edges` 映射中扩展。
- **tools 节点**：`langgraph.prebuilt.ToolNode`，按 `tool_calls` 自动执行对应函数并把 `ToolMessage` 写回 state，执行完固定回到 `llm`。
- **工具定义**（`weather.py` 示例）：用 `@tool` 装饰普通函数，docstring 和参数类型注解会作为工具描述发给模型。
- **数据库工具簇**（`db.py`）：**三件套**已注册进 `tools`——`list_tables`（列出表名+注释）、`get_table_schema(table)`（某表字段+外键）、`query_database(sql)`（执行只读 SELECT）。模型**渐进式发现 schema**：先 list → 再 get_table_schema 看相关表 → 再写 SQL，避免把成百张表的结构写进提示词。写错的 SQL 会连同报错返回给模型自行修正。三工具均强制只读（非 SELECT 拦截）、走 `SET SESSION TRANSACTION READ ONLY`。连接配置从 `.env` 读 `DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME`（默认 MySQL `langgraph_demo`）。**依赖表和字段的 COMMENT 来让模型理解语义，建表时务必填好 COMMENT。**
- **建表与示例数据**（`db_init.py`）：`python db_init.py` 会 DROP 重建三张表（students 学生表、teachers 老师表、scores 成绩表）并写入示例数据。**可重复运行但会清空现有数据**，接到真实数据后不要再执行它。

### 添加新工具 / 新 Agent 能力的模式

1. 新建模块（或在现有模块中）用 `@tool` 定义函数，写清 docstring 和类型注解；
2. 在 `agent.py` 导入并加入 `tools = [...]` 列表 —— `bind_tools` 和 `ToolNode` 共用这一个列表，无需其他注册；
3. 需要持久化、多 Agent 协作或更复杂状态时，再从 `MessagesState` 迁移到自定义 `TypedDict` state。

## 注意事项

- 使用 **LangChain 1.x 统一包导入路径**：`langchain.chat_models.init_chat_model`、`langchain.messages.HumanMessage`、`langchain.tools.tool`，不是旧版教程里的 `langchain_openai` / `langchain_core.messages` 路径（关键版本：langchain 1.4.x、langgraph 1.2.x）。
- 代码注释、CLI 提示和用户交互文案均为中文，新增内容保持中文注释风格。
- `main.py` 顶层仍有 `while True` 交互循环，不要 import 它；要复用图请从 `agent` 导入 `graph`（新入口参照 `app.py` 的写法）。
