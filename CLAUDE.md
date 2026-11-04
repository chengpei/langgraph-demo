# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

这是一个基于 LangGraph 的 AI Agent 开发脚手架，后续所有 Agent 都在此模式上扩展。当前示例是一个**多 Agent 编排**结构：一个 **supervisor 调度者**把任务分派给三个职责单一的子 Agent——**数据查询**（Text-to-SQL）、**短信**、**天气**；每个子 Agent 内部仍是「大模型 + 自己那组工具」的 ReAct 循环。

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

图的定义和编译在 `agent.py`（模块级共享实例 `graph`，带 `InMemorySaver` checkpointer），两个入口共用：`main.py` 是控制台 REPL，`app.py` 是 FastAPI Web 服务（SSE 流式，页面在 `static/index.html`，按浏览器会话的 `thread_id` 保持多轮上下文）。图结构（supervisor 分派 + 每个子 Agent 的独立 ReAct 循环）：

```
START → supervisor ──route──→ data_agent ──→ data_tools → data_agent（循环）
                  │          → sms_agent ──→ sms_tools  → sms_agent（循环）
                  │          → weather_agent → weather_tools → weather_agent（循环）
                  └──────────→ END（无分派时）
每个子 Agent 处理完（不再调工具）自动回到 supervisor 做最终总结
```

- **State**：使用 LangGraph 预置的 `MessagesState`，唯一状态是 `messages` 列表；各节点返回 `{"messages": [...]}` 追加消息。所有节点共享同一份消息历史，sub-agent 能看到 supervisor 及之前的对话。
- **supervisor 节点**：把三个子 Agent 当作可用工具（`AGENT_OPTIONS` 里 `Tool(name=..., description=..., func=lambda: 名字)`）暴露给模型。模型通过 `tool_call` 点选要派给谁；`route_supervisor` 再把 `tool_call` 的名字翻译成「进入哪个子 Agent 节点」。没有 `tool_calls` 即分派结束 → END。
- **子 Agent 节点**：各自绑定自己那组工具（`bind_tools` + 自己一个 `ToolNode`），内部是独立的 ReAct 小循环：要工具→去自己的 tools 节点→再回来，直到不调工具→回 supervisor。
- **`_strip_routing`**：supervisor 纯分派用的 AI 消息（tool_call 全部是 `data_agent/sms_agent/weather_agent` 这些路由名）属于「路由指令」而非真实工具回复，必须从历史里剥掉，否则后续模型调用会因「有 tool_calls 却缺对应 ToolMessage」被 OpenAI 拒收。真正的工具链（`query_database/send_sms` 等）不受影响。
- **`_ensure_system`**：往消息头注入子 Agent 的系统提示词，保证只在第一条时注入一次，循环中不重复。
- **工具定义**（`db.py`/`sms.py`/`weather.py`）：用 `@tool` 装饰普通函数，docstring 和参数类型注解会作为工具描述发给模型。
- **数据库工具簇**（`db.py`）：**三件套** `list_tables` / `get_table_schema` / `query_database`。模型**渐进式发现 schema**：先 list → 再 get_table_schema 看相关表 → 再写 SQL，避免把成百张表的结构写进提示词。三工具均强制只读（非 SELECT 拦截）。**依赖表和字段的 COMMENT 来让模型理解语义，建表时务必填好 COMMENT。**
- **DeepSeek 思考模式兼容**（`deepseek_compat.py`）：DeepSeek 回复会带 `reasoning_content` 且要求多轮时原样回传，而 langchain-openai 默认丢弃两端（解析时、序列化时）。`agent.py` 顶部 `import deepseek_compat` 会给底层解析/序列化函数打薄包装，把 `reasoning_content` 存进 `additional_kwargs` 并出站重写（**空字符串也要保留**，思考模式要求历史每条 assistant 消息都带该字段，否则 400）。换非思考模型时为零副作用空操作。
- **建表与示例数据**（`db_init.py`）：DROP 重建三张表（students/teachers/scores）并写入示例数据。**可重复运行但会清空现有数据**，接到真实数据后不要再执行。

### 添加新工具 / 新 Agent 能力的模式

1. 新建模块（或在现有模块中）用 `@tool` 定义函数，写清 docstring 和类型注解；
2. 在 `agent.py` 归入对应子 Agent 的工具组（`data_tools`/`sms_tools`/`weather_tools`），`bind_tools` 和该子 Agent 的 `ToolNode` 共用这一个列表，无需其他注册；
3. 新增**子 Agent**：先加一个工具组和 `AGENT_OPTIONS` 里的分派`Tool`（名字、描述、`func=lambda: 名字`），再加 `_ROUTING_TOOLS`，然后照抄一个 agent 节点 + `_make_continue` + `add_node/add_conditional_edges/add_edge`。
4. 需要持久化或更复杂状态时，再从 `MessagesState` 迁移到自定义 `TypedDict` state。

## 注意事项

- 使用 **LangChain 1.x 统一包导入路径**：`langchain.chat_models.init_chat_model`、`langchain.messages.HumanMessage`、`langchain.tools.tool`，不是旧版教程里的 `langchain_openai` / `langchain_core.messages` 路径（关键版本：langchain 1.4.x、langgraph 1.2.x）。
- 代码注释、CLI 提示和用户交互文案均为中文，新增内容保持中文注释风格。
- `main.py` 顶层仍有 `while True` 交互循环，不要 import 它；要复用图请从 `agent` 导入 `graph`（新入口参照 `app.py` 的写法）。
- `app.py` 的 SSE 输出遵循「**主回复唯一**」约定：只有 supervisor **无 tool_call 的最终总结**才作为主回复 token 流，子 agent（data/sms/weather）的中间输出和 supervisor 带路由 tool_call 的中间话术一律以 `agent_log` 当日志下发，前端弱化展示（灰底限高可滚动）。因为 tool_call 通常出现在同一条消息末尾，supervisor 文本要按消息缓冲、等该消息结束再判定是主回复还是日志。**注意 `event_stream` 是 async 函数，内部不能用 `yield from`**，转发缓冲统一用 `for evt in flush(p): yield evt`。
