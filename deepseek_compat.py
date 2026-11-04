"""DeepSeek 思考模式（thinking mode）与 langchain-openai 的兼容补丁。

背景：DeepSeek 的回复里会带一个顶层字段 `reasoning_content`（模型思考过程）。
DeepSeek 要求多轮对话时，把历史中 assistant 消息原本的 `reasoning_content`
原样回传给 API，否则返回 400：
    The `reasoning_content` in the thinking mode must be passed back to the API.

但 langchain-openai 默认不保留这个字段：
- 解析响应时丢（_convert_dict_to_message / _convert_delta_to_message_chunk）
- 序列化出站消息时也不带（_convert_message_to_dict）

本模块在导入时给这几个函数打上薄包装，把 reasoning_content 存进
additional_kwargs 并在出站时重新写入消息体。只需在 agent.py 顶部 import 一次。

注意：non-thinking 模型（如 deepseek-chat）不产生该字段，本补丁是零副作用空操作。
"""

import langchain_core.messages as _mc
import langchain_openai.chat_models.base as _b

# ---- 1. 回程：非流式解析，把 reasoning_content 存进 additional_kwargs ----
_orig_dict_to_message = _b._convert_dict_to_message


def _dict_to_message(_dict):
    message = _orig_dict_to_message(_dict)
    if isinstance(message, _mc.AIMessage):
        # 只要响应里带了这个字段就存（含空字符串）：DeepSeek 思考模式要求
        # 历史中每条 assistant 消息都回传 reasoning_content，空也得出站。
        if "reasoning_content" in _dict and "reasoning_content" not in message.additional_kwargs:
            message.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
    return message


_b._convert_dict_to_message = _dict_to_message

# ---- 2. 回程：流式解析，逐块累加 reasoning_content ----
_orig_delta_to_chunk = _b._convert_delta_to_message_chunk


def _delta_to_message_chunk(_dict, default_class):
    chunk = _orig_delta_to_chunk(_dict, default_class)
    if isinstance(chunk, _mc.AIMessageChunk) and "reasoning_content" in _dict:
        # 分块像 content 一样自动拼接：Chunk 合并时 additional_kwargs 里的
        # 字符串会被连续追加，最终拿到完整的 reasoning_content。空值也保留，
        # 保证该 key 在历史中始终存在。
        chunk.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
    return chunk


_b._convert_delta_to_message_chunk = _delta_to_message_chunk

# ---- 3. 去程：序列化出站 assistant 消息时，把 reasoning_content 写回 ----
_orig_message_to_dict = _b._convert_message_to_dict


def _message_to_dict(message, api="chat/completions"):
    d = _orig_message_to_dict(message, api=api)
    if isinstance(message, _mc.AIMessage) and "reasoning_content" in message.additional_kwargs:
        d["reasoning_content"] = message.additional_kwargs["reasoning_content"]
    return d


_b._convert_message_to_dict = _message_to_dict

# 对外可见，便于测试断言补丁已生效
__all__ = ["_message_to_dict"]