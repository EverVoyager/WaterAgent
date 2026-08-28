"""LangGraph 节点使用的提示词模板（按模块拆分）。

模块结构：
- direct_chat.py    闲聊/通用问答
- synthesizer.py    综合研判（含 JSON Schema 和引用规范）
- reflection.py     记忆反思与压缩
- compact.py        上下文压缩
- generate_plan.py  应急预案生成

本 __init__.py 仅做重新导出，保持向后兼容：
    from agent.prompts import DIRECT_CHAT_PROMPT, SYNTHESIZER_PROMPT
"""
from agent.prompts.direct_chat import DIRECT_CHAT_PROMPT
from agent.prompts.synthesizer import (
    CITATION_GUIDANCE,
    SYNTH_ANSWER_PROMPT,
    SYNTH_META_SCHEMA,
    SYNTH_RESPONSE_SCHEMA,
    SYNTHESIZER_PROMPT,
)

__all__ = [
    "DIRECT_CHAT_PROMPT",
    "SYNTHESIZER_PROMPT",
    "SYNTH_ANSWER_PROMPT",
    "CITATION_GUIDANCE",
    "SYNTH_RESPONSE_SCHEMA",
    "SYNTH_META_SCHEMA",
]
