"""三类自动记忆的正文结构；计数独立保存。"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.memory.safety import check_memory_safety


class MemoryBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SemanticBody(MemoryBody):
    title: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=15000)
    reuse_when: str = Field(default="", max_length=2000)
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=50)


class EpisodeBody(MemoryBody):
    event_summary: str = Field(min_length=1, max_length=2000)
    resolution: str = Field(default="", max_length=15000)
    outcome: Literal["success", "failure", "partial"] = "partial"
    happened_at: str
    source_refs: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    event_id: str = Field(min_length=1, max_length=200)


class ProcedureStep(MemoryBody):
    step: int = Field(ge=1, strict=True)
    action: str = Field(min_length=1, max_length=2000)
    tool: str | None = Field(default=None, max_length=128)


class ProcedureBody(MemoryBody):
    name: str = Field(min_length=1, max_length=128)
    applicability: str = Field(min_length=1, max_length=2000)
    steps: list[ProcedureStep] = Field(min_length=1, max_length=100)
    source_refs: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    status: Literal["active", "promoted", "deprecated"] = "active"

    @field_validator("steps")
    @classmethod
    def ordered_steps(cls, steps: list[ProcedureStep]) -> list[ProcedureStep]:
        if [s.step for s in steps] != list(range(1, len(steps) + 1)):
            raise ValueError("步骤必须从1开始连续编号")
        return steps


BODY_MODELS = {"semantic": SemanticBody, "episode": EpisodeBody, "procedure": ProcedureBody}


def validate_body(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    import json

    result = BODY_MODELS[kind].model_validate(body).model_dump()
    pending = [json.dumps(result, ensure_ascii=False), result]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and check_memory_safety(value):
            # 单独扫描原字符串，防止证据中的 JSON 被转义后漏检凭据。
            raise ValueError("记忆正文或来源未通过安全校验")
    return result
