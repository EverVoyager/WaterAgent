"""按明确选择的历史反思复盘；只保存审计和候选，不修改任何记忆。"""
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.memory.memory_store import get_memory_store
from agent.memory.reflection_writer import _semantic_candidate_violation
from agent.memory.safety import _is_sensitive_content, check_memory_safety
from agent.prompts.reflection import REVIEW_SYSTEM_PROMPT
from agent.utils import parse_json_from_llm
from app.core.llm import LLM_TIMEOUTS, get_llm_client, get_llm_config, strip_think


class MemoryReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reflection_ids: list[Annotated[int, Field(strict=True, gt=0)]] = Field(
        min_length=2, max_length=10,
    )
    focus: str = Field(default="重复错误、稳定约定及适用条件冲突", min_length=1, max_length=500)

    @field_validator("reflection_ids")
    @classmethod
    def unique_ids(cls, ids: list[int]) -> list[int]:
        if len(ids) != len(set(ids)):
            raise ValueError("反思 ID 不能重复")
        return ids

    @field_validator("focus")
    @classmethod
    def safe_focus(cls, value: str) -> str:
        if _is_sensitive_content(value):
            raise ValueError("复盘关注点不能包含凭据或个人敏感信息")
        return value


class ReviewCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=8, max_length=2000)


class ReviewCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["knowledge", "workflow", "conflict"]
    title: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=3000)
    knowledge_type: Literal["tool_contract", "project_rule", "domain_fact"] | None = None
    reuse_when: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=600)
    evidence: list[ReviewCitation] = Field(min_length=1, max_length=10)


class MemoryReviewResponse(BaseModel):
    review_id: int
    reflection_ids: list[int]
    reviewed_ids: list[int]
    skipped: list[dict[str, str | int]]
    summary: str
    candidates: list[ReviewCandidate]
    rejected_candidates: int


class ReflectionNotFoundError(LookupError):
    """请求选择了不存在的反思记录。"""


def _collect_cases(rows: list[dict]) -> tuple[list[dict], dict[str, dict], list[dict]]:
    cases, sources, skipped = [], {}, []
    for row in rows:
        context = row.get("context") or {}
        if context.get("kind") != "reflection" or context.get("version") != 1:
            skipped.append({"id": row["id"], "reason": "缺少原始来源上下文，未将旧回答当成证据"})
            continue
        case_sources = {}
        remaining = 6500
        raw_sources = context.get("semantic_sources") or {}
        if not isinstance(raw_sources, dict):
            raw_sources = {}
        for local_id, text in raw_sources.items():
            if (not isinstance(local_id, str) or not isinstance(text, str)
                    or (local_id != "user" and not re.fullmatch(r"tool:\d+:[\w-]+", local_id))
                    or (local_id == "user" and row.get("trigger_reason") not in (
                        "user_correction", "explicit_feedback"))
                    or local_id.rsplit(":", 1)[-1] in (
                        "read_memory_topic", "read_session_archive", "generate_plan")):
                continue
            excerpt = text[:min(2000, remaining)]
            if len(excerpt.strip()) < 8 or check_memory_safety(excerpt, excerpt.replace('"', "")):
                continue
            key = f"reflection:{row['id']}:{local_id}"
            case_sources[key] = excerpt
            sources[key] = {"text": excerpt, "local_id": local_id, "reflection_id": row["id"]}
            remaining -= len(excerpt)
            if remaining < 8:
                break
        event = context.get("event", "")
        if isinstance(event, str) and event.strip():
            event = event[:2500]
            if not check_memory_safety(event, event.replace('"', "")):
                key = f"reflection:{row['id']}:event"
                case_sources[key] = event
                sources[key] = {"text": event, "local_id": "event", "reflection_id": row["id"]}
        if case_sources:
            cases.append({"reflection_id": row["id"], "date": str(row.get("created_at", "")),
                          "sources": case_sources})
        else:
            skipped.append({"id": row["id"], "reason": "没有可安全引用的来源"})
    return cases, sources, skipped


def _candidate_supported(candidate: ReviewCandidate, sources: dict[str, dict]) -> bool:
    if candidate.kind != "knowledge" and candidate.knowledge_type is not None:
        return False
    if check_memory_safety(candidate.title, candidate.content, candidate.reuse_when, candidate.reason):
        return False
    records = set()
    for citation in candidate.evidence:
        source = sources.get(citation.source_id)
        if (not source or len(citation.quote.strip()) < 8
                or citation.quote not in source["text"]
                or check_memory_safety(citation.quote, citation.quote.replace('"', ""))):
            return False
        records.add(source["reflection_id"])
        if candidate.kind == "knowledge":
            if source["local_id"] == "event":
                return False
            memory = {"knowledge_type": candidate.knowledge_type,
                      "reuse_when": candidate.reuse_when,
                      "evidence": {"source_id": source["local_id"], "quote": citation.quote}}
            if _semantic_candidate_violation(memory, {source["local_id"]: source["text"]}):
                return False
    return candidate.kind == "knowledge" or len(records) >= 2


def review_reflections(request: MemoryReviewRequest) -> MemoryReviewResponse:
    """一次有界模型调用；旧上下文缺失时跳过，所有候选只进入审计。"""
    store = get_memory_store()
    rows = store.get_reflections_by_ids(request.reflection_ids)
    missing = set(request.reflection_ids) - {row["id"] for row in rows}
    if missing:
        raise ReflectionNotFoundError(f"反思记录不存在：{sorted(missing)}")
    cases, sources, skipped = _collect_cases(rows)
    candidates, rejected = [], 0
    summary = "至少需要两条有可用来源的历史反思；本次未调用模型。"
    if len(cases) >= 2:
        config = get_llm_config()
        client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["reflector"], max_retries=0)
        response = client.chat.completions.create(
            model=config["model"], temperature=0.1, max_tokens=4096,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"focus": request.focus, "cases": cases}, ensure_ascii=False)},
            ],
        )
        result = parse_json_from_llm(strip_think(response.choices[0].message.content or ""))
        if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
            raise RuntimeError("复盘模型未返回有效候选结构")
        summary = str(result.get("summary", "复盘完成，候选尚待核查。"))[:1000]
        if check_memory_safety(summary):
            summary = "复盘完成，候选尚待核查。"
        for item in result["candidates"][:10]:
            try:
                candidate = ReviewCandidate.model_validate(item)
            except ValidationError:
                rejected += 1
                continue
            if _candidate_supported(candidate, sources):
                candidates.append(candidate)
            else:
                rejected += 1
        rejected += max(0, len(result["candidates"]) - 10)
    report = {
        "kind": "review", "version": 1, "reflection_ids": request.reflection_ids,
        "reviewed_ids": [case["reflection_id"] for case in cases], "skipped": skipped,
        "focus": request.focus, "summary": summary,
        "candidates": [candidate.model_dump() for candidate in candidates],
        "rejected_candidates": rejected,
    }
    review_id = store.add_reflection(
        user_query=request.focus, trigger_reason="memory_review",
        reflection_text=summary, memories_created=0, context=report,
    )
    if not isinstance(review_id, int):
        raise RuntimeError("复盘已完成，但审计保存失败；未应用任何候选")
    return MemoryReviewResponse(review_id=review_id, **report)
