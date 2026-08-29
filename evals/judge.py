"""LLM-as-Judge：Rubric 式软指标评判（--no-judge 可整体跳过）。

书中方法论落地（Scale AI / LLM-as-Judge 一节）：
- Rubric 四类准则：必要项 / 重要项 / 可选项 / 陷阱项，权重加权合成总分；
  **幻觉是否决项（veto）而非评分维度**——一个流畅但编造数据的回答，
  伤害远大于简短但准确的回答，触发否决直接 0 分。
- 评分档写具体可验证的行为标准，不写"展示了深刻理解"式抽象描述。
- 评判模型与主模型异源（LLM_JUDGE_MODEL 独立配置，默认 qwen-max），
  规避同源模型的自我偏好。
- 输出强制 JSON，解析失败容错重试一次，仍失败则该条记为不可用（不阻塞批量）。

注：faithfulness 的"声明分解→逐条对照工具结果"思路对标 RAGAS；
本实现让单次评判同时完成分解与核对，控制评估成本（v1 权衡）。
"""
import json
import logging

from agent.utils import parse_json_from_llm

logger = logging.getLogger(__name__)

# Rubric 权重（必要/重要/可选；陷阱项为扣分；否决项直接 0 分）
_RUBRIC_WEIGHTS = {"necessary": 0.5, "important": 0.3, "optional": 0.2}

JUDGE_SYSTEM_PROMPT = """你是防汛预警系统的回答质量评审员。根据给定的用户问题、Agent 回答和工具返回数据，按 Rubric 评分。

## 评分准则（每档判定标准都是具体行为，不是抽象印象）

**必要项 necessary**（权重 0.5）——全部满足才得满分，每缺一项按比例扣：
- 回答直接回应了用户的问题，没有答非所问
- 涉及预警等级/流量/水位/降雨时，数值与工具返回数据一致，无编造
- 给出的行动建议在等级对应的处置范围内，无危险误导（如Ⅳ级小水情要求全员撤离）

**重要项 important**（权重 0.3）：
- 结构清晰：结论先行，数据支撑在后
- 覆盖了主要风险点（未遗漏工具数据中明显异常的值）
- 语言简洁，无与问题无关的 filler

**可选项 optional**（权重 0.2）：
- 恰当使用编号/分点排版
- 提示了数据局限性或建议持续关注

**陷阱项 trap**（每处 -0.2，上限 -0.4）：
- 对用户明显的错误说法一味附和（如用户声称的数据与工具结果不符却顺着说）
- 冗余客套、重复罗列工具返回的原始 JSON

**否决项 veto**（触发即总分 0）：
- 编造工具数据中不存在的具体数值（流量/水位/降雨/站点名/法规条款号）
- 引用了不存在的数据来源

## 输出格式（严格 JSON，不要多余文本）

{
  "faithfulness": {
    "total_claims": <回答中的事实性声明总数 int>,
    "supported_claims": <与工具数据一致 int>,
    "unsupported_claims": [<未被工具数据支持的具体声明字符串>]
  },
  "answer_quality": {
    "necessary_score": <0.0-1.0>,
    "important_score": <0.0-1.0>,
    "optional_score": <0.0-1.0>,
    "trap_deductions": <0.0-0.4>,
    "veto_triggered": <true|false>,
    "veto_reason": "<触发否决的具体原因，未触发留空>",
    "rationale": "<一句话总评>"
  }
}"""

_JUDGE_USER_TMPL = """## 用户问题
{query}

## Agent 回答
{answer}

## 工具返回数据（JSON）
{tool_data}

请按 Rubric 输出 JSON 评分。"""


def _fmt_tool_results(tool_calls: list[dict]) -> str:
    """工具结果的紧凑文本（剔除时间戳/series 明细，控制评判上下文长度）。"""
    compact = []
    for tc in tool_calls or []:
        entry = {"tool": tc.get("tool_name", ""), "error": tc.get("error", "")}
        res = tc.get("result") or {}
        if isinstance(res, dict) and not tc.get("error"):
            entry["result"] = {
                k: v for k, v in res.items()
                if k not in ("series", "fetched_at", "searched_at", "generated_at",
                             "predicted_at", "analyzed_at", "_from_cache", "source")
            }
        compact.append(entry)
    try:
        return json.dumps(compact, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(compact)


def _score_from_quality(quality: dict) -> float | None:
    """Rubric 加权总分：必要/重要/可选加权 - 陷阱扣分；否决触发直接 0。"""
    try:
        if quality.get("veto_triggered") is True:
            return 0.0
        score = (
            _RUBRIC_WEIGHTS["necessary"] * float(quality.get("necessary_score", 0.0))
            + _RUBRIC_WEIGHTS["important"] * float(quality.get("important_score", 0.0))
            + _RUBRIC_WEIGHTS["optional"] * float(quality.get("optional_score", 0.0))
            - float(quality.get("trap_deductions", 0.0))
        )
        return round(max(0.0, min(1.0, score)), 4)
    except (TypeError, ValueError):
        return None


def judge_record(record: dict, client=None, model: str = "", max_retries: int = 1) -> dict | None:
    """对单条评估记录做 Rubric 评判。

    Returns:
        {
          "faithfulness": {"rate": 支持率, "total": n, "supported": n, "unsupported": [...]},
          "answer_quality": {"score": 加权总分, "veto_triggered": bool, "rationale": str},
        }
        评判不可用（LLM 失败/解析失败）返回 None。
    """
    answer = record.get("final_answer", "")
    if not answer:
        return None
    tool_calls = record.get("_tool_calls") or []
    user_prompt = _JUDGE_USER_TMPL.format(
        query=record.get("query", ""),
        answer=answer[:4000],
        tool_data=_fmt_tool_results(tool_calls)[:4000],
    )
    if client is None:
        return None
    last_err = ""
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            parsed = parse_json_from_llm(resp.choices[0].message.content or "")
            if parsed is None:
                last_err = "judge 输出非 JSON"
                continue
            faith = parsed.get("faithfulness") or {}
            quality = parsed.get("answer_quality") or {}
            total = int(faith.get("total_claims", 0) or 0)
            supported = int(faith.get("supported_claims", 0) or 0)
            score = _score_from_quality(quality)
            if score is None:
                last_err = "judge 分数字段缺失"
                continue
            return {
                "faithfulness": {
                    "rate": round(supported / total, 4) if total > 0 else None,
                    "total": total,
                    "supported": supported,
                    "unsupported": faith.get("unsupported_claims", []) or [],
                },
                "answer_quality": {
                    "score": score,
                    "veto_triggered": quality.get("veto_triggered") is True,
                    "veto_reason": quality.get("veto_reason", ""),
                    "rationale": quality.get("rationale", ""),
                },
            }
        except Exception as e:  # noqa: BLE001 —— 评判失败不阻塞批量评估
            last_err = f"{type(e).__name__}: {e}"
    logger.warning("[eval-judge] case %s 评判失败: %s", record.get("case_id"), last_err)
    return None


def aggregate_judge(judgments: list[dict | None]) -> dict:
    """聚合评判结果（None=不可用，单独计数）。"""
    valid = [j for j in judgments if j is not None]
    n_unavailable = len(judgments) - len(valid)
    if not valid:
        return {"n_judged": 0, "n_unavailable": n_unavailable}
    rates = [j["faithfulness"]["rate"] for j in valid if j["faithfulness"]["rate"] is not None]
    scores = [j["answer_quality"]["score"] for j in valid]
    vetoes = sum(1 for j in valid if j["answer_quality"]["veto_triggered"])
    return {
        "n_judged": len(valid),
        "n_unavailable": n_unavailable,
        "faithfulness_rate": round(sum(rates) / len(rates), 4) if rates else None,
        "quality_score_mean": round(sum(scores) / len(scores), 4),
        "veto_count": vetoes,
        "veto_rate": round(vetoes / len(valid), 4),
    }
