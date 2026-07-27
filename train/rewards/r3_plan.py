"""r3 预案质量与法规依据（0.3）：四要素 0.15 + RAG 引用一致 0.15。"""
import re

_PLAN_ELEMENTS = [
    re.compile(r"转移|撤离|疏散"),
    re.compile(r"物资|编织袋|冲锋舟|沙袋|抢险队"),
    re.compile(r"指挥部|责任人|牵头|防指"),
    re.compile(r"\d+\s*小时|时限|立即|小时\s*内"),
]


def _normalize(text: str) -> str:
    return re.sub(r"[《》\s]", "", text)


def r3_score(completion: str, rag_hits: list) -> float:
    final = completion.split("</tool_call>")[-1]
    score = 0.0
    if all(p.search(final) for p in _PLAN_ELEMENTS):
        score += 0.15
    # 引用条款命中 RAG 检索结果（标题或条文号交集非空）
    final_norm = _normalize(final)
    for hit in rag_hits:
        title = _normalize(str(hit.get("title", "")))
        article = _normalize(str(hit.get("article", "")))
        if (title and title in final_norm) or (article and article in final_norm):
            score += 0.15
            break
    return score
