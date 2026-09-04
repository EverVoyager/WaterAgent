"""会话任务段落盘与按需还原（Context-Folding 工程化实现）。

设计依据 docs/context-compression-research.md：
- 粒度 = 任务段（连续相关轮次聚为一段），段边界 = 相邻轮 query 的
  embedding 余弦低于阈值（语义边界；前端 history 无时间戳，不做时间边界）
- 归档：每段一个 MD 文件（段首轮指纹命名，幂等），正文按轮组织，
  收尾归档补充工具数据（前端 history 不含、跨轮还原的关键增量）
- 摘要：结构化字段（意图/结论/关键数据/工具），生成一次即冻结写盘，
  段追加新轮后生成"续摘要"追加为新条目

KV Cache 三不变量（test_session_archive.py 锁定）：
1. 摘要冻结：已注入上下文的摘要文本永不改写，续摘要只追加
2. 摘要确定性靠文件固化：同段所有请求读同一份冻结文本，不在线重生成
3. 段边界只增不变：边界由历史事实（已有轮次文本）决定，新轮次不改旧边界

降级链：embedding 不可用 → 整个 history 视为单段（行为≈原有合并摘要）；
摘要 LLM 失败 → 规则提取临时文本（不冻结，下次再试）；
归档写入失败 → 只影响还原/工具数据保留，不阻塞压缩主流程。
"""
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 轮 query embedding 进程内缓存（query 指纹 -> 向量），
# 使入口分段与收尾分段的重算成本 ≈ 每轮一次 embedding
_EMB_CACHE: dict[str, list[float]] = {}
_EMB_CACHE_MAX = 512
_EMB_LOCK = threading.Lock()

_WRITE_LOCK = threading.RLock()


# ============ 数据结构 ============

@dataclass
class Round:
    """一轮对话（user query + assistant answer）。"""

    query: str
    answer: str
    fp: str = ""

    def __post_init__(self):
        if not self.fp:
            self.fp = _round_fp(self.query, self.answer)


@dataclass
class Segment:
    """任务段：连续相关的轮次集合。"""

    rounds: list[Round] = field(default_factory=list)

    @property
    def first_fp(self) -> str:
        return self.rounds[0].fp if self.rounds else ""

    @property
    def queries_text(self) -> str:
        """段匹配文本：段内全部 user query 拼接（主题代表性最强）。"""
        return "；".join(r.query for r in self.rounds)


def _round_fp(query: str, answer: str) -> str:
    raw = f"{query}|{answer}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def extract_rounds(history: list[dict[str, Any]]) -> list[Round]:
    """从 history 提取轮次（user 消息 + 其后的 assistant 消息配对）。

    system 消息跳过；user 后无 assistant 时按空答案处理（防御非对齐输入）。
    """
    rounds: list[Round] = []
    pending_query: str | None = None
    for m in history or []:
        role = m.get("role", "")
        content = str(m.get("content", "") or "")
        if role == "user":
            if pending_query is not None:
                rounds.append(Round(query=pending_query, answer=""))
            pending_query = content
        elif role == "assistant" and pending_query is not None:
            rounds.append(Round(query=pending_query, answer=content))
            pending_query = None
    if pending_query is not None:
        rounds.append(Round(query=pending_query, answer=""))
    return rounds


# ============ 分段（语义边界，因果稳定） ============

def _embed_queries(queries: list[str]) -> list[list[float] | None]:
    """批量计算 query embedding（带进程内缓存）。失败对应项为 None。"""
    from agent.rag.embedding import embed_texts

    cached: dict[int, list[float]] = {}
    missing: list[tuple[int, str]] = []
    for i, q in enumerate(queries):
        fp = hashlib.md5(q.encode("utf-8")).hexdigest()
        with _EMB_LOCK:
            vec = _EMB_CACHE.get(fp)
        if vec is not None:
            cached[i] = vec
        else:
            missing.append((i, fp, q))

    if missing:
        try:
            vecs = embed_texts([q for _, _, q in missing])
            if vecs is not None and len(vecs) == len(missing):
                for (i, fp, _), vec in zip(missing, vecs, strict=False):
                    cached[i] = list(map(float, vec))
                    with _EMB_LOCK:
                        if len(_EMB_CACHE) >= _EMB_CACHE_MAX:
                            _EMB_CACHE.clear()
                        _EMB_CACHE[fp] = cached[i]
            else:
                logger.debug("[session-archive] query embedding 批量返回异常")
        except Exception as e:
            logger.debug("[session-archive] query embedding 失败（分段降级单段）：%s", e)

    return [cached.get(i) for i in range(len(queries))]


def segment_rounds(rounds: list[Round]) -> list[Segment]:
    """把轮次序列切成任务段。

    相邻轮 query embedding 余弦 < 阈值 → 开新段。边界只依赖已有轮次文本
    （历史事实），新轮次只可能追加新边界、不会改动旧边界（KV Cache
    不变量 3）。embedding 不可用时降级为单段。
    """
    if not rounds:
        return []
    settings = get_settings()
    threshold = settings.SESSION_SEGMENT_SIM_THRESHOLD
    if not settings.SESSION_ARCHIVE_ENABLED or len(rounds) == 1:
        return [Segment(rounds=list(rounds))]

    vecs = _embed_queries([r.query for r in rounds])
    if any(v is None for v in vecs):
        # 任一向量缺失（embedding 不可用）→ 单段降级，行为接近原合并摘要
        return [Segment(rounds=list(rounds))]

    segments = [Segment(rounds=[rounds[0]])]
    for _prev, cur, pv, cv in zip(rounds, rounds[1:], vecs, vecs[1:], strict=False):
        sim = _cosine(pv, cv)
        if sim < threshold:
            segments.append(Segment(rounds=[cur]))
        else:
            segments[-1].rounds.append(cur)
    return segments


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（embedding 已 L2 归一化，内积即余弦；此处仍防御性归一）。"""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


# ============ 归档文件（MD + JSON meta） ============

def _archive_dir() -> Path:
    settings = get_settings()
    d = Path(settings.SESSION_ARCHIVE_DIR)
    if not d.is_absolute():
        # 相对路径锚定项目根（与 longterm 的 memory/ 目录惯例一致）
        d = Path(__file__).resolve().parents[2] / d
    return d


def _seg_path(first_fp: str) -> Path:
    return _archive_dir() / f"{first_fp}.md"


def _load_meta(first_fp: str) -> dict[str, Any] | None:
    """读段文件 meta（JSON 单行 frontmatter）。文件不存在返回 None。"""
    p = _seg_path(first_fp)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
        m = re.match(r"^---\nmeta: (.*)\n---\n", text, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(1))
    except Exception as e:
        logger.warning("[session-archive] 段文件 meta 解析失败 fp=%s：%s", first_fp, e)
        return None


def _read_body(first_fp: str) -> str:
    p = _seg_path(first_fp)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    m = re.match(r"^---\nmeta: .*\n---\n", text, re.DOTALL)
    return text[m.end():] if m else text


def _save_segment(first_fp: str, meta: dict[str, Any], body: str) -> None:
    """原子写段文件（tmp + replace，与 longterm 的 _atomic_write 同惯例）。"""
    d = _archive_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = _seg_path(first_fp)
    tmp = p.with_suffix(".tmp")
    content = f"---\nmeta: {json.dumps(meta, ensure_ascii=False)}\n---\n{body}"
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)


def upsert_round(
    first_fp: str,
    rounds: list[Round],
    round_: Round,
    tool_data: str = "",
) -> None:
    """把一轮写入段文件（幂等：按轮指纹存在则替换——收尾归档用含工具
    数据的版本覆盖入口归档的纯文本版本）。

    Args:
        rounds: 该段当前已知的全部轮次（用于维护 meta 的轮指纹/queries 序列）
        round_: 待写入轮
        tool_data: 工具轨迹文本（收尾归档提供；入口归档为空）
    """
    with _WRITE_LOCK:
        meta = _load_meta(first_fp) or {
            "first_fp": first_fp,
            "round_fps": [],
            "queries": [],
            "summaries": [],   # [{covered: n, text: "..."}]，冻结后只追加
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        body = _read_body(first_fp)
        section = _format_round_section(round_, tool_data)
        pattern = re.compile(rf"^### 轮 {round_.fp}\n.*?(?=^### 轮 |\Z)", re.DOTALL | re.MULTILINE)
        if pattern.search(body):
            body = pattern.sub(section, body)
        else:
            body = body + section
        # 维护轮序列（以调用方给定的 rounds 为准，幂等去重）
        known = list(meta.get("round_fps", []))
        queries = list(meta.get("queries", []))
        if round_.fp not in known:
            known.append(round_.fp)
            queries.append(round_.query)
        meta["round_fps"] = known
        meta["queries"] = queries
        try:
            _save_segment(first_fp, meta, body)
        except Exception as e:
            logger.warning("[session-archive] 段写入失败 fp=%s：%s", first_fp, e)


def _format_round_section(round_: Round, tool_data: str) -> str:
    lines = [f"### 轮 {round_.fp}", f"用户：{round_.query}"]
    if tool_data:
        lines.append(f"工具轨迹：\n{tool_data}")
    lines.append(f"回答：{round_.answer}")
    return "\n".join(lines) + "\n\n"


def archive_rounds(segments: list[Segment], tool_data_by_fp: dict[str, str] | None = None) -> None:
    """入口归档：把分段结果写入段文件（已存在且轮已归档则跳过）。"""
    settings = get_settings()
    if not settings.SESSION_ARCHIVE_ENABLED:
        return
    tool_data_by_fp = tool_data_by_fp or {}
    for seg in segments:
        meta = _load_meta(seg.first_fp)
        for r in seg.rounds:
            if meta and r.fp in meta.get("round_fps", []):
                continue  # 幂等：该轮已归档
            upsert_round(seg.first_fp, seg.rounds, r, tool_data_by_fp.get(r.fp, ""))


# ============ 冻结摘要 ============

def ensure_summaries(seg: Segment, needed_covered: int, seg_index: int) -> list[str]:
    """确保段有覆盖 needed_covered 轮的冻结摘要，返回摘要文本列表。

    不变量 1（冻结）：已存在的摘要条目永不改写；覆盖不足时生成"续摘要"
    追加新条目。不变量 2（文件固化）：摘要生成一次写盘，后续请求只读。
    LLM 失败 → 规则提取临时文本（不写盘、下次重试）。
    """
    meta = _load_meta(seg.first_fp)
    summaries: list[dict] = (meta or {}).get("summaries", [])
    covered = summaries[-1]["covered"] if summaries else 0

    if covered < needed_covered:
        slice_ = seg.rounds[covered:needed_covered]
        text = _generate_summary_llm(slice_, seg_index, covered)
        if text is None:
            # 降级：规则提取，不冻结（不写盘，跨请求可能变化——仅异常场景）
            return [s["text"] for s in summaries] + [_fallback_summary(slice_)]
        summaries = summaries + [{"covered": needed_covered, "text": text}]
        # 冻结写盘（body 原样保留）
        if meta is not None:
            meta["summaries"] = summaries
            _save_segment(seg.first_fp, meta, _read_body(seg.first_fp))
        else:
            # 段文件尚不存在（归档未跑/失败）——落一份仅含摘要的文件保冻结
            _save_segment(seg.first_fp, {
                "first_fp": seg.first_fp,
                "round_fps": [r.fp for r in seg.rounds[:needed_covered]],
                "queries": [r.query for r in seg.rounds[:needed_covered]],
                "summaries": summaries,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }, "".join(_format_round_section(r, "") for r in seg.rounds[:needed_covered]))
    return [s["text"] for s in summaries]


def _generate_summary_llm(rounds: list[Round], seg_index: int, covered: int) -> str | None:
    """LLM 生成结构化段摘要。失败返回 None（由调用方降级）。"""
    try:
        from agent.prompts.compact import SEGMENT_SUMMARY_SYSTEM_PROMPT
        from app.core.llm import LLM_TIMEOUTS, extract_content, get_llm_client, get_llm_config

        conversation = "\n".join(
            f"用户：{r.query}\n回答：{r.answer[:400]}" for r in rounds
        )
        settings = get_llm_config()
        client = get_llm_client().with_options(timeout=LLM_TIMEOUTS["chat"])
        resp = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": SEGMENT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"待总结的任务段（第 {seg_index + 1} 个历史任务）：\n{conversation}"},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        text = extract_content(resp.choices[0].message).strip()
        return text or None
    except Exception as e:
        logger.warning("[session-archive] 段摘要 LLM 生成失败（降级规则提取）：%s", e)
        return None


def _fallback_summary(rounds: list[Round]) -> str:
    """规则提取的临时摘要（不冻结）：query 列表 + 末轮答案首行。"""
    queries = "；".join(r.query[:40] for r in rounds)
    last_answer = rounds[-1].answer[:80] if rounds else ""
    return f"[历史任务] 意图：{queries}｜结论：{last_answer}"


# ============ 压缩（compact 的分段实现） ============

def compact_with_segments(
    history: list[dict[str, Any]],
    keep_recent_rounds: int,
) -> list[dict[str, Any]]:
    """分段压缩：早段 → 冻结摘要消息（每段可能多条：初始+续），近 N 轮原文。

    返回消息序列 [段摘要 system 消息...] + history[-keep_msgs:]。
    替换原 compact_history 的 LLM 合并摘要路径（有损且跨请求不稳定）。
    """
    rounds = extract_rounds(history)
    segments = segment_rounds(rounds)
    # 异步入档（不阻塞；失败只影响按需还原与工具数据保留）
    _archive_async(segments)

    keep_msgs = keep_recent_rounds * 2
    window = history[len(history) - keep_msgs:] if keep_msgs > 0 else []
    if not window:
        return history  # 窗口覆盖全部 history，无需压缩
    window_rounds = {r.fp for r in extract_rounds(window)}

    # 逐段产出摘要消息；跨窗口段只摘要其窗口外轮次
    msgs: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        outside = [r for r in seg.rounds if r.fp not in window_rounds]
        if not outside:
            continue  # 段整体在窗口内（原文保留）
        summaries = ensure_summaries(seg, needed_covered=len(outside), seg_index=idx)
        for s in summaries:
            msgs.append({"role": "system", "content": s})
    msgs.extend(window)
    return msgs


def _archive_async(segments: list[Segment]) -> None:
    """后台线程执行入口归档（压缩主流程不等待磁盘 IO）。"""
    t = threading.Thread(target=archive_rounds, args=(segments,), daemon=True)
    t.start()


# ============ 按需还原（匹配） ============

def recall_relevant_segments(
    query: str,
    history: list[dict[str, Any]],
    keep_recent_rounds: int,
) -> str:
    """匹配还原：当前 query vs 早段（窗口外）匹配向量，命中返回段全文文本。

    末尾注入（KV Cache 友好）；embedding 不可用/无命中返回空串。
    """
    settings = get_settings()
    if not settings.SESSION_ARCHIVE_ENABLED:
        return ""
    rounds = extract_rounds(history)
    segments = segment_rounds(rounds)
    keep_msgs = keep_recent_rounds * 2
    window = history[len(history) - keep_msgs:] if keep_msgs > 0 else []
    window_rounds = {r.fp for r in extract_rounds(window)}

    early = [seg for seg in segments if any(r.fp not in window_rounds for r in seg.rounds)]
    if not early:
        return ""

    q_vec = _embed_queries([query])[0]
    if q_vec is None:
        # query 向量都算不出（embedding 不可用）→ 对全部早段关键词降级
        return _recall_by_keyword(query, early)

    targets = [(seg, v) for seg in early if (v := _segment_match_vec(seg)) is not None]
    if not targets:
        return _recall_by_keyword(query, early)

    scored = sorted(
        ((seg, _cosine(q_vec, vec)) for seg, vec in targets),
        key=lambda x: x[1], reverse=True,
    )
    recalled: list[str] = []
    for seg, score in scored[: settings.SESSION_ARCHIVE_TOP_K]:
        if score < settings.SESSION_ARCHIVE_MATCH_THRESHOLD:
            continue
        body = _read_body(seg.first_fp).strip()
        if not body:
            body = "\n".join(f"用户：{r.query}\n回答：{r.answer}" for r in seg.rounds)
        recalled.append(body)
    if not recalled:
        return ""
    return ("【相关历史任务还原（系统按相关性加载的往轮记录，含工具数据，参考）】\n"
            + "\n\n".join(recalled))


def _segment_match_vec(seg: Segment) -> list[float] | None:
    """段匹配向量：embedding(段内 query 拼接)。段短时直接现算（有缓存）。"""
    return _embed_queries([seg.queries_text])[0]


def _recall_by_keyword(query: str, segments: list[Segment]) -> str:
    """embedding 不可用时的关键词降级：query 字符与段 query 的重合命中。"""
    chars = {c for c in query if not c.isspace()}
    if not chars:
        return ""
    best: tuple[Segment, float] | None = None
    for seg in segments:
        seg_chars = {c for c in seg.queries_text if not c.isspace()}
        overlap = len(chars & seg_chars) / len(chars)
        if best is None or overlap > best[1]:
            best = (seg, overlap)
    if best and best[1] >= 0.3:
        seg = best[0]
        body = _read_body(seg.first_fp).strip()
        return ("【相关历史任务还原（关键词降级匹配）】\n" + body) if body else ""
    return ""


# ============ 收尾归档（补工具数据） ============

def archive_completed_round_async(
    history: list[dict[str, Any]],
    query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]] | None,
    warning_level: str = "",
) -> None:
    """请求收尾异步归档：本轮（含工具轨迹）追加进所属段。

    分段对 history+本轮 重算（确定性，无进程状态依赖；embedding 缓存使
    增量成本 ≈ 1 次 embed）。工具数据是前端 history 不含的关键增量。
    """
    settings = get_settings()
    if not settings.SESSION_ARCHIVE_ENABLED:
        return

    try:
        t = threading.Thread(
            target=_archive_completed_round_sync,
            args=(history, query, final_answer, tool_calls, warning_level),
            daemon=True,
        )
        t.start()
    except Exception as e:
        logger.debug("[session-archive] 收尾归档调度失败：%s", e)


def _archive_completed_round_sync(
    history: list[dict[str, Any]],
    query: str,
    final_answer: str,
    tool_calls: list[dict[str, Any]] | None,
    warning_level: str = "",
) -> None:
    """收尾归档同步实现（供异步包装与测试调用）。"""
    try:
        all_rounds = extract_rounds(history or []) + [Round(query=query, answer=final_answer)]
        segments = segment_rounds(all_rounds)
        if not segments:
            return
        seg = segments[-1]  # 本轮所属段
        tool_text = _format_tool_calls(tool_calls or [], warning_level)
        last_fp = seg.rounds[-1].fp
        # 段内全部轮入档（入口归档可能没跑过），当前轮带工具数据
        for r in seg.rounds:
            upsert_round(seg.first_fp, seg.rounds, r,
                         tool_text if r.fp == last_fp else "")
        logger.info("[session-archive] 本轮已归档段 fp=%s（含工具数据）", seg.first_fp)
    except Exception as e:
        logger.warning("[session-archive] 收尾归档失败（不影响响应）：%s", e)


def _format_tool_calls(tool_calls: list[dict[str, Any]], warning_level: str) -> str:
    lines = []
    if warning_level:
        lines.append(f"预警等级：{warning_level}")
    for tc in tool_calls:
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        result = tc.get("result", {})
        err = tc.get("error", "")
        line = f"- {name}({json.dumps(args, ensure_ascii=False)[:200]})"
        if err:
            line += f" → 失败：{err}"
        else:
            line += f" → {json.dumps(result, ensure_ascii=False)[:500]}"
        lines.append(line)
    return "\n".join(lines)


def cleanup_archive() -> int:
    """删除超龄归档段（Curator 周期调用）。返回删除的文件数。"""
    settings = get_settings()
    d = _archive_dir()
    if not d.exists():
        return 0
    cutoff = datetime.now().timestamp() - settings.SESSION_ARCHIVE_MAX_AGE_DAYS * 86400
    removed = 0
    for p in d.glob("*.md"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("[session-archive] 清理超龄归档 %d 个（> %d 天）",
                    removed, settings.SESSION_ARCHIVE_MAX_AGE_DAYS)
    return removed
