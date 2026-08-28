"""防汛知识问答轨迹生成器。

与业务轨迹不同，知识问答不调用工具，直接用 LLM 生成回答。
轨迹格式：system + user + assistant（无 tool 消息）。

用途：在 SFT 训练集中混入 5% 知识问答，保留模型的自然对话能力，
避免过度工具化导致用户问概念性问题时也强行调用工具。
"""
import json
import logging
import time
from pathlib import Path

from openai import OpenAI

from train.data_gen.seed_queries import SeedQuery

logger = logging.getLogger(__name__)

_KNOWLEDGE_SYSTEM_PROMPT = """你是黄河吕梁段防汛预警智能体，名叫"水卫"。
用户提出的是防汛知识性问题（非实时数据查询），请基于你的专业知识直接回答。

要求：
1. 回答准确、专业，符合防汛业务规范
2. 涉及法规时引用具体法规名称和条款
3. 涉及等级标准时给出具体数值
4. 回答控制在 200-400 字，条理清晰
5. 不要编造数据或法规条款
"""


def synthesize_knowledge_one(client: OpenAI, model: str, seed: SeedQuery,
                             temperature: float = 0.7) -> dict | None:
    """对单条知识查询生成轨迹（无工具调用）。

    返回 {"query": ..., "messages": [...]} 或 None。
    """
    messages = [
        {"role": "system", "content": _KNOWLEDGE_SYSTEM_PROMPT},
        {"role": "user", "content": seed.query},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=600,
        )
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            return None
        messages.append({"role": "assistant", "content": answer})
        return {
            "query": seed.query,
            "messages": messages,
        }
    except Exception as e:
        logger.warning("[knowledge] 合成异常: %s", e)
        return None


def synthesize_knowledge_dataset(client: OpenAI, model: str, seeds: list[SeedQuery],
                                 out_path: Path, rpm: int = 30) -> int:
    """批量生成知识问答轨迹，追加写盘 + 断点续传。

    Returns:
        本次新写入条数。
    """
    from tqdm import tqdm

    # 断点续传：从 messages[1].content 提取 query 去重
    done_queries = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    # 优先用 query 字段，没有则从 messages 提取
                    q = rec.get("query") or rec.get("messages", [{}])[1].get("content", "")
                    if q:
                        done_queries.add(q)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

    pending = [s for s in seeds if s.query not in done_queries]
    if not pending:
        return 0
    interval = 60.0 / max(rpm, 1)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for seed in tqdm(pending, desc="知识合成", unit="条",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            t0 = time.time()
            result = synthesize_knowledge_one(client, model, seed)
            if result is not None:
                f.write(json.dumps({
                    "scenario_id": f"knwl-{hash(seed.query) % 100000}",
                    "level": "",
                    "query": seed.query,
                    "messages": result["messages"],
                }, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
            time.sleep(max(0.0, interval - (time.time() - t0)))
    return written
