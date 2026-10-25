"""LLM 流的可见输出：隐藏思考内容、过滤引用并记录用量。"""
from agent.utils import CitationMarkerFilter
from app.core.llm_stats import record_llm_usage


def stream_answer_events(stream, usage_node: str, valid_ref_ids: set[int] | None = None):
    """跨 chunk 过滤 <think> 与引用标记，产出增量和最终完整答案。"""
    cite_filter = CitationMarkerFilter(valid_ref_ids=valid_ref_ids)
    filtered_answer: list[str] = []
    buffer = ""
    in_think = False
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage:
            record_llm_usage(usage_node, usage)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if not content:
            continue
        buffer += content
        output = ""
        while buffer:
            if in_think:
                end = buffer.find("</think>")
                if end == -1:
                    break
                buffer = buffer[end + len("</think>"):]
                while buffer and buffer[0] in " \n\t":
                    buffer = buffer[1:]
                in_think = False
            else:
                start = buffer.find("<think>")
                if start == -1:
                    partial = 0
                    for plen in range(min(len(buffer), len("<think>")), 0, -1):
                        if buffer.endswith("<think>"[:plen]):
                            partial = plen
                            break
                    output += buffer[:len(buffer) - partial]
                    buffer = buffer[len(buffer) - partial:]
                    break
                output += buffer[:start]
                buffer = buffer[start + len("<think>"):]
                in_think = True
        if output:
            emitted = cite_filter.feed(output)
            if emitted:
                filtered_answer.append(emitted)
                yield {"type": "answer_delta", "content": emitted}

    tail = cite_filter.flush()
    if tail:
        filtered_answer.append(tail)

    yield {"type": "synth_answer_full", "content": "".join(filtered_answer)}
