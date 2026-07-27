"""测试 SSE 流式接口 v2：验证推理过程 + token 级流式输出。"""
import json
import sys
import time
from pathlib import Path

import requests

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_stream(query: str, label: str):
    print(f"\n{'=' * 70}")
    print(f"[{label}] {query}")
    print("=" * 70)
    url = "http://127.0.0.1:8000/api/agent/query/stream"
    payload = {"query": query, "history": []}
    start = time.time()
    event_count = 0
    answer_text = ""
    reasoning_count = 0
    first_token_at = None
    final_data = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                json_str = raw[5:].strip()
                if not json_str:
                    continue
                try:
                    event = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                event_count += 1
                elapsed = time.time() - start
                etype = event.get("type")
                if etype == "reasoning_step":
                    reasoning_count += 1
                    print(f"  +{elapsed:5.2f}s [reasoning] {event.get('step'):12s} "
                          f"{event.get('phase'):8s} | {event.get('message')}")
                elif etype == "intent":
                    print(f"  +{elapsed:5.2f}s [intent]    {event.get('intent')}")
                elif etype == "tool_call":
                    print(f"  +{elapsed:5.2f}s [tool_call] {event.get('tool')} args={event.get('arguments')}")
                elif etype == "tool_result":
                    err = event.get("error", "")
                    print(f"  +{elapsed:5.2f}s [tool_res]  {event.get('tool')} "
                          f"{'ERR:' + err if err else 'OK'}")
                elif etype == "synth_meta":
                    data = event.get("data", {})
                    print(f"  +{elapsed:5.2f}s [synth_meta] level={data.get('warning_level', '')} "
                          f"actions={len(data.get('actions', []))}")
                elif etype == "answer_delta":
                    if first_token_at is None:
                        first_token_at = elapsed
                        print(f"  +{elapsed:5.2f}s [first_token] 首个 answer_delta 到达")
                    answer_text += event.get("content", "")
                elif etype == "done":
                    final_data = event.get("data")
                    print(f"  +{elapsed:5.2f}s [done] level={final_data.get('warning_level', '')} "
                          f"intent={final_data.get('intent')} rounds={final_data.get('rounds', 0)}")
                elif etype == "error":
                    print(f"  +{elapsed:5.2f}s [error] {event.get('message')}")
                    return False
    except Exception as e:
        print(f"  [exception] {e}")
        return False

    total = time.time() - start
    print(f"\n  事件总数: {event_count}（推理步骤: {reasoning_count}）")
    print(f"  首个 answer_delta 延迟: {first_token_at:.2f}s" if first_token_at else "  无 answer_delta")
    print(f"  流式答案长度: {len(answer_text)}")
    print(f"  总耗时: {total:.2f}s")
    if final_data:
        print(f"  最终 answer 长度: {len(final_data.get('answer', ''))}")
        print(f"  答案前 120 字: {(answer_text or final_data.get('answer', ''))[:120]}")
    return True


def main():
    cases = [
        ("你好", "闲聊"),
        ("发生Ⅱ级预警时应该怎么响应？", "法规咨询"),
        ("吴堡站当前水情如何？", "实时水情"),
    ]
    results = []
    for q, label in cases:
        ok = test_stream(q, label)
        results.append((label, ok))

    print(f"\n{'=' * 70}")
    print("测试结果汇总")
    print("=" * 70)
    for label, ok in results:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
