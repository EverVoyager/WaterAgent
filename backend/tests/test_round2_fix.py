"""验证第2轮规划不再卡住的回归测试。"""
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


def test_stream(query: str, label: str, timeout: int = 180):
    print(f"\n{'=' * 70}")
    print(f"[{label}] {query}")
    print("=" * 70)
    url = "http://127.0.0.1:8000/api/agent/query/stream"
    payload = {"query": query, "history": []}
    start = time.time()
    event_count = 0
    answer_text = ""
    reasoning_count = 0
    planner_count = 0
    final_data = None
    last_event_at = start
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
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
                now = time.time()
                elapsed = now - start
                last_event_at = now
                etype = event.get("type")
                if etype == "reasoning_step":
                    reasoning_count += 1
                    step = event.get("step", "")
                    phase = event.get("phase", "")
                    msg = event.get("message", "")
                    if step == "planner":
                        planner_count += 1
                    print(f"  +{elapsed:6.2f}s [reasoning] {step:12s} {phase:8s} | {msg}")
                elif etype == "intent":
                    print(f"  +{elapsed:6.2f}s [intent]    {event.get('intent')}")
                elif etype == "tool_call":
                    print(f"  +{elapsed:6.2f}s [tool_call] {event.get('tool')} args={event.get('arguments')}")
                elif etype == "tool_result":
                    err = event.get("error", "")
                    src = event.get("result", {}).get("source", "")
                    print(f"  +{elapsed:6.2f}s [tool_res]  {event.get('tool')} "
                          f"{'ERR:' + err if err else 'OK src=' + src}")
                elif etype == "synth_meta":
                    data = event.get("data", {})
                    print(f"  +{elapsed:6.2f}s [synth_meta] level={data.get('warning_level', '')} "
                          f"actions={len(data.get('actions', []))}")
                elif etype == "answer_delta":
                    answer_text += event.get("content", "")
                elif etype == "done":
                    final_data = event.get("data")
                    print(f"  +{elapsed:6.2f}s [done] level={final_data.get('warning_level', '')} "
                          f"intent={final_data.get('intent')} rounds={final_data.get('rounds', 0)}")
                elif etype == "error":
                    print(f"  +{elapsed:6.2f}s [error] {event.get('message')}")
                    return False
    except Exception as e:
        print(f"  [exception] {e}")
        return False

    total = time.time() - start
    print(f"\n  事件总数: {event_count}（推理步骤: {reasoning_count}, planner 步骤: {planner_count}）")
    print(f"  流式答案长度: {len(answer_text)}")
    print(f"  总耗时: {total:.2f}s")
    if final_data:
        print(f"  最终 answer 长度: {len(final_data.get('answer', ''))}")
        print(f"  答案前 150 字: {(answer_text or final_data.get('answer', ''))[:150]}")
    return True


def main():
    # 重点测试：复合问题，会触发多轮规划（水情 + 降雨预测）
    cases = [
        ("吴堡站当前水情如何？未来24小时会不会涨水？", "复合查询-多轮规划"),
        ("吴堡站当前水情如何？", "单工具查询"),
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
