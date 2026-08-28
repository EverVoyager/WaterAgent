"""查看 Qdrant 向量库数据。

用法：
    python inspect_qdrant.py                    # 查看 Collection 概况 + 前 5 条 points
    python inspect_qdrant.py --list 20          # 列出前 20 条 points
    python inspect_qdrant.py --get 3            # 查看 ID=3 的 point 详情
    python inspect_qdrant.py --search "吕梁防汛" # 测试检索
    python inspect_qdrant.py --count            # 只看总数
"""
import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.llm import get_qdrant_client, get_qdrant_config  # noqa: E402


def show_overview():
    """显示 Collection 概况。"""
    client = get_qdrant_client()
    cfg = get_qdrant_config()
    collection = cfg["collection"]

    print("=" * 60)
    print("Qdrant 数据概况")
    print("=" * 60)
    print(f"服务: {cfg['host']}:{cfg['port']}")
    print(f"Collection: {collection}")

    # 所有 collections
    cols = client.get_collections().collections
    print(f"\n所有 Collections ({len(cols)}):")
    for c in cols:
        info = client.get_collection(c.name)
        print(f"  - {c.name}: {info.points_count} points, dim={info.config.params.vectors.size}")

    # 当前 collection 详情
    info = client.get_collection(collection)
    print(f"\n[{collection}] 详情:")
    print(f"  points 数量: {info.points_count}")
    print(f"  向量维度: {info.config.params.vectors.size}")
    print(f"  距离度量: {info.config.params.vectors.distance}")
    print(f"  索引状态: {info.status}")
    print("  payload 索引:")
    for field, schema in (info.payload_schema or {}).items():
        print(f"    - {field}: {schema.data_type}")


def list_points(limit=10):
    """列出前 N 条 points。"""
    client = get_qdrant_client()
    collection = get_qdrant_config()["collection"]

    print(f"\n[{collection}] 前 {limit} 条 points:")
    print("-" * 60)
    result, _ = client.scroll(
        collection_name=collection,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    for p in result:
        payload = p.payload or {}
        print(f"ID={p.id} | {payload.get('title', '')} | {payload.get('chapter', '')}")
        print(f"  article: {payload.get('article', '(无)')}")
        print(f"  doc_type: {payload.get('doc_type', '')} | source: {payload.get('source_file', '')}")
        text = payload.get("text", "").replace("\n", " ")
        print(f"  text: {text[:100]}...")
        print()


def get_point(point_id):
    """查看单个 point 详情。"""
    client = get_qdrant_client()
    collection = get_qdrant_config()["collection"]

    print(f"\n[{collection}] Point ID={point_id}:")
    print("-" * 60)
    result, _ = client.scroll(
        collection_name=collection,
        scroll_filter=None,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    # 用 get 接口更直接
    try:
        points = client.get(collection_name=collection, ids=[point_id])
        for p in points:
            print(json.dumps({
                "id": p.id,
                "payload": p.payload,
                "vector_dim": len(p.vector) if p.vector else 0,
            }, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"获取失败: {e}")


def search_test(query, top_k=3):
    """测试检索。"""
    from agent.rag.vector_store import search_regulations
    print(f"\n检索: {query}")
    print("-" * 60)
    hits = search_regulations(query=query, top_k=top_k)
    if not hits:
        print("无命中")
        return
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h['score']:.4f} | {h['title']} | {h['article'] or h['chapter']}")
        print(f"    doc_type: {h['doc_type']} | source: {h['source_file']}")
        text = h["content"].replace("\n", " ")
        print(f"    content: {text[:120]}...")
        print()


def count_points():
    """统计 points 总数。"""
    client = get_qdrant_client()
    collection = get_qdrant_config()["collection"]
    count = client.count(collection_name=collection, exact=True).count
    print(f"\n[{collection}] 总数: {count} points")


def main():
    parser = argparse.ArgumentParser(description="查看 Qdrant 向量库数据")
    parser.add_argument("--list", type=int, nargs="?", const=10, default=None,
                        help="列出前 N 条 points（默认 10）")
    parser.add_argument("--get", type=int, help="查看指定 ID 的 point 详情")
    parser.add_argument("--search", type=str, help="测试检索")
    parser.add_argument("--count", action="store_true", help="只看总数")
    args = parser.parse_args()

    if args.count:
        count_points()
    elif args.get is not None:
        get_point(args.get)
    elif args.search:
        search_test(args.search)
    elif args.list is not None:
        list_points(args.list)
    else:
        show_overview()
        count_points()
        list_points(5)


if __name__ == "__main__":
    main()
