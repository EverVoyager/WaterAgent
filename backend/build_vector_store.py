"""构建 Qdrant 向量库脚本。

用法（在 backend/ 目录下运行）：
    1. 先启动 Qdrant 服务：双击 start_qdrant.bat（或手动运行 qdrant.exe）
    2. python build_vector_store.py

流程：
    1. 检查 Qdrant 服务可达
    2. 扫描 ../data/raw/regulations/*.md
    3. 切分为 chunks（带 frontmatter metadata）
    4. 调用 DashScope text-embedding-v3 批量向量化
    5. 创建 Qdrant collection + upsert 向量 + payload
    6. 验证检索效果

完成后，Agent 的 search_regulation 工具会自动走真实 RAG 检索。
"""
import sys
from pathlib import Path

# 路径设置
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
# agent 模块在项目根目录下（与 backend/ 平级），需把根目录也加入 sys.path
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.rag.document_loader import load_regulation_files  # noqa: E402
from agent.rag.vector_store import (  # noqa: E402
    build_and_persist_index,
    is_index_ready,
    search_regulations,
)
from app.core.llm import get_qdrant_client, get_qdrant_config  # noqa: E402

# 法规源文件目录
REGULATIONS_DIR = PROJECT_ROOT / "data" / "raw" / "regulations"


def check_qdrant_service() -> bool:
    """检查 Qdrant 服务是否可达。"""
    try:
        client = get_qdrant_client()
        cfg = get_qdrant_config()
        # 调用一次轻量接口
        client.get_collections()
        print(f"[OK] Qdrant 服务可达: {cfg['host']}:{cfg['port']}")
        return True
    except Exception as e:
        print(f"[错误] Qdrant 服务不可达: {e}")
        print()
        print("请先启动 Qdrant 服务：")
        print("  方式1: 双击 backend/start_qdrant.bat")
        print("  方式2: 手动运行下载的 qdrant.exe")
        print("  下载地址: https://github.com/qdrant/qdrant/releases")
        return False


def main():
    print("=" * 60)
    print("构建 Qdrant 法规向量库")
    print("=" * 60)
    cfg = get_qdrant_config()
    print(f"Qdrant 服务: {cfg['host']}:{cfg['port']}")
    print(f"Collection: {cfg['collection']} (dim={cfg['vector_size']})")
    print(f"法规源目录: {REGULATIONS_DIR}")
    print()

    # 0. 检查 Qdrant 服务
    print("[0/3] 检查 Qdrant 服务 ...")
    if not check_qdrant_service():
        sys.exit(1)
    print()

    # 1. 加载并切分法规文档
    if not REGULATIONS_DIR.exists():
        print(f"[错误] 法规源目录不存在: {REGULATIONS_DIR}")
        sys.exit(1)

    md_files = list(REGULATIONS_DIR.glob("*.md"))
    if not md_files:
        print(f"[错误] 法规源目录下没有 .md 文件: {REGULATIONS_DIR}")
        sys.exit(1)

    print(f"[1/3] 加载 {len(md_files)} 个法规文档并切分 ...")
    chunks = load_regulation_files(str(REGULATIONS_DIR))
    if not chunks:
        print("[错误] 切分后 chunks 为空")
        sys.exit(1)
    print(f"      共生成 {len(chunks)} 个 chunks")
    print()

    # 2. 构建并持久化 Qdrant 索引
    print("[2/3] 调用 DashScope embedding 并构建 Qdrant 索引 ...")
    total = build_and_persist_index(chunks)
    print(f"      索引构建完成: {total} 个向量")
    print()

    # 3. 验证：用几个真实问题测试检索
    print("[3/3] 验证检索效果 ...")
    test_queries = [
        ("什么情况下可以宣布进入紧急防汛期？", 3),
        ("黄河中游吴堡站的保证流量是多少？", 3),
        ("Ⅱ级应急响应需要采取哪些措施？", 3),
        ("吕梁市的防汛指挥由谁负责？", 3),
    ]

    if not is_index_ready():
        print("[错误] 索引构建后仍未就绪")
        sys.exit(1)

    for q, k in test_queries:
        print(f"\n  查询: {q}")
        hits = search_regulations(query=q, top_k=k)
        if not hits:
            print("    无命中")
            continue
        for i, h in enumerate(hits, 1):
            print(f"    [{i}] score={h['score']:.4f} | {h['title']} | {h['article'] or h['chapter']}")
            print(f"        {h['content'][:80]}...")

    print()
    print("=" * 60)
    print("构建完成！Agent 的 search_regulation 工具现已走真实 RAG 检索（Qdrant）。")
    print("=" * 60)


if __name__ == "__main__":
    main()
