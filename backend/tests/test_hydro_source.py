"""快速验证水文数据源（qqjjsj.com）可访问性 + 解析结构。"""
import re
import urllib.request
from datetime import datetime


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def find_latest_yellow_river_article() -> tuple[str, str]:
    """从列表页找到最新一篇'黄河水文站实时水位'文章。"""
    html = fetch("http://www.qqjjsj.com/list226a1/")
    # 匹配详情页链接 + 标题
    pattern = re.compile(r'href="([^"]*id=\d+[^"]*)"[^>]*>([^<]*黄河[^<]*)</a>')
    links = pattern.findall(html)
    # 过滤"实时水位"或"水位情况"类
    candidates = [
        (url, title.strip())
        for url, title in links
        if "实时水位" in title or "水位情况" in title
    ]
    if not candidates:
        # 退而求其次
        candidates = links
    return candidates[0] if candidates else ("", "")


def parse_hydro_table(article_url: str) -> list[dict]:
    """解析文章中的水文站表格。"""
    html = fetch(article_url)
    # 简单提取 <table> 里的所有 <tr>
    table_match = re.search(r"<table[^>]*>(.*?)</table>", html, re.S)
    if not table_match:
        return []
    table_html = table_match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    results = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(cells) >= 5 and any(s in cells[0] + cells[1] + cells[2] for s in ["黄河", "渭河"]):
            results.append({
                "river": cells[0],
                "station": cells[1],
                "time": cells[2],
                "water_level": cells[3] if len(cells) > 3 else "",
                "flow": cells[4] if len(cells) > 4 else "",
            })
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("[1] 测试列表页可访问性")
    print("=" * 60)
    url, title = find_latest_yellow_river_article()
    print(f"最新文章: {title}")
    print(f"URL: {url}")

    if not url:
        print("未找到文章")
        exit(1)

    print()
    print("=" * 60)
    print("[2] 测试详情页表格解析")
    print("=" * 60)
    records = parse_hydro_table(url)
    print(f"解析到 {len(records)} 条记录")
    for r in records[:8]:
        print(r)

    print()
    print("=" * 60)
    print("[3] 过滤吴堡/龙门/府谷站")
    print("=" * 60)
    target = ["吴堡", "龙门", "府谷"]
    filtered = [r for r in records if r["station"] in target]
    for r in filtered:
        print(r)
