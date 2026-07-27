"""法规文档加载与切分。

输入：data/raw/regulations/*.md（带 YAML frontmatter）
输出：List[RegulationChunk]，每个 chunk 携带 title/doc_type/chapter/article 等 metadata。

切分策略：
1. 解析 YAML frontmatter 提取 title / doc_type / effective_date 等
2. 按 ## 章节标题分块，每块作为独立 chunk
3. 章节内若超过 chunk_size，使用 RecursiveCharacterTextSplitter 二次切分并保留 overlap
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 默认切分参数
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80
# DashScope text-embedding-v3 单次最多 10 条输入
EMBEDDING_BATCH_SIZE = 10


@dataclass
class RegulationChunk:
    """单个法规 chunk。"""

    text: str                                   # chunk 正文
    title: str                                  # 法规标题（来自 frontmatter）
    doc_type: str = ""                          # 文档类型：法律/行政法规/...
    effective_date: str = ""
    chapter: str = ""                           # 章节标题（如 "第五章 防汛抗洪"）
    article: str = ""                           # 第几条（如 "第四十一条"，可能为空）
    source_file: str = ""                       # 原始文件名
    chunk_index: int = 0                        # 该文件内的 chunk 序号
    extra: Dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, str]:
        """转为可序列化的 metadata dict（用于持久化）。"""
        meta = {
            "title": self.title,
            "doc_type": self.doc_type,
            "effective_date": self.effective_date,
            "chapter": self.chapter,
            "article": self.article,
            "source_file": self.source_file,
            "chunk_index": str(self.chunk_index),
        }
        meta.update(self.extra)
        return meta


# ====== Frontmatter 解析 ======

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[Dict[str, str], str]:
    """解析 Markdown YAML frontmatter。

    Returns:
        (metadata_dict, body_text)
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    yaml_str, body = match.group(1), match.group(2)
    try:
        meta = yaml.safe_load(yaml_str) or {}
        # 统一转字符串
        meta = {k: str(v) for k, v in meta.items() if v is not None}
    except yaml.YAMLError:
        meta = {}
    return meta, body


# ====== 章节切分 ======

_CHAPTER_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_SUBCHAPTER_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
_ARTICLE_RE = re.compile(r"\*\*(第[一二三四五六七八九十百零]+条)\*\*")


def _split_by_chapter(body: str) -> List[tuple[str, str]]:
    """按 ## 章节标题切分正文，再按 ### 子章节进一步切分。

    这样既能保留章节上下文，又能让子章节（如 4.1 Ⅳ级响应、4.2 Ⅲ级响应）
    作为独立 chunk 被精准检索。

    Returns:
        [(chapter_title, sub_content), ...]
        chapter_title 形如 "第五章 防汛抗洪" 或 "4 应急响应 / 4.3 Ⅱ级响应"
    """
    matches = list(_CHAPTER_RE.finditer(body))
    if not matches:
        return [("", body)]

    chunks: List[tuple[str, str]] = []
    for i, m in enumerate(matches):
        chapter_title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chapter_body = body[start:end].strip()
        if not chapter_body:
            continue

        # 在章节内查找 ### 子标题
        sub_matches = list(_SUBCHAPTER_RE.finditer(chapter_body))
        if not sub_matches:
            chunks.append((chapter_title, chapter_body))
            continue

        # 子章节前的引言（若有）
        prelude = chapter_body[: sub_matches[0].start()].strip()
        if prelude:
            chunks.append((chapter_title, prelude))

        for j, sm in enumerate(sub_matches):
            sub_title = sm.group(1).strip()
            sub_start = sm.end()
            sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(chapter_body)
            sub_body = chapter_body[sub_start:sub_end].strip()
            if sub_body:
                # 组合标题：父章节 / 子章节
                chunks.append((f"{chapter_title} / {sub_title}", sub_body))
    return chunks


def _extract_article(text: str) -> str:
    """从 chunk 文本中提取第一个 '第X条' 标记。"""
    m = _ARTICLE_RE.search(text)
    return m.group(1) if m else ""


# ====== 主入口 ======

def split_markdown_to_chunks(
    content: str,
    source_file: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[RegulationChunk]:
    """将单个 Markdown 法规文档切分为 chunks。"""
    meta, body = parse_frontmatter(content)
    title = meta.get("title", Path(source_file).stem or "未命名")
    doc_type = meta.get("doc_type", "")
    effective_date = meta.get("effective_date", "")

    # 移除一级标题（# XXX），避免重复
    body = re.sub(r"^#\s+.+\s*\n", "", body, count=1)

    chapters = _split_by_chapter(body)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks: List[RegulationChunk] = []
    global_idx = 0
    for chapter_title, chapter_body in chapters:
        sub_texts = splitter.split_text(chapter_body)
        for sub in sub_texts:
            sub = sub.strip()
            if not sub:
                continue
            article = _extract_article(sub)
            chunks.append(RegulationChunk(
                text=sub,
                title=title,
                doc_type=doc_type,
                effective_date=effective_date,
                chapter=chapter_title,
                article=article,
                source_file=source_file,
                chunk_index=global_idx,
            ))
            global_idx += 1
    return chunks


def load_regulation_files(directory: str) -> List[RegulationChunk]:
    """加载目录下所有 *.md 法规文档，返回切分后的 chunks。"""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    all_chunks: List[RegulationChunk] = []
    md_files = sorted(dir_path.glob("*.md"))
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[document_loader] 读取失败 {md_file.name}: {e}")
            continue
        chunks = split_markdown_to_chunks(content, source_file=md_file.name)
        all_chunks.extend(chunks)
        print(f"[document_loader] {md_file.name} -> {len(chunks)} chunks")
    return all_chunks
