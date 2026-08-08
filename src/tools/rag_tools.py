"""
RAG 历史检索工具
基于 TF-IDF 向量检索 reports/ 目录下的历史分析报告
让 Agent 能够「回忆」之前生成过的学情分析，支持跨课对比
"""

import os
import glob
from typing import Dict, Any, List

from .base import BaseTool, ToolParameter


class SearchHistoryTool(BaseTool):
    """RAG 工具：检索历史学情分析报告"""

    name = "search_history"
    description = (
        "检索历史学情分析报告。当用户询问「之前的分析」「上周的报告」「历史数据」"
        "或需要对比过往课堂分析结果时调用此工具。基于 TF-IDF 语义相似度匹配最相关的报告片段。"
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="检索查询文本，如「专注度趋势」「走神率分析」「第一节课的报告」",
            required=True,
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description="返回结果数量",
            required=False,
            default=3,
        ),
    ]

    def execute(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """执行 RAG 检索"""
        reports_dir = os.environ.get("REPORTS_DIR", "reports")
        if not os.path.isdir(reports_dir):
            # 也尝试 outputs/reports 目录
            reports_dir = "outputs/reports"

        # 收集所有报告文件
        report_files = []
        for ext in ("*.txt", "*.md"):
            report_files.extend(glob.glob(os.path.join(reports_dir, ext)))

        if not report_files:
            return {
                "total_results": 0,
                "results": [],
                "message": "未找到历史报告文件。请先生成学情分析报告。",
            }

        # 读取报告内容
        documents = []
        for filepath in report_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    # 将长报告分块（每 500 字一块，重叠 100 字）
                    chunks = self._chunk_text(content, chunk_size=500, overlap=100)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "filename": os.path.basename(filepath),
                            "filepath": filepath,
                            "chunk_id": i,
                            "content": chunk,
                        })
            except Exception:
                continue

        if not documents:
            return {
                "total_results": 0,
                "results": [],
                "message": "历史报告文件为空。",
            }

        # TF-IDF 相似度检索
        results = self._tfidf_search(query, documents, top_k)

        return {
            "total_results": len(results),
            "results": results,
            "query": query,
        }

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """将长文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def _tfidf_search(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """基于 TF-IDF 的相似度检索"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            # 降级为简单关键词匹配
            return self._keyword_search(query, documents, top_k)

        corpus = [doc["content"] for doc in documents]
        corpus.append(query)

        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # 查询向量是最后一个
        query_vec = tfidf_matrix[-1:]
        doc_vecs = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()

        # 取 top_k 个最相似的
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        seen_files = set()
        for idx in top_indices:
            if similarities[idx] <= 0:
                continue
            doc = documents[idx]
            # 同一文件只取最相关的一个块
            if doc["filename"] in seen_files and len(results) > 0:
                continue
            seen_files.add(doc["filename"])
            results.append({
                "filename": doc["filename"],
                "filepath": doc["filepath"],
                "score": float(similarities[idx]),
                "snippet": doc["content"][:300],
            })

        return results

    def _keyword_search(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """降级：简单关键词匹配"""
        query_words = set(query.lower().split())
        scored = []
        for doc in documents:
            content_words = set(doc["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_files = set()
        for score, doc in scored[:top_k * 2]:
            if doc["filename"] in seen_files:
                continue
            seen_files.add(doc["filename"])
            results.append({
                "filename": doc["filename"],
                "filepath": doc["filepath"],
                "score": float(score) / max(len(query_words), 1),
                "snippet": doc["content"][:300],
            })

        return results[:top_k]
