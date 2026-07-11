"""HTTP 请求 schema。

这里定义 `/search` 的输入字段：可以通过 `query_id` 复用数据集中的向量，也可以直接传入
原始 query vector。`use_faas` 用于实验时强制切换本地 HNSW / 云函数 PQ 路径。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    request_id: str | None = None
    query_id: int | None = None
    vector: list[float] | None = None
    k: int | None = Field(default=None, gt=0)
    candidate_k: int | None = Field(default=None, gt=0)
    ef_search: int | None = Field(default=None, gt=0)
    use_faas: bool | None = None
