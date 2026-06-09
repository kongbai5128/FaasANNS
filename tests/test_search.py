"""搜索结果精排单元测试。

当前测试覆盖 VectorStore 的 exact rerank 行为，确保候选 id 会按 raw vector L2 距离重新排序。
"""

from __future__ import annotations

from vectors.vector_store import VectorStore


def test_vector_store_rerank_orders_by_l2() -> None:
    store = VectorStore.synthetic(dimension=4, count=10)
    query = store.get(0)
    results = store.rerank(query, [{"id": 3}, {"id": 0}, {"id": 2}], k=2)
    assert results[0].id == 0
    assert len(results) == 2
