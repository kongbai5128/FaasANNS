"""搜索逻辑层。

search 模块负责两阶段检索：本地路径用 HNSW 直接候选召回，FaaS 路径用云函数 PQ 召回，
候选结果统一回到 VM raw vectors 做 exact rerank。
"""
