"""VM 侧原始向量数据层。

vectors 模块只处理 raw vectors：读取 fvecs、保留向量矩阵、按 id 取向量、计算精确距离和 rerank。
这部分是重状态，设计上常驻在本机服务器/VM，不随热点候选搜索一起卸载到 FaaS。
"""
