"""FaaS 调用抽象层。

faas 模块屏蔽本地模拟器、阿里云 HTTP Trigger 或未来 SDK 调用之间的差异。
搜索服务只依赖统一 provider 接口，便于以后把本地 HNSW 候选搜索替换成云函数候选搜索。
"""
