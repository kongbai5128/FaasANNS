"""扩容控制层。

scaling 模块集中处理 QPS 指标、local/FaaS offload 决策和 FaaS 预热。
真实的函数实例并发、队列和扩缩容交给阿里云 FC。
"""
