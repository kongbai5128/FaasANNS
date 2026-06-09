# 02. 接入阿里云函数计算

这一阶段的目标是：让常驻服务器在高 QPS 时，把候选召回阶段卸载到阿里云函数计算。

当前代码中的连接点：

```text
src/faas/aliyun_fc_provider.py
functions/ann_candidate_search/
configs/server.local.json -> faas.provider
```

## 1. 重要原则

函数计算不应该保存或返回 raw vectors。

函数侧只做：

```text
query vector
  -> lightweight candidate index
  -> candidate ids + approximate scores
```

VM/ECS 侧继续做：

```text
candidate ids
  -> raw vector fetch
  -> exact rerank
  -> final top-k
```

## 2. 当前函数目录

```text
functions/ann_candidate_search/
├── handler.py
├── index_loader.py
└── requirements.txt
```

当前 `index_loader.py` 还是占位实现：

```text
返回确定性的 candidate ids，用于打通调用协议
```

真正部署前，需要把它替换成：

```text
加载轻量 HNSW / graph / compressed index
执行候选召回
返回 candidate ids
```

第一版可以先把一个小型 HNSW 索引放入函数验证调用链。

## 3. 函数输入输出协议

服务器发送 payload：

```json
{
  "request_id": "query-0",
  "query": [0.1, 0.2, "..."],
  "candidate_k": 120,
  "ef_search": 80
}
```

函数返回：

```json
{
  "request_id": "query-0",
  "candidates": [
    {"id": 123, "approx_score": 1.23},
    {"id": 456, "approx_score": 2.34}
  ]
}
```

函数不要返回 raw vectors。

## 4. 部署方式选择

阿里云函数计算常见方式：

```text
方式 A：代码包函数
方式 B：自定义容器函数
```

对 FaasANN 更推荐：

```text
自定义容器函数
```

原因：

- hnswlib/faiss 这类依赖更好打包；
- 可以控制系统库；
- 后续函数侧 index 文件也更方便放入镜像或挂载存储。

## 5. 函数侧索引来源

可选方式：

```text
方式 A：索引文件打进容器镜像
  - 优点：冷启动时本地可读
  - 缺点：镜像变大，更新慢

方式 B：函数启动时从 OSS 下载索引到 /tmp
  - 优点：镜像轻
  - 缺点：冷启动有下载开销

方式 C：挂载 NAS
  - 优点：多个函数实例共享索引文件
  - 缺点：需要配置 VPC/NAS，首次访问仍有 I/O 成本
```

第一版建议：

```text
先用小索引打进镜像，验证调用链。
后续再切换到 OSS 或 NAS。
```

## 6. 创建 HTTP Trigger

函数计算需要暴露 HTTP Trigger，供 ECS 服务器调用。

得到一个 endpoint，例如：

```text
https://xxxx.cn-hangzhou.fcapp.run
```

然后修改 ECS 上的配置：

```json
{
  "faas": {
    "provider": "aliyun_http",
    "invoke_timeout_seconds": 10.0,
    "warmup_timeout_seconds": 5.0,
    "endpoints": {
      "default": "https://xxxx.cn-hangzhou.fcapp.run"
    }
  }
}
```

如果先不想自动按 QPS 切换，可以强制所有查询走 FaaS：

```json
{
  "search": {
    "force_faas": true
  }
}
```

也可以请求时传：

```json
{
  "use_faas": true
}
```

## 7. 验证函数调用

先直接 curl 函数：

```bash
curl -X POST 'https://xxxx.cn-hangzhou.fcapp.run' \
  -H 'content-type: application/json' \
  -d '{
    "request_id": "debug",
    "query": [0.0, 0.0, 0.0],
    "candidate_k": 10,
    "ef_search": 80
  }'
```

再从 ECS 服务器验证：

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{
    "query_id": 0,
    "k": 10,
    "use_faas": true
  }'
```

最后跑 benchmark：

```bash
cd /home/admin/FaasANN/scripts
./run_queries.sh --query-num 1000 --concurrent-requests 100 --use-faas
```

## 8. 需要后续改造的地方

当前函数代码还是占位逻辑。后续要改：

```text
functions/ann_candidate_search/index_loader.py
```

目标：

```text
1. 冷启动时加载函数侧轻量索引
2. warmup 请求只触发索引加载
3. 查询请求执行 candidate search
4. 返回全局 vector ids
```

如果后面做分区，则 payload 需要增加：

```json
{
  "shard_id": "...",
  "index_uri": "..."
}
```

服务器侧再做多 shard fanout 和 candidate merge。
