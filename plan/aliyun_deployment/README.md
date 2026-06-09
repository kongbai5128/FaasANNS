# FaasANN 阿里云部署计划

这套部署计划按当前项目架构来写：

```text
客户端 / benchmark
  -> 常驻服务器：ECS 或 ACK
       - FastAPI
       - 100 万 raw vectors
       - full_hnsw.bin
       - exact rerank
       - QPS / cost model / WarmupManager
  -> 函数计算 FC：后续热点候选召回
       - 轻量 HNSW / graph / compressed index
       - 返回 candidate ids
       - 不保存 raw vectors
```

当前第一阶段建议：

1. 先把服务器部署到 **ECS**。
2. 跑通公网或内网访问 `/health`、`/search`。
3. 用 `scripts/run_queries.sh` 做 recall / QPS 测试。
4. 再接入 **阿里云函数计算 FC**，只卸载候选召回阶段。

不建议第一阶段直接把整个 FastAPI 服务塞进函数计算，因为当前服务会加载：

- `sift_base.fvecs`，约 516 MB；
- `full_hnsw.bin`，约 752 MB；
- 运行时还会有 raw vector 矩阵和索引内存开销。

这些更适合常驻 VM/ECS，而不是频繁冷启动的函数。

## 文档列表

- [01_ecs_server.md](01_ecs_server.md)：把当前服务器部署到 ECS。
- [02_function_compute.md](02_function_compute.md)：后续把候选召回接入阿里云函数计算。
- [03_storage_and_network.md](03_storage_and_network.md)：数据、网络、安全组、VPC、日志建议。
- [04_operation_commands.md](04_operation_commands.md)：常用命令清单。

## 推荐部署路线

```text
阶段 1：ECS 单机
  - 服务器和数据都在一台 ECS
  - 目标：功能正确、Recall 正常、日志可看

阶段 2：ECS + FC 本地模拟对比
  - 服务器仍在 ECS
  - faas.provider 先保持 local
  - 目标：确认 WarmupManager / offload planner 行为

阶段 3：ECS + Function Compute
  - 服务器在 ECS
  - 函数计算只做 candidate search
  - 目标：验证突发 query 增长时的动态扩容

阶段 4：ACK / 容器化
  - 如果需要更规范的服务编排，再迁移服务器到 ACK
  - 函数计算仍作为热点候选召回层
```

## 当前代码里需要关注的配置

文件：

```text
configs/server.local.json
```

关键字段：

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8080
  },
  "dataset": {
    "base_path": "data/sift100w/sift_base.fvecs",
    "max_vectors": 1000000
  },
  "search": {
    "hnsw": {
      "hnsw_index_path": "data/index/full/full_hnsw.bin"
    },
    "offload_qps_threshold": 20.0,
    "force_faas": false
  },
  "faas": {
    "provider": "local"
  }
}
```

部署到 ECS 后，如果希望外部机器访问，需要把：

```json
"host": "127.0.0.1"
```

改成：

```json
"host": "0.0.0.0"
```

如果只在 ECS 本机压测，保留 `127.0.0.1` 也可以。
