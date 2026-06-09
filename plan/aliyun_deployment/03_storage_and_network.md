# 03. 存储、网络与安全建议

## 1. 数据放在哪里

当前第一版最简单：

```text
ECS 本地磁盘：
  data/sift100w/sift_base.fvecs
  data/index/full/full_hnsw.bin
  data/index/full/full_hnsw.meta.json
```

优点：

- 读取简单；
- 本地磁盘访问快；
- 不需要额外云服务配置。

缺点：

- 换 ECS 时需要重新同步数据；
- 多台服务器共享不方便。

## 2. OSS

OSS 适合作为数据源和备份：

```text
oss://bucket/faasann/sift_base.fvecs
oss://bucket/faasann/full_hnsw.bin
```

部署时从 OSS 下载到 ECS 本地磁盘。

不建议每次 query 从 OSS 读 raw vectors。
raw vectors 应该常驻 VM/ECS 内存或本地高速存储。

## 3. NAS

NAS 适合多实例共享索引或函数侧加载索引。

适用场景：

```text
多个 FC 实例需要共享同一份 graph-index / HNSW / PQ code
多个 ECS/ACK pod 需要共享数据
```

缺点：

```text
需要 VPC/NAS 配置
首次访问延迟较高
吞吐和本地盘相比需要实测
```

## 4. VPC

推荐：

```text
ECS 和 Function Compute 放在同一个地域、同一个 VPC 或可互通网络里
```

如果 FC 需要访问 ECS 的内网地址：

```text
函数计算需要配置 VPC 访问
ECS 安全组需要允许来自函数侧网段的入站流量
```

如果 ECS 只调用 FC 的公网 HTTP Trigger：

```text
配置简单
但有公网链路和安全暴露问题
```

第一版可以先用 HTTP Trigger 公网地址验证。
论文实验或正式性能测试建议切到内网/VPC。

## 5. 安全组

ECS 服务器：

```text
TCP 22    只允许自己的 IP
TCP 8080  只允许压测机 IP 或内网网段
```

如果用 Nginx 反代：

```text
TCP 80/443 对外
TCP 8080 只本机或内网
```

## 6. 日志

服务器日志：

```text
server.log
journalctl -u faasann
```

压测日志：

```text
logs/run_queries.csv
```

函数计算日志：

```text
FC 控制台日志
SLS 日志服务
```

建议记录：

```text
request_id
mode: local/faas
candidate_ms
rerank_ms
remote_invoke_ms
recall
qps
```

## 7. 成本注意点

ECS 成本：

```text
实例规格
磁盘
公网带宽
```

函数计算成本：

```text
调用次数
执行时间
内存规格
公网流量
日志量
```

当前架构的成本优化重点：

```text
低 QPS：主要用 ECS 常驻服务
高峰 QPS：只把候选召回卸载到 FC
避免 raw vectors 在网络中来回传输
```
