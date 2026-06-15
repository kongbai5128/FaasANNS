# full_search baseline

这个目录是“云函数全量 HNSW 搜索”baseline。函数端加载：

```text
FAASANN_DATA_ROOT=/mnt/faasann
/mnt/faasann/index/full/full_hnsw.bin
/mnt/faasann/index/full/full_hnsw.meta.json
```

接口输入 `query`、`candidate_k`、`ef_search`，返回 `candidates`。这里的 `candidate_k` 是返回候选数量，`ef_search` 是 hnswlib 查询深度。

FC 配置：

```text
启动命令: python3 app.py
监听端口: 9000
环境变量: FAASANN_DATA_ROOT=/mnt/faasann
```

本地打包：

```bash
./baseline/functions/full_search/scripts/build_package.sh
```

发送测试查询：

```bash
./baseline/functions/full_search/test/run_queries.sh
```
