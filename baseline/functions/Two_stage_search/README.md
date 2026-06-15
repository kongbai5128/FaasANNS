# two_stage_search baseline

这个目录是“不使用 VM 的云函数两阶段搜索”baseline。函数端加载：

```text
FAASANN_DATA_ROOT=/mnt/faasann
/mnt/faasann/index/full/pq/faiss_hnswpq.index
/mnt/faasann/sift100w/sift_base.fvecs
```

搜索流程：

```text
query -> HNSW-PQ candidate search -> raw vector exact rerank -> final top-k
```

FC 配置：

```text
启动命令: python3 app.py
监听端口: 9000
环境变量: FAASANN_DATA_ROOT=/mnt/faasann
```

本地打包：

```bash
./baseline/functions/Two_stage\ search/scripts/build_package.sh
```

发送测试查询：

```bash
./baseline/functions/Two_stage\ search/test/run_queries.sh
```
