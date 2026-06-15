# full_search query test

默认直接请求云函数，不经过本地 VM：

```bash
./baseline/functions/full_search/test/run_queries.sh
```

默认只发 10 个 query、1 并发。这个 baseline 每个 query 都在云函数里扫描
100 万条原始向量，不适合直接用 100 并发压默认公网域名。
脚本默认使用 HTTP，避免 FC 默认公网域名在 HTTPS 下偶发 TLS 断连影响测试。

更大规模测试时显式指定参数：

```bash
./baseline/functions/full_search/test/run_queries.sh \
  --query-num 100 \
  --concurrent-requests 1
```

如果 FC HTTP trigger 要求鉴权，需要传：

```bash
./baseline/functions/full_search/test/run_queries.sh \
  --authorization "$FAASANN_FC_AUTHORIZATION"
```

结果默认写入：

```text
baseline/functions/full_search/test/result/run_queries.csv
```
