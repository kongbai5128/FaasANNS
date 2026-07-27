# sharded_hnsw_search baseline

这个 baseline 使用 8 个 k-means 分区 HNSW 索引。8 个云函数上传同一个 zip，区别只在启动命令里的 `shard_id`。

运行形态：

```text
client -> shard 0 coordinator
            |-> shard 0 local HNSW
            |-> shard 1 local_search
            |-> ...
            |-> shard 7 local_search
         -> merge top-k -> client
```

## 索引目录

默认数据目录：

```text
FAASANN_DATA_ROOT=/mnt/faasann
/mnt/faasann/<dataset>/sharding_8/
  manifest.json
  centroids.npy
  labels.npy
  partition_0/
    hnsw.index
    ids.npy
    base.fvecs
    meta.json
  ...
  partition_7/
```

每个云函数只加载自己的分区：

```text
<sharding_dir>/partition_<shard_id>/hnsw.index
<sharding_dir>/partition_<shard_id>/ids.npy
```

## 构建 GIST 8 分区索引

```bash
/home/qian/Code/FaasANNS/.venv/bin/python data_generator/hnsw/build_kmeans_sharded_faiss_hnsw_index.py \
  --src data/gist/gist_base.fvecs \
  --dst data/gist/sharding_8 \
  --shards 8 \
  --train-size 200000 \
  --kmeans-iterations 25 \
  --seed 0 \
  --hnsw-m 48 \
  --hnsw-ef-construction 1250 \
  --hnsw-ef-search 100 \
  --hnsw-batch-size 50000
```

旧的 `data_generator/hnsw/constrained_kmeans_w_clusters.py --k 8` 也能做 k-means 分区，但它输出 hnswlib `partition_*.bin`。本 baseline 使用 Faiss `hnsw.index`。

## 打包

只生成一个 zip：

```bash
./baseline/functions/sharded_hnsw_search/scripts/build_package.sh
```

输出：

```text
dist/baseline_sharded_hnsw_search.zip
```

把这个同一个 zip 上传到 8 个云函数即可。zip 内容不包含 `shard_id.txt`，分区身份由启动命令或 `FAASANN_SHARD_ID` 决定。

## 云函数启动命令

当前 8 个 FC URL：

```text
0: http://sharding-fazpzlktgc.cn-hongkong-vpc.fcapp.run
1: http://sharding-fazozlktgc.cn-hongkong-vpc.fcapp.run
2: http://sharding-faznzlktgc.cn-hongkong-vpc.fcapp.run
3: http://sharding-fazmzlktgc.cn-hongkong-vpc.fcapp.run
4: http://sharding-faztzlktgc.cn-hongkong-vpc.fcapp.run
5: http://sharding-fazszlktgc.cn-hongkong-vpc.fcapp.run
6: http://sharding-fazrzlktgc.cn-hongkong-vpc.fcapp.run
7: http://sharding-fazqzlktgc.cn-hongkong-vpc.fcapp.run
```

每个云函数使用同一个 zip，但启动命令不同：

```bash
# shard 0 coordinator
python3 app.py gist 0

# worker shards
python3 app.py gist 1
python3 app.py gist 2
python3 app.py gist 3
python3 app.py gist 4
python3 app.py gist 5
python3 app.py gist 6
python3 app.py gist 7
```

所有云函数都需要能访问：

```bash
FAASANN_DATA_ROOT=/mnt/faasann
FAASANN_SHARDING_DIR=/mnt/faasann/gist/sharding_8
```

默认情况下索引在第一个查询到达时懒加载，这样 cold-start 查询会触发 shard0 和 1..7 peer shard 的分区加载。若需要恢复启动前预加载，可设置：

```bash
FAASANN_PRELOAD_INDEX=1
```

代码里 shard 0 默认 peer endpoints 已经设置为 1..7 号 URL；需要覆盖时再设置：

```bash
FAASANN_PEER_ENDPOINTS=http://sharding-fazozlktgc.cn-hongkong-vpc.fcapp.run,http://sharding-faznzlktgc.cn-hongkong-vpc.fcapp.run,http://sharding-fazmzlktgc.cn-hongkong-vpc.fcapp.run,http://sharding-faztzlktgc.cn-hongkong-vpc.fcapp.run,http://sharding-fazszlktgc.cn-hongkong-vpc.fcapp.run,http://sharding-fazrzlktgc.cn-hongkong-vpc.fcapp.run,http://sharding-fazqzlktgc.cn-hongkong-vpc.fcapp.run
```

## 本地启动示例

0 号：

```bash
cd baseline/functions/sharded_hnsw_search/src
FAASANN_DATA_ROOT=/home/qian/Code/FaasANNS/data \
FAASANN_SHARDING_DIR=/home/qian/Code/FaasANNS/data/gist/sharding_8 \
FAASANN_PEER_ENDPOINTS=http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003,http://127.0.0.1:9004,http://127.0.0.1:9005,http://127.0.0.1:9006,http://127.0.0.1:9007 \
PORT=9000 python3 app.py gist 0
```

1 号示例：

```bash
cd baseline/functions/sharded_hnsw_search/src
FAASANN_DATA_ROOT=/home/qian/Code/FaasANNS/data \
FAASANN_SHARDING_DIR=/home/qian/Code/FaasANNS/data/gist/sharding_8 \
PORT=9001 python3 app.py gist 1
```

## 运行

```
./baseline/functions/sharded_hnsw_search/test/run_queries.sh
p99尾延迟
P99_WINDOW_SECONDS=1 ./baseline/functions/sharded_hnsw_search/test/run_queries.sh
```

## 请求类型

- 普通请求：发给 0 号 coordinator。
- `type=local_search`：0 号内部发给 peer shard；也可以手动调试单分区搜索。
- `type=status`：查看当前机器加载的 shard。
- `type=warmup`：触发当前机器加载自己的 partition index。

## 计时字段

- `query_elapsed_s` / `qps_query_only`：测试脚本中 1000 次查询阶段的总时间和 QPS。默认没有单独 warmup，所以 cold-start 的首个 query 会包含加载影响。
- `run_elapsed_s` / `qps_run`：脚本总时间和 QPS。
- `query_max_cold_start_index_load_ms`：普通 query 响应中观测到的、本轮新冷启动 shard 的最大索引加载时间。
- `query_max_reported_index_load_ms`：8 个 shard 上报的历史 `index_load_ms` 最大值；热实例也会保留这个值。
- `avg_coordinator_handler_ms`：shard0 完整处理一次查询的时间，包含并发请求 peer、等待返回和 merge。
- `avg_coordinator_fanout_ms`：shard0 从开始 fanout 到 1..7 全部 peer 返回的时间。
- `avg_peer_request_max_ms`：每个 query 中最慢 peer HTTP 请求耗时的平均值。
- `avg_peer_faiss_search_max_ms`：每个 query 中 1..7 号 peer 最慢 Faiss HNSW 搜索耗时的平均值。
- `avg_all_shard_faiss_search_max_ms`：每个 query 中 0..7 号 shard 最慢 Faiss HNSW 搜索耗时的平均值。
