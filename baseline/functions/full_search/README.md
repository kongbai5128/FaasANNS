# full_search baseline

这个目录是“云函数全量 HNSW 搜索”baseline。函数端加载：

```text
FAASANN_DATA_ROOT=/mnt/faasann
/mnt/faasann/sift100w/index/full/full_hnsw.bin
/mnt/faasann/sift100w/index/full/full_hnsw.meta.json
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
./baseline/functions/full_search/test/run_queries.sh --dataset xxx
```

## 请求参数

- `request_id`：请求标识，函数会原样返回。
- `query`：单条查询向量，长度必须等于当前数据集维度，例如 GIST 为 960。
- `candidate_k`：full search 直接返回的结果数量。测试脚本会取前 `k` 个计算 recall，所以这里通常设置为 `k` 即可。
- `ef_search`：hnswlib 查询深度，越大通常 recall 越高、查询越慢。
- `type=status`：返回当前实例索引状态。
- `type=warmup`：触发加载 full HNSW 索引并返回加载后的状态。

## 返回字段

- `request_id`：请求标识。
- `candidates`：搜索结果列表，每个元素包含 `id` 和 `approx_score`。这里是 full HNSW 的近似搜索结果。
- `cold_start_id`：当前函数实例加载索引时生成的唯一 id。同一个热实例处理的请求会看到相同的 `cold_start_id`。
- `index_loaded_at`：当前实例索引加载完成的 Unix 秒级时间戳。
- `timings_ms`：本次请求在函数进程内部统计的耗时，单位是毫秒。
- `function_metrics`：函数实例和本次结果的补充指标，不一定都是耗时。

## timings_ms 字段

- `handler_total`：函数 handler 处理本次请求的总耗时。包含本次请求内部的解析、加载等待、HNSW 查询和格式化；不包含客户端到 FC 网关、FC 排队、HTTP 传输等函数外耗时。
- `search_total`：`search_with_timings` 的总耗时。包含 `load_state`、`query_parse`、`hnsw_set_ef`、`hnsw_knn_query` 和 `format_candidates`。
- `load_state`：进入索引状态获取逻辑的耗时。如果本实例已经加载完成，通常接近 0。
- `load_wait`：等待进程内加载锁的耗时。只有多个请求在同一实例内同时触发加载或等待加载时才可能明显变大。
- `index_load`：本次请求实际执行 full HNSW 索引加载的耗时。现在 `app.py` 会在监听 9000 端口前先 `warmup()`，所以正常 query 里通常是 0；如果用 `type=warmup` 或未预加载路径触发加载，这个字段会包含加载耗时。
- `query_parse`：把 JSON 里的 `query` 转成 `float32` NumPy 向量的耗时。
- `hnsw_set_ef`：调用 `hnswlib.Index.set_ef()` 设置查询深度的耗时。
- `hnsw_knn_query`：hnswlib `knn_query()` 的实际搜索耗时。
- `format_candidates`：把 hnswlib 返回的 ids/distances 转成 JSON `candidates` 的耗时。
- `warmup_total`：`type=warmup` 请求触发预热时的总耗时。

加载耗时有两个口径：`timings_ms.index_load` 是“本次请求是否亲自加载了索引”；`function_metrics.index_load_ms` 是“当前实例加载索引实际用了多久”。如果索引是在启动阶段、监听端口之前加载完成的，query 的 `timings_ms.index_load` 会是 0，但 `function_metrics.index_load_ms` 仍然能看到该实例的加载时延。

## function_metrics 字段

- `candidate_count`：本次返回的结果数量。
- `index_file_size_bytes`：函数端加载的 `full_hnsw.bin` 文件大小，单位是字节。
- `index_load_ms`：当前实例加载索引的实际耗时，单位是毫秒。该值保存在实例状态里，热实例后续请求仍会返回同一个加载耗时。

## /health 或 status 的 index 字段

- `loaded`：当前实例是否已经加载索引。
- `dataset`：当前实例使用的数据集目录名，例如 `sift100w` 或 `gist`。
- `index_path`：实际加载的 full HNSW 索引路径。
- `dimension`：向量维度。
- `vector_count`：索引内向量数量。
- `space`：hnswlib 距离空间，例如 `l2`。
- `cold_start_id`：当前实例加载索引时生成的唯一 id。
- `index_loaded_at`：索引加载完成时间戳。
- `index_load_ms`：当前实例加载索引耗时。
- `index_file_size_bytes`：索引文件大小。

## 测试 CSV 字段

`baseline/functions/full_search/test/run_queries.py` 写入 CSV 时会汇总：

- `client_elapsed_s`：本批测试从客户端视角看到的总耗时，单位是秒。
- `qps_client`：客户端吞吐量，等于 `query_count / client_elapsed_s`。
- `avg_client_ms`：单个请求客户端平均耗时，包含网络、FC 调度、排队和函数执行。
- `avg_function_handler_ms`：云函数内部 `handler_total` 平均值。
- `avg_function_search_ms`：云函数内部 `search_total` 平均值。
- `avg_function_load_state_ms`：云函数内部 `load_state` 平均值。
- `avg_function_index_load_ms`：云函数本次请求级 `index_load` 平均值；如果索引已在启动阶段加载完成，通常为 0。
- `avg_function_hnsw_query_ms`：云函数内部 `hnsw_knn_query` 平均值。
- `avg_function_format_ms`：云函数内部 `format_candidates` 平均值。
- `cold_start_num`：本批测试期间新出现的 `cold_start_id` 数量，也就是观察到几次实例索引加载。
- `avg_cold_start_load_ms`：如果 `cold_start_num` 不为 0，按新出现的 `cold_start_id` 去重后统计每个容器 `function_metrics.index_load_ms` 的平均值；没有新加载容器时为 0。
- `recall`：按测试脚本读取的 groundtruth 计算的 Recall@`k`。
