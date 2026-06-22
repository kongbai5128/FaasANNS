# two_stage_search baseline

这个目录是“不使用 VM 的云函数两阶段搜索”baseline。函数端加载：

```text
FAASANN_DATA_ROOT=/mnt/faasann
/mnt/faasann/sift100w/index/full/pq/faiss_hnswpq.index
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
./baseline/functions/Two_stage_search/scripts/build_package.sh
```

发送测试查询：

```bash
./baseline/functions/Two_stage_search/test/run_queries.sh --dataset xxx
```

## 请求参数

- `request_id`：请求标识，函数会原样返回。
- `query`：单条查询向量，长度必须等于当前数据集维度，例如 GIST 为 960。
- `k`：精排后最终返回的 top-k 数量。
- `candidate_k`：第一阶段 HNSW-PQ 召回的候选数量。它决定后面 raw vector exact rerank 要处理多少条候选。
- `ef_search`：Faiss HNSW-PQ 查询深度，通常越大 recall 越高、函数端候选搜索越慢。
- `type=status`：返回当前实例索引状态。
- `type=warmup`：触发加载 PQ 索引和 raw vector memmap 并返回加载后的状态。

## 返回字段

- `request_id`：请求标识。
- `candidates`：最终精排结果列表，每个元素包含 `id` 和 `score`。`score` 是 raw vector exact L2 距离。
- `cold_start_id`：当前函数实例加载索引时生成的唯一 id。同一个热实例处理的请求会看到相同的 `cold_start_id`。
- `index_loaded_at`：当前实例索引加载完成的 Unix 秒级时间戳。
- `timings_ms`：本次请求在函数进程内部统计的耗时，单位是毫秒。
- `function_metrics`：函数实例和本次结果的补充指标，不一定都是耗时。

## timings_ms 字段

- `handler_total`：函数 handler 处理本次请求的总耗时。包含本次请求内部的解析、加载等待、HNSW-PQ 候选搜索、raw vector 精排和格式化；不包含客户端到 FC 网关、FC 排队、HTTP 传输等函数外耗时。
- `search_total`：`search_with_timings` 的总耗时。包含 `load_state`、`query_parse`、`faiss_search` 和 `rerank_total`。
- `load_state`：进入索引状态获取逻辑的耗时。如果本实例已经加载完成，通常接近 0。
- `load_wait`：等待进程内加载锁的耗时。只有多个请求在同一实例内同时触发加载或等待加载时才可能明显变大。
- `index_load`：本次请求实际执行索引加载的耗时。这里加载的是 Faiss HNSW-PQ 索引，并建立 raw fvecs 的 memmap 视图。现在 `app.py` 会在监听 9000 端口前先 `warmup()`，所以正常 query 里通常是 0；如果用 `type=warmup` 或未预加载路径触发加载，这个字段会包含加载耗时。
- `query_parse`：把 JSON 里的 `query` 转成 `float32` NumPy 向量的耗时。
- `faiss_search`：第一阶段 Faiss HNSW-PQ 候选搜索耗时。
- `rerank_total`：第二阶段 raw vector exact rerank 总耗时。
- `dedupe_ids`：对第一阶段候选 id 去重并过滤非法 id 的耗时。
- `memmap_gather`：从 raw fvecs memmap 中按候选 id 取出候选向量的耗时。
- `l2_scores`：计算候选向量和 query 的 L2 距离耗时。
- `argsort`：按 L2 距离排序并取 top-k 的耗时。
- `format_results`：把精排结果转成 JSON `candidates` 的耗时。
- `warmup_total`：`type=warmup` 请求触发预热时的总耗时。

加载耗时有两个口径：`timings_ms.index_load` 是“本次请求是否亲自加载了索引和 memmap”；`function_metrics.index_load_ms` 是“当前实例加载索引和 memmap 实际用了多久”。如果索引是在启动阶段、监听端口之前加载完成的，query 的 `timings_ms.index_load` 会是 0，但 `function_metrics.index_load_ms` 仍然能看到该实例的加载时延。

## function_metrics 字段

- `candidate_count`：本次返回的最终结果数量，通常等于 `k`。注意这里不是第一阶段 PQ 原始候选数量。
- `index_file_size_bytes`：函数端加载的 `faiss_hnswpq.index` 文件大小，单位是字节。
- `base_file_size_bytes`：函数端读取的 raw fvecs 文件大小，单位是字节。
- `index_load_ms`：当前实例加载 PQ 索引并建立 raw fvecs memmap 的实际耗时，单位是毫秒。该值保存在实例状态里，热实例后续请求仍会返回同一个加载耗时。

## /health 或 status 的 index 字段

- `loaded`：当前实例是否已经加载索引。
- `dataset`：当前实例使用的数据集目录名，例如 `sift100w` 或 `gist`。
- `index_type`：索引类型，这里为 `faiss_hnswpq`。
- `index_path`：实际加载的 HNSW-PQ 索引路径。
- `base_path`：raw fvecs 文件路径。
- `dimension`：向量维度。
- `vector_count`：索引内向量数量。
- `cold_start_id`：当前实例加载索引时生成的唯一 id。
- `index_loaded_at`：索引加载完成时间戳。
- `index_load_ms`：当前实例加载索引和 memmap 耗时。
- `index_file_size_bytes`：HNSW-PQ 索引文件大小。
- `base_file_size_bytes`：raw fvecs 文件大小。

## 测试 CSV 字段

`baseline/functions/Two_stage_search/test/run_queries.py` 写入 CSV 时会汇总：

- `client_elapsed_s`：本批测试从客户端视角看到的总耗时，单位是秒。
- `qps_client`：客户端吞吐量，等于 `query_count / client_elapsed_s`。
- `avg_client_ms`：单个请求客户端平均耗时，包含网络、FC 调度、排队和函数执行。
- `avg_function_handler_ms`：云函数内部 `handler_total` 平均值。
- `avg_function_search_ms`：云函数内部 `search_total` 平均值。
- `avg_function_load_state_ms`：云函数内部 `load_state` 平均值。
- `avg_function_index_load_ms`：云函数本次请求级 `index_load` 平均值；如果索引已在启动阶段加载完成，通常为 0。
- `avg_function_faiss_search_ms`：云函数内部第一阶段 `faiss_search` 平均值。
- `avg_function_rerank_ms`：云函数内部第二阶段 `rerank_total` 平均值。
- `avg_function_memmap_gather_ms`：云函数内部 `memmap_gather` 平均值。
- `avg_function_l2_scores_ms`：云函数内部 `l2_scores` 平均值。
- `cold_start_num`：本批测试期间新出现的 `cold_start_id` 数量，也就是观察到几次实例索引加载。
- `avg_cold_start_load_ms`：如果 `cold_start_num` 不为 0，按新出现的 `cold_start_id` 去重后统计每个容器 `function_metrics.index_load_ms` 的平均值；没有新加载容器时为 0。
- `recall`：按测试脚本读取的 groundtruth 计算的 Recall@`k`。
