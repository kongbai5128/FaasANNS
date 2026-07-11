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
启动命令: python3 app.py [dataset] [memory|nomemory]
监听端口: 9000
环境变量: FAASANN_DATA_ROOT=/mnt/faasann
```

例如：

```bash
python3 app.py gist nomemory
```

`nomemory` 是实例级配置。开启后，函数第二阶段不会把 raw vectors 的 memmap 视图保存在实例状态里，而是在 rerank 时只按候选 id 从 fvecs 文件读取需要的向量。也可以用环境变量 `FAASANN_NOMEMORY=1` 设置。

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

## 字段解释和计时边界

| 字段                              | 起始位置                                                | 结束位置                                    | 来源/说明                                                                                                                                                                                                                         |
| --------------------------------- | ------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `avg_cold_start_load_ms`        | cloud function 实例首次需要加载索引时，进入索引加载逻辑 | 索引文件加载完成并可用于搜索                | 只统计本次测试开始后新出现的 `cold_start_id`。它不是每个请求的耗时，而是冷启动实例的索引加载平均耗时。                                                                                                                          |
| `avg_entry_request_ms`          | 测试客户端调用 `urlopen()` 前                         | 测试客户端收到入口响应并完成 JSON decode 后 | 客户端视角的一次端到端入口请求耗时。baseline 入口是 cloud function；FaasANNS `entrypoint=server` 入口是 VM `/search`；FaasANNS `entrypoint=function` 入口是 cloud function。                                                |
| `avg_function_request_ms`       | 发起一次 cloud function HTTP 请求前                     | 收到 cloud function 响应并解析完成后        | cloud function 外部请求耗时。baseline 中等于 `avg_entry_request_ms`；FaasANNS `entrypoint=server` 中是 VM server 调用 cloud function 的 `remote_invoke`；FaasANNS `entrypoint=function` 中等于 `avg_entry_request_ms`。 |
| `avg_function_handler_ms`       | cloud function handler 开始处理 payload                 | handler 组装完返回 dict/result              | cloud function 内部总耗时，不包含请求到达函数前的 HTTP 网络、网关、排队、客户端 JSON decode 等外部耗时。function-entry 时会包含函数请求 VM `/rerank` 的等待时间。                                                               |
| `avg_function_ann_search_ms`    | cloud function 内部开始执行 ANN 候选搜索                | ANN 搜索返回候选结果                        | 只看 ANN 搜索核。当前 full baseline 是 Faiss HNSWFlat `index.search`；Two-stage baseline 和 FaasANNS 是 Faiss HNSWPQ `index.search`。不包含函数 handler 总包装、HTTP、server rerank。历史 full 结果可能来自 hnswlib。        |
| `avg_function_rerank_ms`        | cloud function 内部开始 exact rerank                    | 函数内 rerank 排序并格式化结果结束          | 只有 Two-stage baseline 在函数内做 exact rerank，所以它非 0。full baseline 不 rerank，FaasANNS 把 rerank 放在 VM server，所以这里为 0。                                                                                           |
| `avg_server_total_ms`           | VM server 收到请求并进入对应 handler/service 逻辑       | VM server 组装完响应 body 前                | VM server 内部总耗时。`entrypoint=server` 时来自 `/search` 的 total；`entrypoint=function` 时来自 `/rerank` 的 total；baseline 没有 VM server，所以为 0。                                                                 |
| `avg_server_candidate_stage_ms` | VM server `/search` 开始候选召回阶段                  | 候选召回阶段结束                            | 只有 FaasANNS `entrypoint=server` 可能非 0，因为这时 VM server 会调用 cloud function 拿候选；`entrypoint=function` 的候选召回已经在函数内完成，所以这里为 0；baseline 也为 0。                                                |
| `avg_server_rerank_ms`          | VM server 开始按原始向量做 exact rerank                 | VM server rerank 排序并生成 top-k 结果结束  | FaasANNS `entrypoint=server` 和 `entrypoint=function` 都可能非 0；baseline 没有 VM server，所以为 0。                                                                                                                         |

## 按路径展开

### baseline

- `avg_entry_request_ms`：client -> cloud function -> client
- `avg_function_request_ms`：client -> cloud function -> client，和 `avg_entry_request_ms` 同口径
- `avg_function_handler_ms`：函数 handler 内部总时间
- `avg_function_ann_search_ms`：函数内 ANN 搜索核
- `avg_function_rerank_ms`：只有 Two-stage baseline 有值
- `avg_server_*`：都为 0

### FaasANNS `entrypoint=server`

- `avg_entry_request_ms`：client -> VM `/search` -> cloud function -> VM rerank -> client
- `avg_function_request_ms`：VM server -> cloud function -> VM server
- `avg_function_handler_ms`：cloud function handler 内部总时间
- `avg_function_ann_search_ms`：cloud function 内 Faiss ANN 搜索
- `avg_server_total_ms`：VM `/search` 内部总时间
- `avg_server_candidate_stage_ms`：VM `/search` 中候选召回阶段，主要包含调用 cloud function 的时间
- `avg_server_rerank_ms`：VM `/search` 中 exact rerank 的时间

### FaasANNS `entrypoint=function`

- `avg_entry_request_ms`：client -> cloud function -> VM `/rerank` -> cloud function -> client
- `avg_function_request_ms`：client -> cloud function -> client，和 `avg_entry_request_ms` 同口径
- `avg_function_handler_ms`：cloud function handler 内部总时间，包含函数内 ANN 搜索和请求 VM `/rerank` 的等待
- `avg_function_ann_search_ms`：cloud function 内 Faiss ANN 搜索
- `avg_server_total_ms`：VM `/rerank` 内部总时间
- `avg_server_candidate_stage_ms`：为 0，因为候选召回不在 VM server 中发生
- `avg_server_rerank_ms`：VM `/rerank` 中 exact rerank 的时间

## 常见对照

- 想看用户实际感受到的一次请求耗时：看 `avg_entry_request_ms`。
- 想看一次云函数请求整体有多慢：看 `avg_function_request_ms`。
- 想看云函数内部代码总耗时：看 `avg_function_handler_ms`。
- 想看 ANN 搜索核本身耗时：看 `avg_function_ann_search_ms`。
- 想看 rerank 在哪里耗时：函数内 rerank 看 `avg_function_rerank_ms`，VM server 内 rerank 看 `avg_server_rerank_ms`。
- `avg_function_request_ms - avg_function_handler_ms` 大致是 cloud function HTTP/运行时包装/网络传输/排队等函数外开销。
- `avg_entry_request_ms - avg_function_handler_ms` 不能直接解释为单一开销，因为它可能跨过 client、VM server、cloud function、rerank 等多层路径。

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
- `index_load`：本次请求实际执行索引加载的耗时。这里加载的是 Faiss HNSW-PQ 索引。现在 `app.py` 会在监听 9000 端口前先 `warmup()`，所以正常 query 里通常是 0；如果用 `type=warmup` 或未预加载路径触发加载，这个字段会包含加载耗时。
- `query_parse`：把 JSON 里的 `query` 转成 `float32` NumPy 向量的耗时。
- `faiss_search`：第一阶段 Faiss HNSW-PQ 候选搜索耗时。
- `rerank_total`：第二阶段 raw vector exact rerank 总耗时。
- `dedupe_ids`：对第一阶段候选 id 去重并过滤非法 id 的耗时。
- `memmap_load`：普通模式下首次建立 raw fvecs memmap 视图的耗时；热实例后续通常为 0。
- `memmap_gather`：普通模式下，从 raw fvecs memmap 中按候选 id 取出候选向量的耗时。
- `direct_fvecs_gather`：`nomemory=true` 时，按候选 id 逐条 seek/read fvecs 记录的耗时。
- `l2_scores`：计算候选向量和 query 的 L2 距离耗时。
- `argsort`：按 L2 距离排序并取 top-k 的耗时。
- `format_results`：把精排结果转成 JSON `candidates` 的耗时。
- `warmup_total`：`type=warmup` 请求触发预热时的总耗时。

加载耗时有两个口径：`timings_ms.index_load` 是“本次请求是否亲自加载了索引和 memmap”；`function_metrics.index_load_ms` 是“当前实例加载索引和 memmap 实际用了多久”。如果索引是在启动阶段、监听端口之前加载完成的，query 的 `timings_ms.index_load` 会是 0，但 `function_metrics.index_load_ms` 仍然能看到该实例的加载时延。

## function_metrics 字段

- `candidate_count`：本次返回的最终结果数量，通常等于 `k`。注意这里不是第一阶段 PQ 原始候选数量。
- `index_file_size_bytes`：函数端加载的 `faiss_hnswpq.index` 文件大小，单位是字节。
- `base_file_size_bytes`：函数端读取的 raw fvecs 文件大小，单位是字节。
- `index_load_ms`：当前实例加载 PQ 索引的实际耗时，单位是毫秒。该值保存在实例状态里，热实例后续请求仍会返回同一个加载耗时。
- `nomemory`：本次请求是否启用按需读取 raw vectors 的模式。
- `vector_memmap_loaded`：当前实例是否已经建立 raw fvecs memmap 视图。
- `vector_memmap_load_ms`：当前实例首次建立 raw fvecs memmap 视图的耗时。

## /health 或 status 的 index 字段

- `loaded`：当前实例是否已经加载索引。
- `dataset`：当前实例使用的数据集目录名，例如 `sift100w` 或 `gist`。
- `index_type`：索引类型，这里为 `faiss_hnswpq`。
- `index_path`：实际加载的 HNSW-PQ 索引路径。
- `base_path`：raw fvecs 文件路径。
- `dimension`：向量维度。
- `vector_count`：索引内向量数量。
- `vector_memmap_loaded`：当前实例是否已经建立 raw fvecs memmap 视图。
- `cold_start_id`：当前实例加载索引时生成的唯一 id。
- `index_loaded_at`：索引加载完成时间戳。
- `index_load_ms`：当前实例加载索引耗时。
- `vector_memmap_load_ms`：当前实例首次建立 raw fvecs memmap 视图耗时。
- `index_file_size_bytes`：HNSW-PQ 索引文件大小。
- `base_file_size_bytes`：raw fvecs 文件大小。

## 测试 CSV 字段

`baseline/functions/Two_stage_search/test/run_queries.py` 写入 CSV 时会汇总：

- `client_elapsed_s`：本批测试从客户端视角看到的总耗时，单位是秒。
- `qps_client`：客户端吞吐量，等于 `query_count / client_elapsed_s`。
- `avg_entry_request_ms`：客户端到本次测试入口的一次请求平均耗时；Two-stage baseline 的入口就是云函数。
- `avg_function_request_ms`：一次云函数 HTTP 请求平均耗时；Two-stage baseline 中与 `avg_entry_request_ms` 相同。
- `avg_function_handler_ms`：云函数 handler 总耗时，来自函数内部 `handler_total`。
- `avg_function_ann_search_ms`：云函数 ANN 搜索 kernel 耗时；Two-stage baseline 对应 Faiss HNSW-PQ `faiss_search`。
- `avg_function_rerank_ms`：云函数内 raw-vector exact rerank 总耗时。
- `avg_server_total_ms`：VM server 总耗时；Two-stage baseline 不经过 VM server，固定为 0。
- `avg_server_candidate_stage_ms`：VM server 候选召回阶段耗时；Two-stage baseline 固定为 0。
- `avg_server_rerank_ms`：VM server 精排耗时；Two-stage baseline 固定为 0。
- `cold_start_num`：本批测试期间新出现的 `cold_start_id` 数量，也就是观察到几次实例索引加载。
- `avg_cold_start_load_ms`：如果 `cold_start_num` 不为 0，按新出现的 `cold_start_id` 去重后统计每个容器 `function_metrics.index_load_ms` 的平均值；没有新加载容器时为 0。
- `recall`：按测试脚本读取的 groundtruth 计算的 Recall@`k`。
