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

## 字段解释和计时边界

| 字段                              | 起始位置                                                | 结束位置                                    | 来源/说明                                                                                                                                                                                                                         |
| --------------------------------- | ------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `avg_cold_start_load_ms`        | cloud function 实例首次需要加载索引时，进入索引加载逻辑 | 索引文件加载完成并可用于搜索                | 只统计本次测试开始后新出现的 `cold_start_id`。它不是每个请求的耗时，而是冷启动实例的索引加载平均耗时。                                                                                                                          |
| `avg_entry_request_ms`          | 测试客户端调用 `urlopen()` 前                         | 测试客户端收到入口响应并完成 JSON decode 后 | 客户端视角的一次端到端入口请求耗时。baseline 入口是 cloud function；FaasANNS `entrypoint=server` 入口是 VM `/search`；FaasANNS `entrypoint=function` 入口是 cloud function。                                                |
| `avg_function_request_ms`       | 发起一次 cloud function HTTP 请求前                     | 收到 cloud function 响应并解析完成后        | cloud function 外部请求耗时。baseline 中等于 `avg_entry_request_ms`；FaasANNS `entrypoint=server` 中是 VM server 调用 cloud function 的 `remote_invoke`；FaasANNS `entrypoint=function` 中等于 `avg_entry_request_ms`。 |
| `avg_function_handler_ms`       | cloud function handler 开始处理 payload                 | handler 组装完返回 dict/result              | cloud function 内部总耗时，不包含请求到达函数前的 HTTP 网络、网关、排队、客户端 JSON decode 等外部耗时。function-entry 时会包含函数请求 VM `/rerank` 的等待时间。                                                               |
| `avg_function_ann_search_ms`    | cloud function 内部开始执行 ANN 候选搜索                | ANN 搜索返回候选结果                        | 只看 ANN 搜索核。full baseline 是 `hnswlib.knn_query`；Two-stage baseline 和 FaasANNS 是 Faiss HNSWPQ `index.search`。不包含函数 handler 总包装、HTTP、server rerank。                                                        |
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
- `avg_entry_request_ms`：客户端到本次测试入口的一次请求平均耗时；full baseline 的入口就是云函数。
- `avg_function_request_ms`：一次云函数 HTTP 请求平均耗时；full baseline 中与 `avg_entry_request_ms` 相同。
- `avg_function_handler_ms`：云函数 handler 总耗时，来自函数内部 `handler_total`。
- `avg_function_ann_search_ms`：云函数 ANN 搜索 kernel 耗时；full baseline 对应 `hnsw_knn_query`。
- `avg_function_rerank_ms`：云函数内 exact rerank 耗时；full baseline 没有该阶段，固定为 0。
- `avg_server_total_ms`：VM server 总耗时；full baseline 不经过 VM server，固定为 0。
- `avg_server_candidate_stage_ms`：VM server 候选召回阶段耗时；full baseline 固定为 0。
- `avg_server_rerank_ms`：VM server 精排耗时；full baseline 固定为 0。
- `cold_start_num`：本批测试期间新出现的 `cold_start_id` 数量，也就是观察到几次实例索引加载。
- `avg_cold_start_load_ms`：如果 `cold_start_num` 不为 0，按新出现的 `cold_start_id` 去重后统计每个容器 `function_metrics.index_load_ms` 的平均值；没有新加载容器时为 0。
- `recall`：按测试脚本读取的 groundtruth 计算的 Recall@`k`。
