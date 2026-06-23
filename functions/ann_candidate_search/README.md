# ann_candidate_search

这个目录是上传到阿里云函数计算 FC Web 函数的 Faiss HNSW-PQ 第一阶段候选召回代码。

函数端只保存压缩 Faiss 索引，不保存 raw vectors，也不做 exact rerank。VM 服务器负责保存原始向量，
并在收到函数返回的 candidate ids 后做第二阶段精确搜索。

函数端需要从挂载目录加载：

```text
/mnt/faasann/<dataset>/index/full/pq/faiss_hnswpq.index
```

## FC 配置

- 函数类型：Web 函数
- 运行环境：自定义运行时 / Linux / Python 3.10
- 启动命令：`python3 app.py gist`，或用 `FAASANN_DATASET=gist`
- 监听端口：`9000`

函数依赖为 `numpy` 和 `faiss-cpu`。构建上传包：

```bash
cd /home/qian/Code/FaasANN
./scripts/build_fc_package.sh
```

生成 Faiss HNSW-PQ 索引：

```bash
python data_generator/hnsw/build_pq_index.py \
  --src data/gist/gist_base.fvecs \
  --dst data/gist/index/full/pq \
  --subspaces 120 \
  --codebook-size 256 \
  --train-size 1000000 \
  --iterations 50 \
  --seed 0 \
  --hnsw-space l2 \
  --hnsw-m 48 \
  --hnsw-ef-construction 400 \
  --hnsw-ef-search 1000 \
  --hnsw-batch-size 50000
```

## 环境变量

默认路径以 `FAASANN_DATA_ROOT=/mnt/faasann` 为根目录：

```text
FAASANN_DATA_ROOT=/mnt/faasann
```

候选召回：

```bash
curl --noproxy '*' -X POST https://函数地址 \
  -H "content-type: application/json" \
  -d '{"request_id":"test","query":[...960 floats...],"candidate_k":1000,"ef_search":1000}'
```

## 请求参数

- `request_id`：请求标识，函数会原样返回，方便和客户端日志对齐。
- `query`：单条查询向量，长度必须等于当前数据集维度，例如 GIST 为 960。
- `candidate_k`：从函数端 HNSW-PQ 返回的候选数量。FaasANN 的最终 top-k 精排不在函数里做，而是在 VM 服务器上用 raw vectors 做。
- `ef_search`：Faiss HNSW 搜索深度，通常越大 recall 越高、函数端搜索越慢。
- `type=status`：返回当前实例索引状态，不执行查询。
- `type=warmup`：触发加载索引并返回加载后的状态。

## 返回字段

- `request_id`：请求标识。
- `candidates`：函数召回的候选列表，每个元素包含 `id` 和 `approx_score`。`id` 是向量编号，`approx_score` 是 PQ/HNSW 近似距离，只用于候选召回。
- `cold_start_id`：当前函数实例加载索引时生成的唯一 id。同一个热实例处理的请求会看到相同的 `cold_start_id`。
- `index_loaded_at`：当前实例索引加载完成的 Unix 秒级时间戳。
- `timings_ms`：本次请求在函数进程内部统计的耗时，单位是毫秒。
- `function_metrics`：函数实例和本次结果的补充指标，不一定都是耗时。

## timings_ms 字段

- `handler_total`：函数 handler 处理本次请求的总耗时。包含本次请求内部的解析、加载等待、搜索和格式化；不包含客户端到 FC 网关、FC 排队、HTTP 传输等函数外耗时。
- `search_total`：`search_with_timings` 的总耗时。包含 `load_state`、`query_parse`、`faiss_search` 和 `format_candidates`。
- `load_state`：进入索引状态获取逻辑的耗时。如果本实例已经加载完成，通常接近 0。
- `load_wait`：等待进程内加载锁的耗时。只有多个请求在同一实例内同时触发加载或等待加载时才可能明显变大。
- `index_load`：本次请求实际执行索引加载的耗时。现在 `app.py` 会在监听 9000 端口前先 `warmup()`，所以正常 query 里通常是 0；如果用 `type=warmup` 或未预加载路径触发加载，这个字段会包含加载耗时。
- `query_parse`：把 JSON 里的 `query` 转成 `float32` NumPy 向量的耗时。
- `index_stat`：读取索引文件大小等文件元信息的耗时。只在本次请求实际加载索引时出现。
- `faiss_read_index`：`faiss.read_index()` 读取 HNSW-PQ 索引文件的耗时。只在本次请求实际加载索引时出现。
- `faiss_search`：Faiss HNSW-PQ 候选搜索耗时。
- `format_candidates`：把 Faiss 返回的 ids/distances 转成 JSON `candidates` 的耗时。
- `warmup_total`：`type=warmup` 请求触发预热时的总耗时。

加载耗时有两个口径：`timings_ms.index_load` 是“本次请求是否亲自加载了索引”；`function_metrics.index_load_ms` 是“当前实例加载索引实际用了多久”。如果索引是在启动阶段、监听端口之前加载完成的，query 的 `timings_ms.index_load` 会是 0，但 `function_metrics.index_load_ms` 仍然能看到该实例的加载时延。

## function_metrics 字段

- `candidate_count`：本次返回的候选数量。
- `index_file_size_bytes`：函数端加载的 `faiss_hnswpq.index` 文件大小，单位是字节。
- `index_load_ms`：当前实例加载索引的实际耗时，单位是毫秒。该值保存在实例状态里，热实例后续请求仍会返回同一个加载耗时。

## /health 或 status 的 index 字段

- `loaded`：当前实例是否已经加载索引。
- `dataset`：当前实例使用的数据集目录名，例如 `sift100w` 或 `gist`。
- `index_path`：实际加载的 HNSW-PQ 索引路径。
- `dimension`：向量维度。
- `vector_count`：索引内向量数量。
- `cold_start_id`：当前实例加载索引时生成的唯一 id。
- `index_loaded_at`：索引加载完成时间戳。
- `index_load_ms`：当前实例加载索引耗时。
- `index_file_size_bytes`：索引文件大小。

## FaasANN 服务器和测试 CSV 字段

通过 VM 服务器 `/search` 调用时，函数返回的 `timings_ms` 会被透传为 `function_timings_ms`，函数返回的 `function_metrics` 会被透传为 `function_metrics`。如果使用 function entry 模式，客户端直接请求云函数，云函数会把 VM `/rerank` 返回的 `timings_ms` 透传为 `server_timings_ms`。

`tests/hnsw/run_queries.py` 写入 CSV 时会汇总：

- `avg_entry_request_ms`：客户端到本次测试入口的一次请求平均耗时。server entry 时入口是 VM `/search`；function entry 时入口是云函数。
- `avg_function_request_ms`：一次云函数 HTTP 请求平均耗时。server entry 时是 VM 调用云函数；function entry 时等于客户端到云函数入口请求。
- `avg_function_handler_ms`：云函数 handler 总耗时，来自函数内部 `handler_total`。
- `avg_function_ann_search_ms`：云函数 ANN 搜索 kernel 耗时，对应 Faiss HNSW-PQ `faiss_search`。
- `avg_function_rerank_ms`：云函数内 exact rerank 耗时；当前 `ann_candidate_search` 只返回候选，固定为 0。
- `avg_server_total_ms`：VM server 总耗时。server entry 时来自 `/search` 的 `total`；function entry 时来自 `/rerank` 的 `total`。
- `avg_server_candidate_stage_ms`：VM server 候选召回阶段耗时。function entry 时候选召回在云函数内完成，因此为 0。
- `avg_server_rerank_ms`：VM server raw-vector exact rerank 耗时。
- `cold_start_num`：本批测试期间新出现的 `cold_start_id` 数量，也就是观察到几次实例索引加载。
- `avg_cold_start_load_ms`：如果 `cold_start_num` 不为 0，按新出现的 `cold_start_id` 去重后统计每个容器 `function_metrics.index_load_ms` 的平均值；没有新加载容器时为 0。
