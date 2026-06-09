# ann_candidate_search

这个目录是上传到阿里云函数计算 FC Web 函数的 PQ 第一阶段候选召回代码。

函数端只保存轻量 PQ 索引，不保存 raw vectors，也不做 exact rerank。VM 服务器负责保存原始向量，
并在收到函数返回的 candidate ids 后做第二阶段精确搜索。

函数端需要从挂载目录加载：

```text
/mnt/faasann/index/pq/pq_meta.json
/mnt/faasann/index/pq/pq_codebooks.npy
/mnt/faasann/index/pq/pq_codes.npy
/mnt/faasann/index/pq/pq_ids.npy
```

`pq_meta.json` 必须包含：

```json
{
  "dimension": 128,
  "vector_count": 1000000,
  "subspace_count": 16
}
```

数组约定：

```text
pq_codebooks.npy  shape=(subspace_count, codebook_size, dimension / subspace_count)
pq_codes.npy      shape=(vector_count, subspace_count)
pq_ids.npy        shape=(vector_count,)
```

## FC 配置

- 函数类型：Web 函数
- 运行环境：自定义运行时 / Linux / Python 3.10
- 启动命令：`python3 app.py`
- 监听端口：`9000`
- 单实例并发度：按实际延迟压测后设置

函数依赖只有 `numpy`。构建上传包：

```bash
cd /home/qian/Code/FaasANN
./scripts/build_fc_package.sh
```

## 环境变量

默认路径以 `FAASANN_DATA_ROOT=/mnt/faasann` 为根目录。也可以显式指定：

```text
FAASANN_PQ_META_PATH=/mnt/faasann/index/pq/pq_meta.json
FAASANN_PQ_CODEBOOKS_PATH=/mnt/faasann/index/pq/pq_codebooks.npy
FAASANN_PQ_CODES_PATH=/mnt/faasann/index/pq/pq_codes.npy
FAASANN_PQ_IDS_PATH=/mnt/faasann/index/pq/pq_ids.npy
```

## 接口

状态：

```bash
curl --noproxy '*' -X POST https://函数地址 \
  -H "content-type: application/json" \
  -d '{"type":"status"}'
```

预热：

```bash
curl --noproxy '*' -X POST https://函数地址 \
  -H "content-type: application/json" \
  -d '{"type":"warmup"}'
```

候选召回：

```bash
curl --noproxy '*' -X POST https://函数地址 \
  -H "content-type: application/json" \
  -d '{"request_id":"test","query":[...128 floats...],"candidate_k":120}'
```

返回：

```json
{
  "request_id": "test",
  "candidates": [
    {"id": 123, "approx_score": 0.42}
  ]
}
```
