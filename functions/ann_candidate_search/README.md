# ann_candidate_search

这个目录是上传到阿里云函数计算 FC Web 函数的 Faiss HNSW-PQ 第一阶段候选召回代码。

函数端只保存压缩 Faiss 索引，不保存 raw vectors，也不做 exact rerank。VM 服务器负责保存原始向量，
并在收到函数返回的 candidate ids 后做第二阶段精确搜索。

函数端需要从挂载目录加载：

```text
/mnt/faasann/index/full/pq/faiss_hnswpq.index
```

## FC 配置

- 函数类型：Web 函数
- 运行环境：自定义运行时 / Linux / Python 3.10
- 启动命令：`python3 app.py`
- 监听端口：`9000`

函数依赖为 `numpy` 和 `faiss-cpu`。构建上传包：

```bash
cd /home/qian/Code/FaasANN
./scripts/build_fc_package.sh
```

生成 Faiss HNSW-PQ 索引：

```bash
python data_generator/hnsw/build_pq_index.py \
  --src data/sift100w/sift_base.fvecs \
  --dst data/index/full/pq \
  --subspaces 16 \
  --codebook-size 256 \
  --train-size 1000000 \
  --iterations 25 \
  --seed 0 \
  --hnsw-space l2 \
  --hnsw-m 32 \
  --hnsw-ef-construction 200 \
  --hnsw-ef-search 160 \
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
  -d '{"request_id":"test","query":[...128 floats...],"candidate_k":120,"ef_search":160}'
```
