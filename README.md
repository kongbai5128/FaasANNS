# FaasANN

FaasANN is a Python prototype for hybrid VM + FaaS approximate nearest neighbor search.

The current version uses a two-stage VM + FaaS pipeline:

- the VM/server keeps raw vectors and final rerank state;
- local direct search uses one HNSW index;
- when query pressure grows, the server can route first-stage candidate search to Function Compute;
- the function stores only a lightweight PQ index and returns candidate ids;
- exact rerank always happens on the VM against raw vectors.

## Current Layout

```text
configs/                         Runtime config.
data/                            Local datasets, for example SIFT fvecs files.
src/main.py                      Server entrypoint.
src/server/                      FastAPI routes, request schemas, middleware.
src/vectors/                     Raw vector loading, storage, and exact rerank on VM.
src/search/                      Local HNSW search and two-stage PQ-offload service.
src/faas/                        Local and Alibaba Cloud Function Compute providers.
src/scaling/                     QPS metrics, cost/offload planner, function slot pool.
src/utils/                       Config, logging, and timing helpers.
functions/ann_candidate_search/  Function Compute PQ candidate-search handler.
scripts/                         Local run and benchmark helpers.
tests/                           Small unit tests for search/scaling behavior.
```

`vectors` and `search` are intentionally separate:

- `vectors` means heavy raw data on the VM.
- `search` means the retrieval algorithm, currently local HNSW or remote PQ candidate search plus rerank orchestration.

This avoids the earlier `storage` / `index` split, while still making the data and search responsibilities explicit.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build Indexes

Build the local VM HNSW index:

```bash
python data_generator/constrained_kmeans_w_clusters.py \
  --src data/sift100w/sift_base.fvecs \
  --dst data/index/full \
  --k 1 \
  --m 32 \
  --ef-construction 200 \
  --ef-search 80
```

Build the Function Compute PQ index:

```bash
python data_generator/build_pq_index.py \
  --src data/sift100w/sift_base.fvecs \
  --dst data/index/pq \
  --subspaces 16 \
  --codebook-size 256 \
  --train-size 100000 \
  --iterations 25 \
  --seed 0 \
  --batch-size 50000
```

Upload `data/index/pq/pq_meta.json`, `pq_codebooks.npy`, `pq_codes.npy`, and `pq_ids.npy` to the function mount path.

## Run

```bash
python src/main.py --config configs/server.local.json
```

Example query:

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"query_id": 0, "k": 10}'
```

Force the offload path:

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"query_id": 0, "k": 10, "use_faas": true}'
```
