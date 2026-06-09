# FaasANN

FaasANN is a Python prototype for hybrid VM + FaaS approximate nearest neighbor search.

The first version is intentionally simple:

- the VM/server keeps raw vectors and final rerank state;
- local search uses one Faiss HNSW index, with a NumPy fallback if Faiss is unavailable;
- when query pressure grows, the server can route the candidate-search stage to a FaaS provider;
- Alibaba Cloud Function Compute is represented by an HTTP provider interface and can be connected later.

## Current Layout

```text
configs/                         Runtime config.
data/                            Local datasets, for example SIFT fvecs files.
src/main.py                      Server entrypoint.
src/server/                      FastAPI routes, request schemas, middleware.
src/vectors/                     Raw vector loading, storage, and exact rerank on VM.
src/search/                      Faiss HNSW candidate search and two-stage search service.
src/faas/                        Local and Alibaba Cloud Function Compute providers.
src/scaling/                     QPS metrics, cost/offload planner, function slot pool.
src/utils/                       Config, logging, and timing helpers.
functions/ann_candidate_search/  Function Compute handler sketch.
scripts/                         Local run and benchmark helpers.
tests/                           Small unit tests for search/scaling behavior.
```

`vectors` and `search` are intentionally separate:

- `vectors` means heavy raw data on the VM.
- `search` means the retrieval algorithm, currently HNSW candidate search plus rerank orchestration.

This avoids the earlier `storage` / `index` split, while still making the data and search responsibilities explicit.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

Force the FaaS path through the local in-process function emulator:

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"query_id": 0, "k": 10, "use_faas": true}'
```
