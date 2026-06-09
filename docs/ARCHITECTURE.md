# Architecture

The first version keeps the shape small and experimental:

```text
POST /search
  -> resolve query vector
  -> read current QPS
  -> planner chooses local or FaaS candidate search
  -> HNSW candidate search returns candidate ids
  -> VM vector store performs exact rerank
  -> return top-k
```

## Why `vectors` and `search` are separate

`vectors` owns raw vector data. In the target design this state is heavy, persistent, and should stay on the VM.

`search` owns retrieval logic. In this first version it builds one Faiss HNSW index locally. Later, the HNSW graph or compressed candidate index can be moved into Function Compute while the VM still keeps raw vectors for rerank.

## Why there is no partition module now

The first version does not need shard management. It is better to first prove:

- HNSW candidate search works;
- exact rerank works;
- query pressure can trigger FaaS offload;
- queue and prewarm policy behave as expected.

Once that is stable, shard placement and graph-index distribution can be added under `search` and `faas`.
