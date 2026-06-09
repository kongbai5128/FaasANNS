"""生成 FaasANN 本地 HNSW 索引。

这个脚本保留了原来的 constrained k-means 分区能力，但第一版推荐使用 `--k 1`：
直接从 fvecs 读取向量，构建一个完整 hnswlib 索引并输出到 data/full_hnsw.bin。

第一版服务端不做分区，因此 `--k 1` 模式不会导入 k_means_constrained 或 matplotlib。
只有后续需要分区实验时，`--k > 1` 才会走 constrained k-means 并生成 partition_*.bin。
.venv/bin/python scripts/build_index.py --base data/sift100w/sift_base.fvecs --out-dir data/index/full --topk 1000000 --m 32 --ef-construction 200 --ef-search 80
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import hnswlib
import numpy as np


def read_fvecs(filename: str | Path, count: int | None = None) -> np.ndarray:
    """读取 SIFT/GIST 标准 .fvecs 文件。

    格式为重复的 `[dim:int32][vector:dim*float32]`。
    返回连续的 float32 矩阵，shape 为 `(n, dim)`。
    """
    filename = Path(filename)
    print(f"Reading .fvecs file: {filename}...")
    with filename.open("rb") as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError(f"file is empty: {filename}")
        dim = int.from_bytes(dim_bytes, "little")

        record_bytes = 4 + dim * 4
        if count is None:
            f.seek(0, os.SEEK_END)
            count = f.tell() // record_bytes

        f.seek(0)
        data = np.fromfile(f, dtype=np.int32, count=count * (dim + 1))

    if data.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    data = data.reshape(-1, dim + 1)
    if not np.all(data[:, 0] == dim):
        raise ValueError(f"dimension header mismatch in {filename}")
    vectors = data[:, 1:].view(np.float32)
    vectors = np.ascontiguousarray(vectors)
    print(f"Read complete. Shape: {vectors.shape}")
    return vectors


def build_hnsw_index(
    vectors: np.ndarray,
    output_path: str | Path,
    ids: np.ndarray | None = None,
    space: str = "l2",
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 80,
) -> None:
    """构建并保存 hnswlib 索引。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ids is None:
        ids = np.arange(vectors.shape[0], dtype=np.int64)

    print(
        f"Building HNSW index: vectors={vectors.shape[0]}, dim={vectors.shape[1]}, "
        f"space={space}, M={m}, ef_construction={ef_construction}"
    )
    index = hnswlib.Index(space=space, dim=vectors.shape[1])
    index.init_index(max_elements=vectors.shape[0], ef_construction=ef_construction, M=m)
    index.add_items(vectors, ids=ids)
    index.set_ef(ef_search)
    index.save_index(str(output_path))
    print(f"Saved HNSW index to {output_path}")


def save_index_meta(
    output_path: str | Path,
    *,
    source: str,
    index_file: str,
    vector_count: int,
    dimension: int,
    space: str,
    m: int,
    ef_construction: int,
    ef_search: int,
    partitioned: bool,
    partitions: int,
) -> None:
    meta = {
        "source": source,
        "index_file": index_file,
        "vector_count": vector_count,
        "dimension": dimension,
        "space": space,
        "m": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "partitioned": partitioned,
        "partitions": partitions,
    }
    Path(output_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved index metadata to {output_path}")


def save_centroids_binary(centroids: np.ndarray, sizes: list[int], output_path: str | Path) -> None:
    """保存 centroids 为旧实验使用的二进制格式。"""
    n_clusters, dim = centroids.shape
    output_path = Path(output_path)
    print(f"Saving centroids to {output_path}")
    with output_path.open("wb") as f:
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<I", n_clusters))
        for size in sizes:
            f.write(struct.pack("<I", int(size)))
        f.write(centroids.astype(np.float32).tobytes())
    print(f"Saved centroids.bin: dim={dim}, n_clusters={n_clusters}")


def build_full_index(args: argparse.Namespace) -> None:
    vectors = read_fvecs(args.src, count=args.topk)
    output_dir = Path(args.dst)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / args.index_name
    build_hnsw_index(
        vectors,
        index_path,
        space=args.space,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    save_index_meta(
        output_dir / "full_hnsw.meta.json",
        source=str(Path(args.src).resolve()),
        index_file=index_path.name,
        vector_count=vectors.shape[0],
        dimension=vectors.shape[1],
        space=args.space,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        partitioned=False,
        partitions=1,
    )


def build_partitioned_indexes(args: argparse.Namespace) -> None:
    """保留后续分区实验路径：constrained k-means + 每个分区一个 HNSW。"""
    from k_means_constrained import KMeansConstrained
    import matplotlib.pyplot as plt

    vectors = read_fvecs(args.src, count=args.topk)
    output_dir = Path(args.dst)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_size = vectors.shape[0]
    n_clusters = args.k
    avg_size = db_size // n_clusters
    size_min = int(avg_size * 0.9)
    size_max = int(avg_size * 1.1)

    print(f"Clustering into {n_clusters} partitions.")
    print(f"Constraint per partition: min={size_min}, max={size_max}")

    clf = KMeansConstrained(
        n_init=10,
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=0,
        init="k-means++",
        max_iter=100,
        tol=0.0001,
        verbose=True,
    )

    print("Fitting KMeans...")
    clf.fit(vectors)
    labels = clf.labels_
    centers = clf.cluster_centers_

    unique, counts = np.unique(labels, return_counts=True)
    sizes = []
    print("\nPartition Statistics:")
    for cluster_id, count in zip(unique, counts):
        print(f"Partition {cluster_id}: {count} vectors")
        sizes.append(int(count))

    plt.figure()
    plt.bar([f"P{i}" for i in unique], counts)
    plt.title("Data Distribution per Partition")
    plt.savefig(output_dir / "partition_distribution.png")

    save_centroids_binary(centers, sizes, output_dir / "centroids.bin")
    np.savetxt(output_dir / "labels.csv", labels.astype(int), delimiter=",", fmt="%d")
    print("Saved labels.csv")

    for i in range(n_clusters):
        indices = np.where(labels == i)[0].astype(np.int64)
        partition_vecs = vectors[indices]
        if len(partition_vecs) == 0:
            continue
        index_path = output_dir / f"partition_{i}.bin"
        build_hnsw_index(
            partition_vecs,
            index_path,
            ids=indices,
            space=args.space,
            m=args.m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )

    save_index_meta(
        output_dir / "partitioned_hnsw.meta.json",
        source=str(Path(args.src).resolve()),
        index_file="partition_*.bin",
        vector_count=vectors.shape[0],
        dimension=vectors.shape[1],
        space=args.space,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        partitioned=True,
        partitions=n_clusters,
    )


def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full or partitioned hnswlib indexes for FaasANN")
    parser.add_argument("--src", required=True, help="input .fvecs file")
    parser.add_argument("--dst", required=True, help="output directory")
    parser.add_argument("--topk", type=int, default=1000000, help="number of vectors to index")
    parser.add_argument("--k", type=int, default=1, help="number of partitions; k=1 builds one full index")
    parser.add_argument("--index-name", default="full_hnsw.bin", help="output file name for k=1")
    parser.add_argument("--space", default="l2", choices=["l2", "ip", "cosine"], help="hnswlib distance space")
    parser.add_argument("--m", type=int, default=32, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction")
    parser.add_argument("--ef-search", type=int, default=80, help="HNSW ef search stored in metadata")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = process_args()
    if parsed.k <= 1:
        build_full_index(parsed)
    else:
        build_partitioned_indexes(parsed)
