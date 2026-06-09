"""ivecs 文件读写工具。

SIFT groundtruth 常用 ivecs 格式，每条记录由一个 int32 维度头和若干 int32 id 组成。
请求评测脚本会读取 sift_groundtruth.ivecs 来计算 Recall@K；如果后续需要针对新的数据子集
重建正确结果，也可以用这里的 write_ivecs 保存。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ivecs(path: str | Path, max_vectors: int | None = None) -> np.ndarray:
    file_path = Path(path)
    raw = np.fromfile(file_path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty ivecs file: {file_path}")

    dim = int(raw[0])
    record_width = dim + 1
    if raw.size % record_width != 0:
        raise ValueError(f"{file_path} does not look like ivecs with dimension {dim}")

    records = raw.reshape(-1, record_width)
    if not np.all(records[:, 0] == dim):
        raise ValueError(f"dimension header mismatch in {file_path}")

    vectors = records[:, 1:]
    if max_vectors is not None:
        vectors = vectors[:max_vectors]
    return np.ascontiguousarray(vectors.astype(np.int32, copy=False))


def write_ivecs(path: str | Path, values: np.ndarray) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.int32)
    if values.ndim != 2:
        raise ValueError("ivecs values must be a 2D array")

    dim = values.shape[1]
    header = np.full((values.shape[0], 1), dim, dtype=np.int32)
    records = np.concatenate([header, values], axis=1)
    records.tofile(file_path)
