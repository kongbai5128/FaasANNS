"""fvecs 数据读取工具。

SIFT 数据集常用 fvecs 格式，每条记录由一个 int32 维度头和若干 float32 分量组成。
本模块把 fvecs 文件加载成连续的 NumPy float32 矩阵，供 VM 侧 VectorStore 使用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_fvecs(path: str | Path, dimension: int | None = None, max_vectors: int | None = None) -> np.ndarray:
    file_path = Path(path)
    raw = np.fromfile(file_path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty fvecs file: {file_path}")

    first_dim = int(raw[0])
    dim = dimension or first_dim
    record_width = dim + 1
    if raw.size % record_width != 0:
        raise ValueError(f"{file_path} does not look like fvecs with dimension {dim}")

    records = raw.reshape(-1, record_width)
    dims = records[:, 0]
    if not np.all(dims == dim):
        raise ValueError(f"dimension mismatch in {file_path}: expected {dim}")

    vectors = records[:, 1:].view(np.float32)
    if max_vectors is not None:
        vectors = vectors[:max_vectors]
    return np.ascontiguousarray(vectors)
