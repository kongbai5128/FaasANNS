"""构建第一版完整 HNSW 索引。

这个脚本是 data_generator/constrained_kmeans_w_clusters.py 的轻量包装。
第一版不做分区，因此默认 `--k 1`，会在 data/ 下生成 full_hnsw.bin 和 full_hnsw.meta.json。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "data_generator" / "constrained_kmeans_w_clusters.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full HNSW index used by the local server")
    parser.add_argument("--base", default="data/sift100w/sift_base.fvecs")
    parser.add_argument("--out-dir", default="data/index/full")
    parser.add_argument("--topk", type=int, default=1000000)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=80)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(GENERATOR),
        "--src",
        str(ROOT / args.base),
        "--dst",
        str(ROOT / args.out_dir),
        "--topk",
        str(args.topk),
        "--k",
        "1",
        "--m",
        str(args.m),
        "--ef-construction",
        str(args.ef_construction),
        "--ef-search",
        str(args.ef_search),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
