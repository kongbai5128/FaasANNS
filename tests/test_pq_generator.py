"""PQ generator tests."""

from __future__ import annotations

import numpy as np

from data_generator.build_pq_index import build_pq


def test_build_pq_outputs_codebooks_and_codes() -> None:
    vectors = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [9.0, 9.0, 9.0, 9.0],
        ],
        dtype="float32",
    )

    codebooks, codes = build_pq(
        vectors,
        subspace_count=2,
        codebook_size=2,
        train_size=4,
        iterations=2,
        seed=0,
        batch_size=2,
    )

    assert codebooks.shape == (2, 2, 2)
    assert codes.shape == (4, 2)
    assert codes.dtype == np.uint8
