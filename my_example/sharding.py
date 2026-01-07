from __future__ import annotations

from typing import Optional, Tuple

import jax


def infer_mesh_counts(num_tpus: int) -> Tuple[int, int]:
    if num_tpus == 8:
        return (1, 8)
    if num_tpus == 4:
        return (1, 4)
    if num_tpus == 1:
        return (1, 1)
    raise ValueError(f"Unsupported number of TPUs: {num_tpus}")


def make_mesh(mesh_counts: Optional[Tuple[int, int]] = None):
    num_tpus = len(jax.devices())
    counts = mesh_counts or infer_mesh_counts(num_tpus)
    mesh = jax.make_mesh(
        counts,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * len(counts),
    )
    return mesh, counts
