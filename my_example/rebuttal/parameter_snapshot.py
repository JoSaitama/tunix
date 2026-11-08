"""Detached LoRA snapshots: never retain live NNX Variable wrappers.

Flax 0.12.x nnx.state() can reference model Variables. A State is therefore not
by itself an immutable checkpoint for counterfactual parameter updates.
"""

from __future__ import annotations

import numpy as np


class ParameterSnapshot:
    def __init__(self, state):
        import jax
        import jax.numpy as jnp

        self.jax = jax
        pairs, self._treedef = jax.tree_util.tree_flatten_with_path(state)
        if not pairs:
            raise ValueError("cannot snapshot an empty parameter tree")
        self.paths = tuple(str(path) for path, _ in pairs)
        leaves = tuple(leaf for _, leaf in pairs)
        if any(not isinstance(leaf, jax.Array) for leaf in leaves):
            raise TypeError("snapshot requires concrete JAX arrays, not mutable ArrayRefs or tracers")
        # Only an immutable tuple of copied ARRAY VALUES is authoritative.
        # No Variable/State returned by materialize() is retained here or reused.
        self._arrays = tuple(jnp.array(leaf, copy=True) for leaf in leaves)
        jax.block_until_ready(self._arrays)
        self.contract = tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in self._arrays)
        self.bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in self._arrays)

        @jax.jit
        def equal_bits(current, baseline):
            checks = []
            for value, original in zip(current, baseline):
                unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}[value.dtype.itemsize]
                a = jax.lax.bitcast_convert_type(value, unsigned)
                b = jax.lax.bitcast_convert_type(original, unsigned)
                checks.append(jnp.all(a == b))
            return jnp.all(jnp.stack(checks))
        self._equal_bits = equal_bits

    def materialize(self):
        """Fresh State/Variable wrappers around the protected array values."""
        return self._treedef.unflatten(self._arrays)

    def _current(self, state):
        pairs, _ = self.jax.tree_util.tree_flatten_with_path(state)
        paths = tuple(str(path) for path, _ in pairs)
        arrays = tuple(value for _, value in pairs)
        contract = tuple((tuple(value.shape), str(value.dtype)) for value in arrays)
        return paths, arrays, contract

    def matches(self, state):
        paths, arrays, contract = self._current(state)
        if paths != self.paths or contract != self.contract:
            return False
        return bool(self.jax.device_get(self._equal_bits(arrays, self._arrays)))

    def differences(self, state, limit=8):
        """Error-only host diagnostics; do not weaken the bitwise criterion."""
        paths, arrays, contract = self._current(state)
        if paths != self.paths or contract != self.contract:
            return {"same_paths": paths == self.paths, "same_shape_dtype": contract == self.contract,
                    "expected_leaf_count": len(self.paths), "actual_leaf_count": len(paths)}
        changed_leaves, changed_elements, examples = 0, 0, []
        for path, value, original in zip(self.paths, arrays, self._arrays):
            a, b = (np.asarray(item) for item in self.jax.device_get((value, original)))
            dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
            a_bits = np.frombuffer(a.tobytes(), dtype=dtype)
            b_bits = np.frombuffer(b.tobytes(), dtype=dtype)
            count = int(np.count_nonzero(a_bits != b_bits))
            if count:
                changed_leaves += 1
                changed_elements += count
                if len(examples) < limit:
                    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    examples.append({"path": path, "shape": list(a.shape), "dtype": str(a.dtype),
                                     "changed_elements_bitwise": count,
                                     "max_abs_error": float(delta.max()) if np.isfinite(delta).all() else None})
        return {"same_paths": True, "same_shape_dtype": True, "changed_leaves": changed_leaves,
                "changed_elements_bitwise": changed_elements, "examples": examples}
