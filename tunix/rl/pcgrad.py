"""PCGrad Algorithm 1 in the span of the original task gradients."""

import jax
import jax.numpy as jnp


@jax.default_matmul_precision("highest")
def coefficients(gram: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Return mean projected-gradient coefficients and the projection count.

  ``gram[i, j]`` is the inner product of original gradients i and j.
  The returned vector weights those original gradients. Each row starts at
  e_i and is projected against the original references in a random order.
  """
  size = gram.shape[0]
  basis = jnp.eye(size, dtype=gram.dtype)
  references = (jnp.arange(size)[:, None] + jnp.arange(1, size)) % size
  orders = jax.vmap(jax.random.permutation)(
      jax.random.split(key, size), references
  )

  def project(row, order):
    def visit(state, j):
      weights, count = state
      dot = weights @ gram[:, j]
      norm_squared = gram[j, j]
      conflict = (dot < 0) & (norm_squared > 0)
      denominator = jnp.where(norm_squared > 0, norm_squared, 1)
      change = jnp.where(conflict, dot / denominator, 0)
      return (weights.at[j].add(-change), count + conflict), None

    (weights, count), _ = jax.lax.scan(visit, (row, jnp.int32(0)), order)
    return weights, count

  weights, counts = jax.vmap(project)(basis, orders)
  return jnp.mean(weights, axis=0), jnp.sum(counts)
