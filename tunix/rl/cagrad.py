"""Solve the CAGrad primal for one four-completion GRPO prompt group."""

import jax
import jax.numpy as jnp


# Rank tolerance for the pseudoinverse of the active constraints' Gram matrix.
RANK_RTOL = 1e-6


@jax.default_matmul_precision("highest")
def coefficients(gram: jax.Array, c: float) -> jax.Array:
  """Weight the original gradients to maximize the worst task improvement.

  Solve Eq. (3) of CAGrad, max_d min_i <g_i, d> with
  ||d - mean(g)|| <= c * ||mean(g)||. Enumerate the 15 nonempty possible
  active sets for G=4. On each set, the active inner products are equal.
  The best direction in that subspace follows from projecting the ball
  center and the active gradient into it. Score each feasible candidate
  against all four tasks. This also handles singular/zero gradients without
  dividing by the norm of a zero dual solution.
  """
  if gram.shape != (4, 4):
    raise ValueError("CAGrad expects a four-completion Gram matrix.")
  mean = jnp.full((4,), 0.25, dtype=gram.dtype)
  if c == 0:
    return mean

  # Positive rescaling preserves both the optimum and the radius constraint.
  scale = jnp.max(jnp.diag(gram))
  gram = gram / jnp.where(scale > 0, scale, 1)
  radius_squared = c**2 * jnp.maximum(mean @ gram @ mean, 0)
  basis = jnp.eye(4, dtype=gram.dtype)
  members = (jnp.arange(1, 16)[:, None] & (1 << jnp.arange(4))) != 0

  def candidate(active):
    # Any active gradient gives the same objective on this face. Starting
    # with the smallest one avoids subtracting two large nearly equal
    # vectors when the feasible improving component is small.
    anchor = basis[jnp.argmin(jnp.where(active, jnp.diag(gram), jnp.inf))]
    # A row represents the difference g_i - g_anchor for each active i.
    differences = (basis - anchor) * active[:, None]
    products = differences @ gram
    inverse = jnp.linalg.pinv(
        products @ differences.T, rtol=RANK_RTOL, hermitian=True
    )

    def project(weights):
      return weights - (inverse @ (products @ weights)) @ differences

    center = project(mean)
    axis = project(anchor)
    offset = center - mean
    offset_squared = jnp.maximum(offset @ gram @ offset, 0)
    axis_squared = jnp.maximum(axis @ gram @ axis, 0)
    # A flat objective on this subspace uses the center. Active supersets
    # also cover optima where additional constraints limit a flat face.
    nonflat = axis_squared > RANK_RTOL**2
    norm = jnp.sqrt(jnp.where(nonflat, axis_squared, 1))
    distance = jnp.sqrt(jnp.maximum(radius_squared - offset_squared, 0))
    weights = center + jnp.where(nonflat, distance / norm, 0) * axis
    # Clip numerical drift (and any infeasible face center) back to the ball.
    delta = weights - mean
    delta_squared = jnp.maximum(delta @ gram @ delta, 0)
    fraction = jnp.sqrt(
        jnp.minimum(
            radius_squared / jnp.where(delta_squared > 0, delta_squared, 1), 1
        )
    )
    return mean + fraction * delta

  candidates = jnp.concatenate((mean[None, :], jax.vmap(candidate)(members)))
  worst_improvements = jnp.min(candidates @ gram, axis=1)
  # The mean is a feasible candidate and wins exact ties.
  return candidates[jnp.argmax(worst_improvements)]
