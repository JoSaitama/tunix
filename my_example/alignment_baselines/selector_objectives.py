"""Method-specific on-policy selector objectives; never used for real updates.

Both gradients are loss gradients (negative ascent gradients). Flipping every
candidate and reference consistently leaves the alignment scores unchanged.
Imports are lazy so the mathematical contract can also be tested with NumPy.
"""

SELECTOR_VERSION = "alignment_selector_v2"
PROJECTION_VERSION = "signed_feature_hash_v2"


def selector_metadata(method, training_beta):
    if method not in {"learnalign", "gradalign"}:
        raise ValueError(f"unsupported selector method: {method}")
    return {
        "selector_version": SELECTOR_VERSION,
        "projection": "deterministic_signed_feature_hash",
        "projection_version": PROJECTION_VERSION,
        "projection_sign_stream": "separate_seed_from_bucket_stream",
        "projection_guarantee": "engineering_approximation_no_JL_guarantee_claimed",
        "gradient_space": "actor_trainable_LoRA_parameters",
        "gradient_orientation": "negative_ascent_loss_gradient",
        "selector_beta": float(training_beta) if method == "learnalign" else 0.0,
        "selector_token_reduction": "mean" if method == "learnalign" else "sum",
        "selector_objective": (
            "on_policy_advantage_plus_configured_low_variance_KL"
            if method == "learnalign" else "on_policy_advantage_log_probability"
        ),
    }


def selector_loss_from_logps(logps, mask, advantage, *, method, beta,
                             reference_logps=None, array_module=None):
    """One response's negative Eq.7 objective / GradAlign PG surrogate.

    LearnAlign loss derivative is -(A + beta*(ref/policy - 1))/length,
    exactly the coefficient in Eq.7 under the current on-policy snapshot.
    GradAlign uses -A*sum_t(log pi_t), without KL or length normalization.
    """
    if array_module is None:
        import jax.numpy as array_module
    xp = array_module
    if method not in {"learnalign", "gradalign"}:
        raise ValueError(f"unsupported selector method: {method}")
    logps = xp.asarray(logps, dtype=xp.float32)
    mask = xp.asarray(mask, dtype=xp.float32)
    # Invalid tokens must not produce exp overflow or influence the gradient.
    valid_logps = xp.where(mask > 0, logps, 0.0)
    token_loss = -xp.asarray(advantage, dtype=xp.float32) * valid_logps
    if method == "learnalign" and beta != 0.0:
        if reference_logps is None:
            raise ValueError("LearnAlign nonzero beta requires reference logps")
        ref = xp.where(mask > 0, xp.asarray(reference_logps, dtype=xp.float32), 0.0)
        difference = ref - valid_logps
        token_loss = token_loss + beta * (xp.exp(difference) - difference - 1.0)
    result = xp.sum(token_loss * mask)
    if method == "learnalign":
        result = result / xp.maximum(xp.sum(mask), 1.0)
    return result


def selector_loss(model, example, *, method, beta, pad_id, eos_id):
    import jax
    import jax.numpy as jnp
    from tunix.rl import common

    logps = common.compute_per_token_logps(
        model, prompt_tokens=jnp.atleast_2d(example.prompt_ids),
        completion_tokens=jnp.atleast_2d(example.completion_ids),
        pad_id=pad_id, eos_id=eos_id, stop_gradient=False,
        return_logits=False,
    )
    reference = example.ref_per_token_logps
    if reference is not None:
        reference = jax.lax.stop_gradient(jnp.atleast_2d(reference))
    return selector_loss_from_logps(
        logps, jnp.atleast_2d(example.completion_mask), example.advantages,
        method=method, beta=beta, reference_logps=reference,
    )
