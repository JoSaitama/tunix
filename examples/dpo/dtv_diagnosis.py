# Copyright 2026.
# Lightweight diagnostics for DTV score decomposition.
#
# This module is intentionally isolated from dpo_trainer.py.
# The trainer should only import and call maybe_log_dtv_decomposition(...)
# after DTV per-sample scores and per-sample gradient norms are available.

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_true(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _to_numpy_1d(x: Any, *, dtype: Optional[Any] = np.float64) -> Optional[np.ndarray]:
    """Convert JAX/NumPy/Python array-like to a 1D NumPy array.

    This function avoids importing JAX as a hard dependency, but if JAX is
    available it calls jax.device_get to materialize arrays on host.
    """
    if x is None:
        return None

    try:
        import jax  # type: ignore

        x = jax.device_get(x)
    except Exception:
        pass

    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)

    if dtype is None:
        return arr

    try:
        return arr.astype(dtype, copy=False)
    except Exception:
        # JAX bfloat16 or object-like arrays may be safer through float32 first.
        return arr.astype(np.float32, copy=False).astype(dtype, copy=False)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def _safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_sum(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.sum(arr))


def _quantile_dict(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    q10, q25, q50, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}_p10": float(q10),
        f"{prefix}_p25": float(q25),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p75": float(q75),
        f"{prefix}_p90": float(q90),
    }


def _append_csv(path: Path, fieldnames: Iterable[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(fieldnames)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_config_once(out_dir: Path) -> None:
    path = out_dir / "diagnostic_config.json"
    if path.exists():
        return

    cfg = {
        "DPO_DTV_PLOT_DIAG": os.environ.get("DPO_DTV_PLOT_DIAG", ""),
        "DPO_DTV_PLOT_DIAG_EVERY": os.environ.get("DPO_DTV_PLOT_DIAG_EVERY", "1"),
        "DPO_DTV_PLOT_DIAG_TOPK": os.environ.get("DPO_DTV_PLOT_DIAG_TOPK", "20"),
        "DPO_DTV_PLOT_DIAG_FULL_SAMPLES": os.environ.get("DPO_DTV_PLOT_DIAG_FULL_SAMPLES", "0"),
        "DPO_DTV_PLOT_DIAG_CORRUPTION_MAP": os.environ.get("DPO_DTV_PLOT_DIAG_CORRUPTION_MAP", ""),
        "notes": (
            "DTV decomposition diagnostics. "
            "self_term = ||g_j||^2; cross_term = g_j^T G_{-j}; "
            "dtv_score = self_term + cross_term; loo_score = cross_term."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


_CORRUPTION_MAP_CACHE: Dict[str, Dict[int, Tuple[bool, str]]] = {}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUE_VALUES


def _load_corruption_map(path_str: str) -> Dict[int, Tuple[bool, str]]:
    """Load optional index-level corruption metadata.

    Supported CSV columns:
      global_window_index or index
      is_corrupted
      corruption_type

    Supported JSON formats:
      1) {"123": {"is_corrupted": true, "corruption_type": "..."}}
      2) [{"global_window_index": 123, "is_corrupted": true, ...}, ...]
    """
    if not path_str:
        return {}

    path = Path(path_str)
    key = str(path.resolve())
    if key in _CORRUPTION_MAP_CACHE:
        return _CORRUPTION_MAP_CACHE[key]

    mapping: Dict[int, Tuple[bool, str]] = {}
    if not path.exists():
        _CORRUPTION_MAP_CACHE[key] = mapping
        return mapping

    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for idx_str, meta in data.items():
                    idx = int(idx_str)
                    if isinstance(meta, dict):
                        mapping[idx] = (
                            _parse_bool(meta.get("is_corrupted", True)),
                            str(meta.get("corruption_type", "")),
                        )
                    else:
                        mapping[idx] = (_parse_bool(meta), "")
            elif isinstance(data, list):
                for row in data:
                    idx = int(row.get("global_window_index", row.get("index")))
                    mapping[idx] = (
                        _parse_bool(row.get("is_corrupted", True)),
                        str(row.get("corruption_type", "")),
                    )
        else:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx_value = row.get("global_window_index", row.get("index"))
                    if idx_value is None or idx_value == "":
                        continue
                    idx = int(idx_value)
                    mapping[idx] = (
                        _parse_bool(row.get("is_corrupted", True)),
                        str(row.get("corruption_type", "")),
                    )
    except Exception:
        # Diagnostics should never crash training.
        mapping = {}

    _CORRUPTION_MAP_CACHE[key] = mapping
    return mapping


def _annotate_corruption(
    global_window_index: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    map_path = os.environ.get("DPO_DTV_PLOT_DIAG_CORRUPTION_MAP", "").strip()
    if not map_path or global_window_index is None:
        return None, None

    mapping = _load_corruption_map(map_path)
    if not mapping:
        return None, None

    is_corrupted = []
    corruption_type = []
    for idx in global_window_index.astype(np.int64):
        corrupted, ctype = mapping.get(int(idx), (False, ""))
        is_corrupted.append(bool(corrupted))
        corruption_type.append(str(ctype))
    return np.asarray(is_corrupted, dtype=bool), np.asarray(corruption_type, dtype=object)


def maybe_log_dtv_decomposition(
    *,
    run_dir: str | Path,
    step: int,
    variant: str,
    method_code: Any,
    dtv_score: Any,
    self_term: Any,
    cross_term: Any = None,
    loo_score: Any = None,
    window_index: Any = None,
    global_window_index: Any = None,
    process_zero: bool = True,
) -> None:
    """Log DTV decomposition diagnostics.

    Required inputs:
      dtv_score: per-sample original DTV score, S_j^DTV = g_j^T G
      self_term: per-sample self term, ||g_j||^2

    Optional:
      cross_term: per-sample cross term, g_j^T G_{-j}
                  If absent, computed as dtv_score - self_term.
      loo_score: per-sample DTV-LOO score.
                 If absent, set to cross_term.
      window_index/global_window_index: sample identifiers for top-k cases.

    This function is safe to call every step. It returns immediately unless
    DPO_DTV_PLOT_DIAG=1/true/yes/on.
    """
    if not process_zero:
        return
    if not _env_true("DPO_DTV_PLOT_DIAG", default=False):
        return

    every = max(1, _env_int("DPO_DTV_PLOT_DIAG_EVERY", 1))
    try:
        step_int = int(step)
    except Exception:
        step_int = 0
    if step_int % every != 0:
        return

    dtv = _to_numpy_1d(dtv_score)
    self_arr = _to_numpy_1d(self_term)
    if dtv is None or self_arr is None:
        return

    n = min(len(dtv), len(self_arr))
    dtv = dtv[:n]
    self_arr = self_arr[:n]

    cross = _to_numpy_1d(cross_term)
    if cross is None:
        cross = dtv - self_arr
    else:
        cross = cross[:n]

    loo = _to_numpy_1d(loo_score)
    if loo is None:
        loo = cross
    else:
        loo = loo[:n]

    win_idx = _to_numpy_1d(window_index, dtype=np.int64)
    if win_idx is None or len(win_idx) < n:
        win_idx = np.arange(n, dtype=np.int64)
    else:
        win_idx = win_idx[:n].astype(np.int64, copy=False)

    glob_idx = _to_numpy_1d(global_window_index, dtype=np.int64)
    if glob_idx is None or len(glob_idx) < n:
        glob_idx = win_idx.copy()
    else:
        glob_idx = glob_idx[:n].astype(np.int64, copy=False)

    finite = _finite_mask(dtv, self_arr, cross, loo)
    n_total = int(n)
    n_finite = int(np.sum(finite))
    if n_finite == 0:
        return

    dtv_f = dtv[finite]
    self_f = self_arr[finite]
    cross_f = cross[finite]
    loo_f = loo[finite]

    kept_dtv = dtv_f >= 0.0
    kept_loo = loo_f >= 0.0

    keep_both = kept_dtv & kept_loo
    dtv_keep_loo_drop = kept_dtv & (~kept_loo)
    dtv_drop_loo_keep = (~kept_dtv) & kept_loo
    drop_both = (~kept_dtv) & (~kept_loo)

    is_corrupted_all, corruption_type_all = _annotate_corruption(glob_idx)
    if is_corrupted_all is not None:
        is_corrupted_f = is_corrupted_all[finite]
        n_corrupted_finite = int(np.sum(is_corrupted_f))
        n_corrupted_dtv_keep_loo_drop = int(np.sum(is_corrupted_f & dtv_keep_loo_drop))
        ratio_corrupted_dtv_keep_loo_drop = (
            n_corrupted_dtv_keep_loo_drop / max(1, n_corrupted_finite)
        )
    else:
        n_corrupted_finite = ""
        n_corrupted_dtv_keep_loo_drop = ""
        ratio_corrupted_dtv_keep_loo_drop = ""

    out_dir = Path(run_dir) / "plot_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_config_once(out_dir)

    summary_fields = [
        "step",
        "variant",
        "method_code",
        "n_total",
        "n_finite",
        "self_mean",
        "cross_mean",
        "dtv_mean",
        "loo_mean",
        "self_sum",
        "cross_sum",
        "dtv_sum",
        "loo_sum",
        "self_p10",
        "self_p25",
        "self_p50",
        "self_p75",
        "self_p90",
        "cross_p10",
        "cross_p25",
        "cross_p50",
        "cross_p75",
        "cross_p90",
        "dtv_p10",
        "dtv_p25",
        "dtv_p50",
        "dtv_p75",
        "dtv_p90",
        "loo_p10",
        "loo_p25",
        "loo_p50",
        "loo_p75",
        "loo_p90",
        "n_keep_both",
        "n_dtv_keep_loo_drop",
        "n_dtv_drop_loo_keep",
        "n_drop_both",
        "ratio_keep_both",
        "ratio_dtv_keep_loo_drop",
        "ratio_dtv_drop_loo_keep",
        "ratio_drop_both",
        "n_corrupted_finite",
        "n_corrupted_dtv_keep_loo_drop",
        "ratio_corrupted_dtv_keep_loo_drop",
    ]

    summary: Dict[str, Any] = {
        "step": step_int,
        "variant": str(variant),
        "method_code": float(method_code) if np.isscalar(method_code) else str(method_code),
        "n_total": n_total,
        "n_finite": n_finite,
        "self_mean": _safe_mean(self_f),
        "cross_mean": _safe_mean(cross_f),
        "dtv_mean": _safe_mean(dtv_f),
        "loo_mean": _safe_mean(loo_f),
        "self_sum": _safe_sum(self_f),
        "cross_sum": _safe_sum(cross_f),
        "dtv_sum": _safe_sum(dtv_f),
        "loo_sum": _safe_sum(loo_f),
        "n_keep_both": int(np.sum(keep_both)),
        "n_dtv_keep_loo_drop": int(np.sum(dtv_keep_loo_drop)),
        "n_dtv_drop_loo_keep": int(np.sum(dtv_drop_loo_keep)),
        "n_drop_both": int(np.sum(drop_both)),
        "ratio_keep_both": float(np.mean(keep_both)),
        "ratio_dtv_keep_loo_drop": float(np.mean(dtv_keep_loo_drop)),
        "ratio_dtv_drop_loo_keep": float(np.mean(dtv_drop_loo_keep)),
        "ratio_drop_both": float(np.mean(drop_both)),
        "n_corrupted_finite": n_corrupted_finite,
        "n_corrupted_dtv_keep_loo_drop": n_corrupted_dtv_keep_loo_drop,
        "ratio_corrupted_dtv_keep_loo_drop": ratio_corrupted_dtv_keep_loo_drop,
    }
    summary.update(_quantile_dict("self", self_f))
    summary.update(_quantile_dict("cross", cross_f))
    summary.update(_quantile_dict("dtv", dtv_f))
    summary.update(_quantile_dict("loo", loo_f))

    _append_csv(out_dir / "dtv_decomposition_summary.csv", summary_fields, summary)

    # Top-k self-protected conflict cases:
    # cross/LOO score < 0 but original DTV score >= 0.
    top_k = max(0, _env_int("DPO_DTV_PLOT_DIAG_TOPK", 20))
    if top_k > 0:
        finite_indices = np.nonzero(finite)[0]
        spc_local = np.nonzero(dtv_keep_loo_drop)[0]
        if spc_local.size > 0:
            spc_global = finite_indices[spc_local]
            # Rank by self term first: dominant self-confirmation examples.
            order = np.argsort(self_arr[spc_global])[::-1]
            chosen_indices = spc_global[order[:top_k]]

            case_fields = [
                "step",
                "variant",
                "method_code",
                "case_type",
                "rank_in_case",
                "window_index",
                "global_window_index",
                "self_term",
                "cross_term",
                "dtv_score",
                "loo_score",
                "kept_dtv",
                "kept_loo",
                "finite",
                "is_corrupted",
                "corruption_type",
            ]

            for rank, idx in enumerate(chosen_indices):
                if is_corrupted_all is not None:
                    is_corr_value: Any = int(bool(is_corrupted_all[idx]))
                    ctype_value: Any = (
                        str(corruption_type_all[idx]) if corruption_type_all is not None else ""
                    )
                else:
                    is_corr_value = ""
                    ctype_value = ""

                row = {
                    "step": step_int,
                    "variant": str(variant),
                    "method_code": float(method_code)
                    if np.isscalar(method_code)
                    else str(method_code),
                    "case_type": "self_protected_conflict",
                    "rank_in_case": int(rank),
                    "window_index": int(win_idx[idx]),
                    "global_window_index": int(glob_idx[idx]),
                    "self_term": float(self_arr[idx]),
                    "cross_term": float(cross[idx]),
                    "dtv_score": float(dtv[idx]),
                    "loo_score": float(loo[idx]),
                    "kept_dtv": int(dtv[idx] >= 0.0),
                    "kept_loo": int(loo[idx] >= 0.0),
                    "finite": int(finite[idx]),
                    "is_corrupted": is_corr_value,
                    "corruption_type": ctype_value,
                }
                _append_csv(out_dir / "dtv_decomposition_cases.csv", case_fields, row)

    # Optional full sample dump for later density/heatmap figures.
    # Off by default to keep training logs small.
    if _env_true("DPO_DTV_PLOT_DIAG_FULL_SAMPLES", default=False):
        sample_fields = [
            "step",
            "variant",
            "method_code",
            "window_index",
            "global_window_index",
            "self_term",
            "cross_term",
            "dtv_score",
            "loo_score",
            "kept_dtv",
            "kept_loo",
            "finite",
            "is_corrupted",
            "corruption_type",
        ]
        for idx in range(n):
            if is_corrupted_all is not None:
                is_corr_value = int(bool(is_corrupted_all[idx]))
                ctype_value = (
                    str(corruption_type_all[idx]) if corruption_type_all is not None else ""
                )
            else:
                is_corr_value = ""
                ctype_value = ""

            row = {
                "step": step_int,
                "variant": str(variant),
                "method_code": float(method_code) if np.isscalar(method_code) else str(method_code),
                "window_index": int(win_idx[idx]),
                "global_window_index": int(glob_idx[idx]),
                "self_term": float(self_arr[idx]) if np.isfinite(self_arr[idx]) else "",
                "cross_term": float(cross[idx]) if np.isfinite(cross[idx]) else "",
                "dtv_score": float(dtv[idx]) if np.isfinite(dtv[idx]) else "",
                "loo_score": float(loo[idx]) if np.isfinite(loo[idx]) else "",
                "kept_dtv": int(dtv[idx] >= 0.0) if np.isfinite(dtv[idx]) else "",
                "kept_loo": int(loo[idx] >= 0.0) if np.isfinite(loo[idx]) else "",
                "finite": int(finite[idx]),
                "is_corrupted": is_corr_value,
                "corruption_type": ctype_value,
            }
            _append_csv(out_dir / "dtv_decomposition_samples.csv", sample_fields, row)
