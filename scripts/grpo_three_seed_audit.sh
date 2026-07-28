cd ~/Project/tunix
source .venv_jax081/bin/activate

mkdir -p analysis

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="analysis/six_methods_three_seeds_audit_${STAMP}.output"
CSV="analysis/six_methods_three_seeds_audit_${STAMP}.csv"

python - "$PWD/logs" "$CSV" >"$OUT" 2>&1 <<'PY'
import csv
import json
import math
import re
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


LOGS_ROOT = Path(sys.argv[1])
CSV_PATH = Path(sys.argv[2])
SEEDS = (0, 5, 21)
BATCH_SIZE = 16
EXCLUDED = ("rank_noise0p2", "keep75", "policy_only")

METHODS = OrderedDict([
    (
        "Baseline",
        {
            0: (
                "gsm8k_baseline_full_*",
                "gsm8k_baseline_seed0_full_*",
            ),
            5: ("gsm8k_baseline_seed5_full_*",),
            21: ("gsm8k_baseline_seed21_full_*",),
        },
    ),
    (
        "L2 outlier",
        {
            0: (
                "gsm8k_dtv_outlier_l2_full_*",
                "gsm8k_dtv_outlier_l2_seed0_full_*",
            ),
            5: ("gsm8k_dtv_outlier_l2_seed5_full_*",),
            21: ("gsm8k_dtv_outlier_l2_seed21_full_*",),
        },
    ),
    (
        "DTV-Policy batch",
        {
            0: (
                "gsm8k_dtv_selfinf_batch_policy_seed0_full_*",
                "gsm8k_dtv_selfinf_batch_policy_full_*",
            ),
            5: ("gsm8k_dtv_selfinf_batch_policy_seed5_full_*",),
            21: ("gsm8k_dtv_selfinf_batch_policy_seed21_full_*",),
        },
    ),
    (
        "DTV-Policy group",
        {
            0: (
                "gsm8k_dtv_selfinf_group_policy_seed0_full_*",
                "gsm8k_dtv_selfinf_group_policy_full_*",
            ),
            5: ("gsm8k_dtv_selfinf_group_policy_seed5_full_*",),
            21: ("gsm8k_dtv_selfinf_group_policy_seed21_full_*",),
        },
    ),
    (
        "Policy-LOO batch",
        {
            0: (
                "gsm8k_dtv_selfinf_batch_loo_policy_seed0_full_*",
            ),
            5: (
                "gsm8k_dtv_selfinf_batch_loo_policy_seed5_full_*",
            ),
            21: (
                "gsm8k_dtv_selfinf_batch_loo_policy_seed21_full_*",
            ),
        },
    ),
    (
        "Policy-LOO group",
        {
            0: (
                "gsm8k_dtv_selfinf_group_loo_policy_seed0_full_*",
            ),
            5: (
                "gsm8k_dtv_selfinf_group_loo_policy_seed5_full_*",
            ),
            21: (
                "gsm8k_dtv_selfinf_group_loo_policy_seed21_full_*",
            ),
        },
    ),
])

METRIC_PATTERN = re.compile(
    r"(pre-train|post-train):\s*"
    r"num_correct=(\d+),\s*total=(\d+),\s*"
    r"accuracy=([-+0-9.eE]+)%,\s*"
    r"partial_accuracy=([-+0-9.eE]+)%,\s*"
    r"format_accuracy=([-+0-9.eE]+)%"
)


def finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values):
    values = [
        value for value in values
        if finite_number(value) is not None
    ]
    return statistics.fmean(values) if values else None


def sample_std(values):
    values = [
        value for value in values
        if finite_number(value) is not None
    ]
    return statistics.stdev(values) if len(values) >= 2 else None


def percentile(values, probability):
    values = sorted(
        value for value in (
            finite_number(item) for item in values
        )
        if value is not None
    )
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def describe(values):
    values = [
        value for value in (
            finite_number(item) for item in values
        )
        if value is not None
    ]
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def fmt(value, digits=4):
    if value is None:
        return "MISSING"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def discover(patterns):
    candidates = []
    for pattern in patterns:
        for path in LOGS_ROOT.glob(pattern):
            lowered = path.name.lower()
            if (
                path.is_dir()
                and not any(token in lowered for token in EXCLUDED)
            ):
                candidates.append(path)
    candidates = sorted(
        set(candidates),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return (candidates[0] if candidates else None), candidates


def find_stdout(root):
    if root is None:
        return None
    direct = root / "nohup.log"
    if direct.is_file():
        return direct
    matches = sorted(
        root.glob("results/*__stdout.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def read_exit_code(root):
    if root is None:
        return None
    path = root / "exit_code"
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def parse_eval(log_path):
    phases = {}
    if log_path is None:
        return phases
    text = log_path.read_text(errors="replace")
    for phase, correct, total, accuracy, partial, format_accuracy in (
        METRIC_PATTERN.findall(text)
    ):
        phases[phase] = {
            "correct": int(correct),
            "total": int(total),
            "accuracy": float(accuracy),
            "partial": float(partial),
            "format": float(format_accuracy),
        }
    return phases


def load_tensorboard(root):
    if root is None:
        return None
    tb_root = root / "tensorboard"
    if not tb_root.is_dir():
        return None
    accumulator = EventAccumulator(
        str(tb_root),
        size_guidance={"scalars": 0},
    )
    try:
        accumulator.Reload()
    except Exception as error:
        print(f"WARNING: TensorBoard load failed for {root}: {error}")
        return None
    return accumulator


def scalar_values(accumulator, tag):
    if accumulator is None:
        return []
    if tag not in set(accumulator.Tags().get("scalars", [])):
        return []
    return [
        float(event.value)
        for event in accumulator.Scalars(tag)
    ]


def find_selection(root):
    if root is None:
        return None
    matches = sorted(
        root.glob("results/*__selection.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        matches = sorted(
            root.rglob("*__selection.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    return matches[0] if matches else None


def load_selection_summary(path):
    if path is None:
        return None

    records = []
    with path.open(errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error

    loo_scores = []
    standard_self = []
    standard_cross = []
    standard_score = []
    actual_drop = []
    threshold_drop = []
    retained_by_cap = []
    objectives = set()
    applications = set()

    for record in records:
        scores = [float(value) for value in record["loo_scores"]]
        final_mask = [
            bool(value) for value in record["loo_final_mask"]
        ]
        threshold_mask = [
            bool(value) for value in record["loo_threshold_mask"]
        ]
        cap_mask = [
            bool(value) for value in record.get(
                "loo_retained_by_cap_mask",
                [False] * len(scores),
            )
        ]

        loo_scores.extend(scores)
        standard_self.extend(
            float(value)
            for value in record["loo_standard_self_term"]
        )
        standard_cross.extend(
            float(value)
            for value in record["loo_standard_cross_term"]
        )
        standard_score.extend(
            float(value)
            for value in record["loo_standard_score"]
        )
        actual_drop.extend(not value for value in final_mask)
        threshold_drop.extend(not value for value in threshold_mask)
        retained_by_cap.extend(cap_mask)
        objectives.add(record.get("score_objective", "total"))
        applications.add(record.get("mask_application", "full"))

    total = len(actual_drop)
    filtered = sum(actual_drop)
    threshold_filtered = sum(threshold_drop)
    rescued = sum(retained_by_cap)

    return {
        "path": path,
        "records": len(records),
        "total": total,
        "filtered": filtered,
        "filter_fraction": filtered / total if total else None,
        "kept_fraction": 1 - filtered / total if total else None,
        "threshold_filtered": threshold_filtered,
        "threshold_filter_fraction": (
            threshold_filtered / total if total else None
        ),
        "rescued_by_cap": rescued,
        "cap_records": sum(
            bool(record.get("loo_retention_cap_triggered", False))
            for record in records
        ),
        "negative": sum(value < 0 for value in loo_scores),
        "zero": sum(value == 0 for value in loo_scores),
        "positive": sum(value > 0 for value in loo_scores),
        "objectives": sorted(objectives),
        "applications": sorted(applications),
        "loo_score": describe(loo_scores),
        "standard_self": describe(standard_self),
        "standard_cross": describe(standard_cross),
        "standard_score": describe(standard_score),
        "filtered_loo_score": describe([
            value for value, dropped in zip(loo_scores, actual_drop)
            if dropped
        ]),
        "kept_loo_score": describe([
            value for value, dropped in zip(loo_scores, actual_drop)
            if not dropped
        ]),
    }


def build_run(method, seed, root, candidates):
    log_path = find_stdout(root)
    evaluations = parse_eval(log_path)
    pre = evaluations.get("pre-train", {})
    post = evaluations.get("post-train", {})
    accumulator = load_tensorboard(root)
    skipped = scalar_values(
        accumulator,
        "actor/train/skipped_samples",
    )
    kept_fraction_events = scalar_values(
        accumulator,
        "actor/train/self_inf_kept_fraction",
    )
    dot_mean = scalar_values(
        accumulator,
        "actor/train/self_inf_dot_mean",
    )
    dot_std = scalar_values(
        accumulator,
        "actor/train/self_inf_dot_std",
    )
    grad_norm_mean = scalar_values(
        accumulator,
        "actor/train/grad_norm_mean",
    )
    grad_norm_std = scalar_values(
        accumulator,
        "actor/train/grad_norm_std",
    )
    selection = load_selection_summary(find_selection(root))

    if selection is not None:
        filter_fraction = selection["filter_fraction"]
        filter_source = "selection_jsonl"
        filtered_total = selection["filtered"]
        filter_denominator = selection["total"]
    elif skipped:
        filtered_total = sum(skipped)
        filter_denominator = len(skipped) * BATCH_SIZE
        filter_fraction = (
            filtered_total / filter_denominator
            if filter_denominator
            else None
        )
        filter_source = "tensorboard"
    elif method == "Baseline":
        filtered_total = 0
        filter_denominator = None
        filter_fraction = 0.0
        filter_source = "baseline"
    else:
        filtered_total = None
        filter_denominator = None
        filter_fraction = None
        filter_source = "missing"

    return {
        "method": method,
        "seed": seed,
        "root": root,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "log_path": log_path,
        "exit_code": read_exit_code(root),
        "pre_correct": pre.get("correct"),
        "pre_total": pre.get("total"),
        "pre_accuracy": pre.get("accuracy"),
        "pre_partial": pre.get("partial"),
        "pre_format": pre.get("format"),
        "post_correct": post.get("correct"),
        "post_total": post.get("total"),
        "post_accuracy": post.get("accuracy"),
        "post_partial": post.get("partial"),
        "post_format": post.get("format"),
        "accuracy_delta": (
            post["accuracy"] - pre["accuracy"]
            if "accuracy" in post and "accuracy" in pre
            else None
        ),
        "filter_fraction": filter_fraction,
        "kept_fraction": (
            1 - filter_fraction
            if filter_fraction is not None
            else None
        ),
        "filtered_total": filtered_total,
        "filter_denominator": filter_denominator,
        "filter_source": filter_source,
        "filter_events": len(skipped),
        "steps_with_filtering": sum(value > 0 for value in skipped),
        "skipped_per_step": describe(skipped),
        "kept_fraction_events": describe(kept_fraction_events),
        "dot_mean": describe(dot_mean),
        "dot_std": describe(dot_std),
        "grad_norm_mean": describe(grad_norm_mean),
        "grad_norm_std": describe(grad_norm_std),
        "selection": selection,
    }


runs = []

print("=" * 120)
print("SIX-METHOD × THREE-SEED CLEAN GSM8K AUDIT")
print("=" * 120)
print("logs root:", LOGS_ROOT)
print("seeds:", SEEDS)
print("excluded directory tokens:", EXCLUDED)
print()

print("=" * 120)
print("DISCOVERED RUN DIRECTORIES")
print("=" * 120)

for method, seed_patterns in METHODS.items():
    for seed in SEEDS:
        root, candidates = discover(seed_patterns[seed])
        run = build_run(method, seed, root, candidates)
        runs.append(run)

        print(f"{method:24s} seed={seed:2d}: {root or 'MISSING'}")
        if len(candidates) > 1:
            print(
                "  WARNING: multiple clean candidates; newest selected:"
            )
            for candidate in candidates:
                print("   -", candidate)

print()

print("=" * 120)
print("PER-RUN TRAINING AND FILTERING SUMMARY")
print("=" * 120)
header = (
    f"{'Method':24s} {'Seed':>4s} {'Exit':>5s} "
    f"{'Pre':>9s} {'Post':>9s} {'Delta':>9s} "
    f"{'Correct':>11s} {'Partial':>9s} {'Format':>9s} "
    f"{'Filter%':>9s} {'Keep%':>9s} {'Source':>16s}"
)
print(header)
print("-" * len(header))

for run in runs:
    correct = (
        f"{run['post_correct']}/{run['post_total']}"
        if run["post_correct"] is not None
        else "MISSING"
    )
    print(
        f"{run['method']:24s} "
        f"{run['seed']:4d} "
        f"{str(run['exit_code']):>5s} "
        f"{fmt(run['pre_accuracy']):>9s} "
        f"{fmt(run['post_accuracy']):>9s} "
        f"{fmt(run['accuracy_delta']):>9s} "
        f"{correct:>11s} "
        f"{fmt(run['post_partial']):>9s} "
        f"{fmt(run['post_format']):>9s} "
        f"{fmt(
            run['filter_fraction'] * 100
            if run['filter_fraction'] is not None
            else None
        ):>9s} "
        f"{fmt(
            run['kept_fraction'] * 100
            if run['kept_fraction'] is not None
            else None
        ):>9s} "
        f"{run['filter_source']:>16s}"
    )

print()
print("=" * 120)
print("PER-METHOD THREE-SEED AGGREGATE (sample SD, ddof=1)")
print("=" * 120)
summary_header = (
    f"{'Method':24s} {'N':>2s} "
    f"{'Post mean':>10s} {'Post SD':>9s} "
    f"{'Delta mean':>11s} {'Partial':>9s} {'Format':>9s} "
    f"{'Filter%':>9s} {'Filter SD':>9s}"
)
print(summary_header)
print("-" * len(summary_header))

method_runs = {}
for method in METHODS:
    selected = [run for run in runs if run["method"] == method]
    method_runs[method] = {run["seed"]: run for run in selected}
    post_values = [run["post_accuracy"] for run in selected]
    delta_values = [run["accuracy_delta"] for run in selected]
    partial_values = [run["post_partial"] for run in selected]
    format_values = [run["post_format"] for run in selected]
    filter_values = [
        run["filter_fraction"] * 100
        if run["filter_fraction"] is not None
        else None
        for run in selected
    ]
    completed = sum(
        run["post_accuracy"] is not None
        and run["exit_code"] in (0, None)
        for run in selected
    )

    print(
        f"{method:24s} {completed:2d} "
        f"{fmt(mean(post_values)):>10s} "
        f"{fmt(sample_std(post_values)):>9s} "
        f"{fmt(mean(delta_values)):>11s} "
        f"{fmt(mean(partial_values)):>9s} "
        f"{fmt(mean(format_values)):>9s} "
        f"{fmt(mean(filter_values)):>9s} "
        f"{fmt(sample_std(filter_values)):>9s}"
    )

print()
print("=" * 120)
print("MATCHED-SEED POST-ACCURACY DIFFERENCES")
print("=" * 120)

for reference in ("Baseline", "L2 outlier"):
    print()
    print(f"Reference: {reference}")
    print(
        f"{'Method':24s} "
        + " ".join(f"{'seed'+str(seed):>11s}" for seed in SEEDS)
        + f" {'Mean':>11s} {'SD':>11s}"
    )
    for method in METHODS:
        if method == reference:
            continue
        differences = []
        displayed = []
        for seed in SEEDS:
            current = method_runs[method][seed]["post_accuracy"]
            baseline = method_runs[reference][seed]["post_accuracy"]
            difference = (
                current - baseline
                if current is not None and baseline is not None
                else None
            )
            differences.append(difference)
            displayed.append(f"{fmt(difference):>11s}")
        print(
            f"{method:24s} "
            + " ".join(displayed)
            + f" {fmt(mean(differences)):>11s}"
            + f" {fmt(sample_std(differences)):>11s}"
        )

print()
print("=" * 120)
print("FILTERING DETAILS")
print("=" * 120)

for run in runs:
    print()
    print("-" * 120)
    print(f"{run['method']} seed={run['seed']}")
    print("-" * 120)
    print("root:", run["root"])
    print("exit_code:", run["exit_code"])
    print("stdout:", run["log_path"])
    print("filter source:", run["filter_source"])
    print("filter events:", run["filter_events"])
    print("steps with filtering:", run["steps_with_filtering"])
    print("filtered total:", run["filtered_total"])
    print("filter denominator:", run["filter_denominator"])
    print("filter fraction:", run["filter_fraction"])
    print("kept fraction:", run["kept_fraction"])

    for name in (
        "skipped_per_step",
        "kept_fraction_events",
        "dot_mean",
        "dot_std",
        "grad_norm_mean",
        "grad_norm_std",
    ):
        stats = run[name]
        if stats:
            print(f"{name}: {stats}")

    selection = run["selection"]
    if selection is None:
        print("selection JSONL: MISSING/not recorded for this method")
        continue

    print("selection JSONL:", selection["path"])
    print("selection records:", selection["records"])
    print("score objectives:", selection["objectives"])
    print("mask applications:", selection["applications"])
    print("total samples:", selection["total"])
    print(
        "strict threshold filtered:",
        selection["threshold_filtered"],
        selection["threshold_filter_fraction"],
    )
    print(
        "actual filtered:",
        selection["filtered"],
        selection["filter_fraction"],
    )
    print("rescued by cap:", selection["rescued_by_cap"])
    print("cap-triggered records:", selection["cap_records"])
    print(
        "LOO signs:",
        {
            "negative": selection["negative"],
            "zero": selection["zero"],
            "positive": selection["positive"],
        },
    )
    print("LOO score:", selection["loo_score"])
    print("ordinary DTV self term:", selection["standard_self"])
    print("ordinary DTV cross term:", selection["standard_cross"])
    print("ordinary DTV score:", selection["standard_score"])
    print("filtered LOO score:", selection["filtered_loo_score"])
    print("kept LOO score:", selection["kept_loo_score"])

print()
print("=" * 120)
print("COMPLETENESS AND VALIDITY CHECKS")
print("=" * 120)

problems = []
for run in runs:
    identity = f"{run['method']} seed={run['seed']}"
    if run["root"] is None:
        problems.append(f"MISSING DIRECTORY: {identity}")
    if run["candidate_count"] > 1:
        problems.append(
            f"DUPLICATE CANDIDATES ({run['candidate_count']}): {identity}"
        )
    if run["exit_code"] not in (0, None):
        problems.append(
            f"NONZERO EXIT {run['exit_code']}: {identity}"
        )
    if run["post_accuracy"] is None:
        problems.append(f"MISSING POST EVAL: {identity}")
    if run["method"].startswith("Policy-LOO"):
        selection = run["selection"]
        if selection is None:
            problems.append(f"MISSING SELECTION JSONL: {identity}")
        else:
            if selection["objectives"] != ["policy"]:
                problems.append(
                    f"WRONG SCORE OBJECTIVE {selection['objectives']}: "
                    f"{identity}"
                )
            if selection["applications"] != ["full"]:
                problems.append(
                    f"WRONG MASK APPLICATION {selection['applications']}: "
                    f"{identity}"
                )

if problems:
    print("AUDIT STATUS: ATTENTION REQUIRED")
    for problem in problems:
        print(" -", problem)
else:
    print("AUDIT STATUS: COMPLETE")
    print("All 18 clean runs were found with post-train metrics.")
    print("All Policy-LOO runs identify policy scoring and Full masking.")

print()
print("LIMITATIONS:")
print(
    "1. Historical Baseline/L2/DTV-Policy runs do not store exact "
    "per-sample selection identities."
)
print(
    "2. TensorBoard filtering fractions assume 16 samples per recorded "
    "actor update; event counts are printed for verification."
)
print(
    "3. Do not compare sample indices across independently trained runs; "
    "their rollout/model trajectories diverge after updates."
)
print(
    "4. rank_noise0p2, keep75, and policy_only directories are explicitly "
    "excluded from this clean six-method audit."
)

fieldnames = [
    "method",
    "seed",
    "root",
    "exit_code",
    "pre_correct",
    "pre_total",
    "pre_accuracy",
    "post_correct",
    "post_total",
    "post_accuracy",
    "accuracy_delta",
    "post_partial",
    "post_format",
    "filter_fraction",
    "kept_fraction",
    "filtered_total",
    "filter_denominator",
    "filter_source",
    "filter_events",
    "steps_with_filtering",
]

with CSV_PATH.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for run in runs:
        writer.writerow({
            key: str(run[key]) if key == "root" else run[key]
            for key in fieldnames
        })

print()
print("CSV:", CSV_PATH)
PY

echo "Wrote:"
echo "  $OUT"
echo "  $CSV"
wc -l "$OUT"
ls -lh "$OUT" "$CSV"

echo
echo "Key summary:"
grep -A 10 "PER-METHOD THREE-SEED AGGREGATE" "$OUT"
echo
grep -A 80 "MATCHED-SEED POST-ACCURACY DIFFERENCES" "$OUT" | head -n 45
echo
tail -n 20 "$OUT"
