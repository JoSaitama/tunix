"""Evaluate a Qwen2.5 PKU run on PKU test and RewardBench 2 Safety."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from examples.dpo import eval_qwen2p5_rewardbench_v2 as rewardbench
from examples.dpo import qwen2p5_dpo_eval_lib as eval_lib


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("sft_model_path", type=Path)
  args = parser.parse_args()

  cfg = OmegaConf.load(args.run_dir / "launch.yaml")
  rewardbench_rows = [
      row for row in rewardbench._load_rewardbench_v2_rows(None)
      if row["subset"].lower() == "safety"
  ]
  if not rewardbench_rows:
    raise ValueError("RewardBench 2 test contains no Safety rows.")

  bundle = eval_lib.load_eval_bundle(
      base_cfg=cfg,
      actor_model_path=str(args.run_dir / "exported_model"),
      reference_model_path=str(args.sft_model_path),
      metrics_prefix="pku_safety",
  )
  trainer, metrics_logger = eval_lib.create_eval_trainer(bundle)
  try:
    test_batches = eval_lib.load_eval_dataset(
        module_spec="tunix.examples.data.pku_saferlhf_dpo:create_dataset(split='test')",
        tokenizer=bundle.tokenizer,
        batch_size=cfg.eval_batch_size,
        num_batches=None,
        dpo_config=bundle.training_config,
    )
    correct = count = 0
    with bundle.actor_mesh:
      for batch in test_batches:
        metrics = eval_lib.iter_preference_batch_metrics(
            trainer=trainer,
            prompts=batch["prompts"],
            chosen_responses=batch["chosen_responses"],
            rejected_responses=batch["rejected_responses"],
        )
        correct += sum(int(row["rewards_accuracy"]) for row in metrics)
        count += len(metrics)
    if not count:
      raise ValueError("No PKU test pairs remained after length filtering.")

    scored_rows = []
    for row in rewardbench_rows:
      scores = rewardbench._score_candidate_group(
          bundle=bundle, trainer=trainer, row=row, batch_size=cfg.eval_batch_size
      )
      scored_rows.append({
          "id": row["id"],
          "subset": row["subset"],
          "num_correct": row["num_correct"],
          "scores": scores,
      })
    rb2_scores = rewardbench._summarize_rewardbench_v2_rows(
        scored_rows=scored_rows
    )
  finally:
    eval_lib.close_eval_trainer(trainer, metrics_logger)
    eval_lib.close_eval_bundle(bundle)

  summary = {
      "pku_test_accuracy": correct / count,
      "pku_test_num_pairs": count,
      "rewardbench2_safety": rb2_scores["rewardbench2_safety"],
      "rewardbench2_safety_num_prompts": len(scored_rows),
      "scoring_backend": rewardbench.SCORING_BACKEND,
  }
  result = json.dumps(summary, indent=2)
  (args.run_dir / "safety_metrics.json").write_text(result + "\n")
  print(result)


if __name__ == "__main__":
  main()
