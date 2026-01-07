
import os
import csv
from tensorboard.backend.event_processing import event_accumulator

log_dir = "/tmp/content/tmp/tensorboard/grpo"
files = [f for f in os.listdir(log_dir) if "tfevents" in f]
files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
latest_file = os.path.join(log_dir, files[-1])

print(f"Reading file: {latest_file}")
ea = event_accumulator.EventAccumulator(latest_file)
ea.Reload()

tags_to_check = [
    'global/train/rewards/match_format_exactly',
    'global/train/rewards/check_answer',
    'global/train/rewards/match_format_approximately',
    'global/train/rewards/check_numbers'
]

results = {}

for tag in tags_to_check:
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        # Get last 10 values to average
        values = [e.value for e in events[-10:]]
        if values:
            avg_val = sum(values) / len(values)
            results[tag] = avg_val
            print(f"{tag}: {avg_val:.4f}")
        else:
            print(f"{tag}: No data")
    else:
        print(f"{tag}: Not found")
