
import os
import sys
from tensorboard.backend.event_processing import event_accumulator

log_dir = "/tmp/content/tmp/tensorboard/grpo"
# specific_file = "events.out.tfevents.1766905689.t1v-n-62e31e26-w-0" 
# Find the latest file
files = [f for f in os.listdir(log_dir) if "tfevents" in f]
files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
latest_file = os.path.join(log_dir, files[-1])

print(f"Reading file: {latest_file}")

ea = event_accumulator.EventAccumulator(latest_file)
ea.Reload()

print("Tags found:")
for tag in ea.Tags()['scalars']:
    print(tag)
