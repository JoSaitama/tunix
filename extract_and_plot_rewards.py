
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

train_rewards = []
if 'global/train/rewards/sum' in ea.Tags()['scalars']:
    events = ea.Scalars('global/train/rewards/sum')
    for event in events:
        train_rewards.append((event.step, event.value))

eval_rewards = []
if 'global/eval/rewards/sum' in ea.Tags()['scalars']:
    events = ea.Scalars('global/eval/rewards/sum')
    for event in events:
        eval_rewards.append((event.step, event.value))

print(f"Found {len(train_rewards)} train reward points.")
print(f"Found {len(eval_rewards)} eval reward points.")

if train_rewards:
    print(f"Last train reward: Step {train_rewards[-1][0]}, Value {train_rewards[-1][1]:.4f}")

# Save to CSV
csv_path = "rewards.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["step", "train_reward_sum", "eval_reward_sum"])
    # Merge by step
    steps = sorted(list(set([x[0] for x in train_rewards] + [x[0] for x in eval_rewards])))
    train_dict = dict(train_rewards)
    eval_dict = dict(eval_rewards)
    
    for step in steps:
        writer.writerow([step, train_dict.get(step, ""), eval_dict.get(step, "")])

print(f"Saved rewards to {csv_path}")

# Generate simple SVG
if not train_rewards and not eval_rewards:
    print("No rewards to plot.")
    exit(0)

# SVG Viewport
width = 800
height = 400
margin = 50

# Determine scales
all_steps = [x[0] for x in train_rewards] + [x[0] for x in eval_rewards]
all_values = [x[1] for x in train_rewards] + [x[1] for x in eval_rewards]

if not all_steps:
    exit(0)

min_step, max_step = min(all_steps), max(all_steps)
min_val, max_val = min(all_values), max(all_values)

# Add some padding to value range
val_range = max_val - min_val
if val_range == 0: val_range = 1.0
min_val -= val_range * 0.1
max_val += val_range * 0.1

def val_to_y(v):
    return height - margin - ((v - min_val) / (max_val - min_val)) * (height - 2 * margin)

def step_to_x(s):
    if max_step == min_step: return margin
    return margin + ((s - min_step) / (max_step - min_step)) * (width - 2 * margin)

svg_lines = []
svg_lines.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
svg_lines.append(f'<rect width="100%" height="100%" fill="white"/>')

# Axes
svg_lines.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>') # X
svg_lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>') # Y

# Plot Train (Blue)
if train_rewards:
    points = []
    for s, v in train_rewards:
        points.append(f'{step_to_x(s)},{val_to_y(v)}')
    svg_lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="blue" stroke-width="2"/>')

# Plot Eval (Red)
if eval_rewards:
    points = []
    for s, v in eval_rewards:
        points.append(f'{step_to_x(s)},{val_to_y(v)}')
    svg_lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="red" stroke-width="2"/>')

# Labels
svg_lines.append(f'<text x="{width//2}" y="{height-10}" text-anchor="middle">Step</text>')
svg_lines.append(f'<text x="10" y="{height//2}" transform="rotate(-90 10,{height//2})" text-anchor="middle">Reward Sum</text>')
svg_lines.append(f'<text x="{width-margin}" y="{height-margin+20}" text-anchor="end">{max_step}</text>')
svg_lines.append(f'<text x="{margin}" y="{height-margin+20}" text-anchor="middle">{min_step}</text>')
svg_lines.append(f'<text x="{margin-5}" y="{margin}" text-anchor="end">{max_val:.2f}</text>')
svg_lines.append(f'<text x="{margin-5}" y="{height-margin}" text-anchor="end">{min_val:.2f}</text>')

# Legend
svg_lines.append(f'<rect x="{width-150}" y="20" width="130" height="50" fill="white" stroke="black"/>')
svg_lines.append(f'<line x1="{width-140}" y1="35" x2="{width-110}" y2="35" stroke="blue" stroke-width="2"/>')
svg_lines.append(f'<text x="{width-105}" y="40">Train</text>')
svg_lines.append(f'<line x1="{width-140}" y1="55" x2="{width-110}" y2="55" stroke="red" stroke-width="2"/>')
svg_lines.append(f'<text x="{width-105}" y="60">Eval</text>')

svg_lines.append('</svg>')

with open("reward_curve.svg", "w") as f:
    f.write("\n".join(svg_lines))

print("Saved plot to reward_curve.svg")
