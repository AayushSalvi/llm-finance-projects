"""
log_experiment.py - Save experiment config + results for FinSentiment-LLM.
"""

import json
import os
from datetime import datetime


def log_experiment(config, results, notes=""):
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "notes": notes,
        "config": {},
        "results": results,
    }

    for k, v in config.items():
        try:
            json.dumps(v)
            experiment["config"][k] = v
        except TypeError:
            experiment["config"][k] = str(v)

    log_path = "logs/experiments.jsonl"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(experiment) + "\n")

    print(f"Experiment logged to {log_path}")


def view_experiments(log_path="logs/experiments.jsonl"):
    if not os.path.exists(log_path):
        print("No experiments logged yet.")
        return

    with open(log_path, "r") as f:
        experiments = [json.loads(line) for line in f]

    print(f"\n{'=' * 100}")
    print(f"{'#':>3} {'Date':>12} {'Model':>15} {'Rank':>5} {'Alpha':>6} {'LR':>10} {'Epochs':>6} {'Base Acc':>9} {'FT Acc':>9} {'Time':>7}")
    print(f"{'-' * 100}")

    for i, exp in enumerate(experiments):
        c = exp["config"]
        r = exp["results"]
        print(
            f"{i+1:>3} "
            f"{exp['timestamp'][:10]:>12} "
            f"{str(c.get('model_name', '-'))[-15:]:>15} "
            f"{c.get('r', '-'):>5} "
            f"{c.get('lora_alpha', '-'):>6} "
            f"{c.get('learning_rate', '-'):>10} "
            f"{c.get('num_train_epochs', '-'):>6} "
            f"{r.get('base_accuracy', '-'):>9} "
            f"{r.get('ft_accuracy', '-'):>9} "
            f"{r.get('training_time_minutes', '-'):>7}"
        )

    print(f"{'=' * 100}")
    for i, exp in enumerate(experiments):
        if exp.get("notes"):
            print(f"  #{i+1}: {exp['notes']}")


if __name__ == "__main__":
    view_experiments()