"""
Enhanced Autoresearch Runner with Meta-Observer
================================================
Drop-in wrapper around Karpathy's autoresearch that adds
persistent learning, 12D observation, and strategy evolution.

Usage:
    python run_enhanced.py --agent-id autoresearch-001 --tag mar18

This wraps the standard autoresearch loop with meta-observation,
making the agent learn from every cycle rather than just keep/discard.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from meta_observer_runtime import MetaObserver


def parse_results(log_path: str) -> dict:
    """Parse train.py output log for metrics."""
    result = {
        "metric_name": "val_bpb",
        "metric_value": float('inf'),
        "peak_vram_mb": 0,
        "training_seconds": 0,
        "total_tokens_M": 0,
        "num_steps": 0,
        "num_params_M": 0,
        "depth": 0,
        "mfu_percent": 0,
        "outcome": "crash",
        "error": "",
    }

    try:
        with open(log_path, "r") as f:
            content = f.read()

        # Parse key metrics from output
        patterns = {
            "val_bpb": r"^val_bpb:\s*([\d.]+)",
            "peak_vram_mb": r"^peak_vram_mb:\s*([\d.]+)",
            "training_seconds": r"^training_seconds:\s*([\d.]+)",
            "total_tokens_M": r"^total_tokens_M:\s*([\d.]+)",
            "num_steps": r"^num_steps:\s*(\d+)",
            "num_params_M": r"^num_params_M:\s*([\d.]+)",
            "depth": r"^depth:\s*(\d+)",
            "mfu_percent": r"^mfu_percent:\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                result[key] = float(match.group(1))

        if result["metric_value"] < float('inf'):
            result["outcome"] = "pending"  # Will be set to keep/discard

        # Check for errors
        if "Error" in content or "Traceback" in content:
            result["outcome"] = "crash"
            # Extract last line of traceback
            lines = content.strip().split("\n")
            for line in reversed(lines):
                if line.strip():
                    result["error"] = line.strip()[:200]
                    break

    except Exception as e:
        result["error"] = str(e)

    return result


def classify_diff(diff: str) -> tuple[str, list[str]]:
    """
    Classify a git diff into action type and tags.
    This is how the meta-observer learns WHAT KIND of change was made.
    """
    diff_lower = diff.lower()
    tags = []
    action_type = "unknown"

    # Learning rate changes
    if any(k in diff_lower for k in ["_lr", "learning_rate", "lr ="]):
        action_type = "hyperparameter_lr"
        tags.append("lr")
        if "embedding_lr" in diff_lower:
            tags.append("embedding_lr")
        if "matrix_lr" in diff_lower:
            tags.append("matrix_lr")
        if "scalar_lr" in diff_lower:
            tags.append("scalar_lr")

    # Batch size changes
    elif any(k in diff_lower for k in ["batch_size", "total_batch", "device_batch"]):
        action_type = "hyperparameter_batch"
        tags.append("batch_size")

    # Architecture depth
    elif any(k in diff_lower for k in ["n_layer", "depth", "num_layers"]):
        action_type = "architecture_depth"
        tags.append("depth")

    # Architecture width
    elif any(k in diff_lower for k in ["n_embd", "n_head", "head_dim", "aspect_ratio"]):
        action_type = "architecture_width"
        tags.append("width")

    # Attention changes
    elif any(k in diff_lower for k in ["attention", "n_kv_head", "window_pattern", "rope", "flash"]):
        action_type = "architecture_attention"
        tags.append("attention")

    # Optimizer changes
    elif any(k in diff_lower for k in ["optimizer", "adam", "muon", "momentum", "beta"]):
        action_type = "optimizer_config"
        tags.append("optimizer")

    # Scheduler changes
    elif any(k in diff_lower for k in ["warmup", "warmdown", "schedule", "final_lr"]):
        if "warmup" in diff_lower:
            action_type = "scheduler_warmup"
            tags.append("warmup")
        else:
            action_type = "scheduler_decay"
            tags.append("decay")

    # Regularization
    elif any(k in diff_lower for k in ["weight_decay", "dropout", "regulariz"]):
        action_type = "regularization"
        tags.append("regularization")

    # Activation function
    elif any(k in diff_lower for k in ["relu", "gelu", "silu", "activation", "swish"]):
        action_type = "activation_function"
        tags.append("activation")

    # Normalization
    elif any(k in diff_lower for k in ["rmsnorm", "layernorm", "norm"]):
        action_type = "normalization"
        tags.append("normalization")

    # Multiple changes
    if len(tags) == 0:
        action_type = "mixed_change"
        tags.append("multi")

    return action_type, tags


def run_experiment(observer: MetaObserver, best_bpb: float,
                   results_file: str, dry_run: bool = False) -> float:
    """
    Run a single experiment cycle with meta-observation.
    Returns the new best_bpb.
    """
    cycle = observer.cycle

    # Get suggestion from meta-observer
    suggestion = observer.suggest_next()
    epoch = suggestion.get('epoch', '?')
    epoch_cycle = suggestion.get('epoch_cycle', '?')
    phase = suggestion.get('epoch_phase', '?')
    becoming = suggestion.get('becoming_score', 0)

    print(f"\n{'='*60}")
    print(f"CYCLE {cycle} | Epoch {epoch}:{epoch_cycle}/57 | Phase: {phase}")
    print(f"Strategy: {suggestion['strategy']} | Suggested: {suggestion['action_type']}")
    print(f"Reasoning: {suggestion['reasoning']}")
    if becoming > 0:
        print(f"Becoming: {becoming:.6f}")
    if suggestion.get("parent_seed_resume_rule"):
        print(f"Seed guidance: {suggestion['parent_seed_resume_rule'][:100]}")
    if suggestion.get("warning"):
        print(f"WARNING: {suggestion['warning']}")
    if suggestion.get("cross_agent_insights"):
        for ci in suggestion["cross_agent_insights"]:
            print(f"  Cross-agent [{ci['from']}]: {ci['insight']}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Skipping actual training")
        return best_bpb

    # Get the diff of what was changed (agent should have committed already)
    diff_result = subprocess.run(
        ["git", "diff", "HEAD~1", "--", "train.py"],
        capture_output=True, text=True
    )
    diff = diff_result.stdout[:5000]

    # Classify the change
    action_type, tags = classify_diff(diff)

    # Get commit message as description
    commit_msg = subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        capture_output=True, text=True
    ).stdout.strip()

    commit_hash = subprocess.run(
        ["git", "log", "-1", "--format=%h"],
        capture_output=True, text=True
    ).stdout.strip()

    # Run training
    print(f"Running train.py...")
    log_path = "run.log"
    start_time = time.time()

    train_result = subprocess.run(
        ["uv", "run", "train.py"],
        capture_output=True, text=True,
        timeout=600  # 10 minute hard timeout
    )

    # Write log
    with open(log_path, "w") as f:
        f.write(train_result.stdout)
        if train_result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(train_result.stderr)

    elapsed = time.time() - start_time

    # Parse results
    result = parse_results(log_path)

    # Determine outcome
    if result["outcome"] == "crash":
        outcome = "crash"
    elif result["metric_value"] < best_bpb:
        outcome = "keep"
        best_bpb = result["metric_value"]
    else:
        outcome = "discard"

    result["outcome"] = outcome
    result["metric_delta"] = best_bpb - result["metric_value"] if outcome != "crash" else 0

    # OBSERVE: Feed everything to meta-observer
    obs = observer.observe(
        action={
            "type": action_type,
            "description": commit_msg,
            "diff": diff,
            "strategy": suggestion["strategy"],
            "tags": tags,
            "from_cross_agent": bool(suggestion.get("cross_agent_insights")),
        },
        result={
            "metric_name": "val_bpb",
            "metric_value": result["metric_value"],
            "metric_delta": result.get("metric_delta", 0),
            "outcome": outcome,
            "environment": {
                "vram_gb": result.get("peak_vram_mb", 0) / 1024,
                "mfu": result.get("mfu_percent", 0),
                "training_seconds": result.get("training_seconds", 0),
                "elapsed": elapsed,
            },
        }
    )

    # DECIDE
    observer.decide(result)

    # Log to results.tsv
    memory_gb = result.get("peak_vram_mb", 0) / 1024
    with open(results_file, "a") as f:
        f.write(f"{commit_hash}\t{result['metric_value']:.6f}\t"
                f"{memory_gb:.1f}\t{outcome}\t{commit_msg}\n")

    # Print observation summary
    print(f"\n--- Cycle {cycle} Result ---")
    print(f"  val_bpb: {result['metric_value']:.6f} ({outcome})")
    print(f"  VRAM: {memory_gb:.1f} GB")
    print(f"  12D magnitude: {obs.magnitude():.3f}")
    print(f"  Velocity: {obs.velocity:.4f}")
    print(f"  Pattern match: {obs.pattern_match or 'none'}")
    print(f"  Best so far: {best_bpb:.6f} (cycle {observer.best_metric_cycle})")

    # Handle discard: reset git
    if outcome == "discard":
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)
        print(f"  → Discarded (git reset)")
    elif outcome == "crash":
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)
        print(f"  → Crashed: {result.get('error', 'unknown')}")
    else:
        print(f"  → KEPT! New best: {best_bpb:.6f}")

    # Periodic report
    if cycle % 10 == 0:
        report = observer.report()
        print(f"\n{'='*60}")
        print(f"META-OBSERVER REPORT (Cycle {cycle})")
        print(f"{'='*60}")
        print(f"  Total cycles: {report['total_cycles']}")
        print(f"  Best metric: {report['best_metric']:.6f}")
        print(f"  Current strategy: {report['current_strategy']}")
        print(f"  Patterns discovered: {len(report['patterns'])}")
        for p in report['patterns'][:5]:
            print(f"    - {p['action_type']}: {p['success_rate']:.0%} success "
                  f"({p['sample_count']} samples)")
        if report['top_recommendations']:
            print(f"  Top recommendations:")
            for r in report['top_recommendations'][:3]:
                print(f"    - [{r['type']}] {r['action']}: {r['reason']}")
        print(f"{'='*60}\n")

    return best_bpb


def main():
    parser = argparse.ArgumentParser(description="Enhanced Autoresearch with Meta-Observer")
    parser.add_argument("--agent-id", default="autoresearch-001",
                        help="Unique agent identifier")
    parser.add_argument("--tag", default="enhanced",
                        help="Experiment tag for git branch")
    parser.add_argument("--db", default="meta_observer.db",
                        help="Path to meta-observer database")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Max cycles (0=infinite)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print suggestions without running experiments")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         AUTORESEARCH + ATHENA META-OBSERVER                 ║
║         Self-Improving Autonomous ML Research               ║
╠══════════════════════════════════════════════════════════════╣
║  Agent: {args.agent_id:<50s}  ║
║  Tag:   {args.tag:<50s}  ║
║  DB:    {args.db:<50s}  ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Initialize meta-observer
    with MetaObserver(
        agent_id=args.agent_id,
        project="autoresearch-nanochat",
        db_path=args.db,
        log_dir="meta_observer_logs",
    ) as observer:

        # Initialize results.tsv if needed
        results_file = "results.tsv"
        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

        # Get current best from existing results
        best_bpb = observer.best_metric
        if best_bpb == float('inf'):
            best_bpb = 999.0  # Will be replaced by first actual result

        # Main loop
        for cycle in observer.loop():
            if args.max_cycles > 0 and cycle > args.max_cycles:
                print(f"\nReached max cycles ({args.max_cycles}). Generating final report...")
                report = observer.report()
                print(json.dumps(report, indent=2, default=str))
                break

            try:
                best_bpb = run_experiment(
                    observer, best_bpb, results_file,
                    dry_run=args.dry_run
                )
            except KeyboardInterrupt:
                print(f"\n\nInterrupted at cycle {cycle}. Generating final report...")
                report = observer.report()
                print(json.dumps(report, indent=2, default=str))
                break
            except Exception as e:
                print(f"Cycle {cycle} error: {e}")
                observer.observe(
                    action={"type": "error", "description": str(e)},
                    result={"outcome": "crash", "error": str(e), "metric_value": float('inf')}
                )
                continue


if __name__ == "__main__":
    main()
