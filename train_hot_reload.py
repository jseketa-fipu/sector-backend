#!/usr/bin/env python3
"""
Continuous evolutionary training with periodic weight writes.

The live sim will pick up new weights on restart (no mid-run hot reload).

Example:
    python train_hot_reload.py --weights models/policy_live.pt --generations 10 --population 12 --episodes 3 --sigma 0.1 --loop --sleep 2 --log-rollouts logs/rollouts_live.jsonl
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, TextIO

import torch

from train_evo import evolutionary_train
from storage import RunStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous evolutionary training with hot reload.")
    parser.add_argument("--weights", type=Path, required=True, help="Where to write policy weights for the live sim.")
    parser.add_argument("--generations", type=int, default=10, help="Generations per cycle.")
    parser.add_argument("--population", type=int, default=10, help="Population size.")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per individual per generation.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Mutation noise stddev.")
    parser.add_argument("--max-ticks", type=int, default=400, help="Max ticks per episode.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to pause between cycles.")
    parser.add_argument("--loop", action="store_true", help="Run indefinitely instead of a single cycle.")
    parser.add_argument("--log-rollouts", type=Path, default=None, help="Optional JSONL file to append rollout actions.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory to store run artifacts (defaults to runs/...).")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name. Defaults to hot_<timestamp>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roll_fh: Optional[TextIO] = None
    store_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    store = RunStore(
        run_type="hot_reload",
        root=args.run_dir or Path("runs"),
        name=args.run_name,
        config=store_config,
    )

    if args.log_rollouts is None:
        args.log_rollouts = store.dir / "rollouts.jsonl"

    if args.log_rollouts:
        args.log_rollouts.parent.mkdir(parents=True, exist_ok=True)
        roll_fh = args.log_rollouts.open("a", encoding="utf-8")

    cycle = 1
    with store or nullcontext():
        while True:
            print(f"[cycle {cycle}] starting evolutionary training...")
            best_net, _ = evolutionary_train(
                generations=args.generations,
                population=args.population,
                episodes=args.episodes,
                sigma=args.sigma,
                max_ticks=args.max_ticks,
                device=device,
                rollout_log=roll_fh,
                store=store,
            )
            args.weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_net.state_dict(), args.weights)
            print(f"[cycle {cycle}] saved weights to {args.weights}")
            if store is not None:
                store.save_checkpoint(best_net.state_dict(), f"cycle_{cycle:03d}.pt")
                store.log({"event": "cycle_end", "cycle": cycle})

            if not args.loop:
                break
            cycle += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    if roll_fh:
        roll_fh.close()
    if store:
        store.close()


if __name__ == "__main__":
    main()
