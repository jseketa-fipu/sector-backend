#!/usr/bin/env python3
"""
Evolutionary self-play trainer for the neural policy.

Usage (offline):
    python train_evo.py --generations 20 --population 12 --episodes 3 --sigma 0.1 --save models/policy_evo.pt
"""
from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, TextIO, Optional

import torch

from bots import ATTEMPTS_PER_TICK, reset_factions
from world import World, Order, advance_world, create_sector, FACTION_NAMES
from train_selfplay import PolicyNet, build_candidates, idle_strength_by_system
from storage import RunStore


@dataclass
class EvalResult:
    reward: float
    winner: str | None
    ticks: int
    orders: int


def detect_winner(world: World) -> str | None:
    owners = {sys.owner for sys in world.systems.values() if sys.owner is not None}
    if len(owners) == 1:
        return next(iter(owners))
    return None


def policy_act(world: World, faction: str, net: PolicyNet, device: torch.device, max_orders: int = ATTEMPTS_PER_TICK) -> List[Order]:
    """Sample orders from the policy (used for evaluation only, no gradients)."""
    remaining = idle_strength_by_system(world, faction)
    used_origins: set[int] = set()
    chosen: List[Order] = []

    for _ in range(max_orders):
        candidates = build_candidates(world, faction, remaining, used_origins)
        if not candidates:
            break
        feats = torch.tensor([c.features for c in candidates], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = net(feats)
            probs = torch.softmax(logits, dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()

        selected = candidates[idx]
        if selected.order is None:
            break

        chosen.append(selected.order)
        if selected.origin_id is not None:
            used_origins.add(selected.origin_id)
            remaining.pop(selected.origin_id, None)

    return chosen


def run_episode(net: PolicyNet, device: torch.device, max_ticks: int, rollout_log: TextIO | None, episode_idx: int) -> EvalResult:
    reset_factions()
    world = create_sector()
    factions = list(FACTION_NAMES.keys())
    winner: str | None = None
    orders_count = 0
    rollout_entries: List[dict] = []

    while world.tick < max_ticks and winner is None:
        orders: List[Order] = []
        for fid in factions:
            orders_for = policy_act(world, fid, net, device)
            orders.extend(orders_for)
            orders_count += len(orders_for)
            if rollout_log is not None:
                for order in orders_for:
                    rollout_entries.append(
                        {
                            "episode": episode_idx,
                            "tick": world.tick,
                            "faction": fid,
                            "origin": order.origin_id,
                            "target": order.target_id,
                            "reason": order.reason,
                        }
                    )
        advance_world(world, orders)
        winner = detect_winner(world)

    reward = 0.0
    if winner:
        reward = 1.0

    if rollout_log is not None and rollout_entries:
        for entry in rollout_entries:
            entry["winner"] = winner
            rollout_log.write(json.dumps(entry) + "\n")

    return EvalResult(reward=reward, winner=winner, ticks=world.tick, orders=orders_count)


def flatten_params(net: PolicyNet) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in net.parameters()])


def assign_params(net: PolicyNet, vector: torch.Tensor) -> None:
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        slice_ = vector[offset : offset + numel].view_as(p.data)
        p.data.copy_(slice_)
        offset += numel


def mutate(base: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(base) * sigma
    return base + noise


def evolutionary_train(
    generations: int,
    population: int,
    episodes: int,
    sigma: float,
    max_ticks: int,
    device: torch.device,
    rollout_log: TextIO | None,
    store: Optional[RunStore] = None,
) -> Tuple[PolicyNet, List[EvalResult]]:
    assert population >= 2, "Population size must be at least 2."
    net = PolicyNet().to(device)
    base_params = flatten_params(net)
    param_dim = base_params.numel()

    # Initialize population around the base parameters
    pop_vectors = [mutate(base_params, sigma) for _ in range(population)]
    history: List[EvalResult] = []

    for gen in range(1, generations + 1):
        scores: List[Tuple[float, torch.Tensor, EvalResult]] = []
        for idx, vec in enumerate(pop_vectors):
            assign_params(net, vec)
            rewards = []
            last_result: EvalResult | None = None
            for _ in range(episodes):
                result = run_episode(net, device, max_ticks, rollout_log, episode_idx=(gen - 1) * episodes + _ + 1)
                rewards.append(result.reward)
                last_result = result
                history.append(result)
            avg_reward = sum(rewards) / len(rewards)
            scores.append((avg_reward, vec, last_result))  # type: ignore[arg-type]

        scores.sort(key=lambda t: t[0], reverse=True)
        best_reward, best_vec, best_res = scores[0]

        print(
            f"[gen {gen:03d}] best_reward={best_reward:.3f} "
            f"winner={best_res.winner or 'none':>4} ticks={best_res.ticks:4d} "
            f"orders={best_res.orders:3d}"
        )
        if store is not None:
            store.log(
                {
                    "event": "generation_end",
                    "generation": gen,
                    "best_reward": best_reward,
                    "winner": best_res.winner,
                    "ticks": best_res.ticks,
                    "orders": best_res.orders,
                }
            )

        elite_count = max(1, population // 5)
        elites = [vec for _, vec, _ in scores[:elite_count]]

        # Rebuild population: keep elites and spawn mutants around them
        new_pop: List[torch.Tensor] = []
        new_pop.extend(elites)
        while len(new_pop) < population:
            parent = random.choice(elites)
            child = mutate(parent, sigma)
            new_pop.append(child)

        pop_vectors = new_pop

    # Return best individual as policy
    assign_params(net, scores[0][1])
    if store is not None:
        store.save_checkpoint(net.state_dict(), "best.pt")
    return net, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolutionary self-play trainer.")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations.")
    parser.add_argument("--population", type=int, default=10, help="Population size.")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per individual per generation.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Mutation noise stddev.")
    parser.add_argument("--max-ticks", type=int, default=400, help="Max ticks per episode.")
    parser.add_argument("--save", type=Path, default=None, help="Where to save best policy weights.")
    parser.add_argument("--log-rollouts", type=Path, default=None, help="Optional JSONL file to log actions.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory to store run artifacts (defaults to runs/...).")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name. Defaults to evo_<timestamp>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fh: TextIO | None = None
    store_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    store = RunStore(
        run_type="evo",
        root=args.run_dir or Path("runs"),
        name=args.run_name,
        config=store_config,
    )

    if args.log_rollouts is None:
        args.log_rollouts = store.dir / "rollouts.jsonl"

    if args.log_rollouts:
        args.log_rollouts.parent.mkdir(parents=True, exist_ok=True)
        log_fh = args.log_rollouts.open("w", encoding="utf-8")

    with store or nullcontext():  # type: ignore[name-defined]
        best_net, _ = evolutionary_train(
            generations=args.generations,
            population=args.population,
            episodes=args.episodes,
            sigma=args.sigma,
            max_ticks=args.max_ticks,
            device=device,
            rollout_log=log_fh,
            store=store,
        )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_net.state_dict(), args.save)
        print(f"Saved best policy to {args.save}")

    if log_fh:
        log_fh.close()
    if store:
        store.close()


if __name__ == "__main__":
    main()
