#!/usr/bin/env python3
"""
Self-play training harness for a simple neural policy that issues orders.

Usage (offline):
    python train_selfplay.py --episodes 50 --max-ticks 400 --lr 5e-4 --save models/policy.pt

The policy runs the same network for every faction (self-play) and optimizes
for winning the simulated match. This is a lightweight REINFORCE loop; it is
intended as a starting point rather than a tuned solution.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TextIO

import torch
from torch import nn
from torch.distributions import Categorical

from bots import (
    ATTEMPTS_PER_TICK,
    MIN_LAUNCH_STRENGTH,
    reset_factions,
)
from world import (
    CAPTURE_TICKS,
    GARRISON_MAX,
    Order,
    World,
    advance_world,
    create_sector,
    FACTION_NAMES,
)
from storage import RunStore

# --------- Policy and featurization ---------


@dataclass
class CandidateOrder:
    """A single candidate action for the policy to pick from."""

    order: Optional[Order]  # None represents a deliberate "do nothing"
    features: List[float]
    origin_id: Optional[int]  # track to avoid double-launching from the same system
    attack_strength: float = 0.0
    enemy_strength: float = 0.0


def idle_strength_by_system(world: World, faction: str) -> Dict[int, float]:
    """Return total idle strength per system for a faction."""
    strength: Dict[int, float] = {}
    for fl in world.fleets.values():
        if (
            fl.owner == faction
            and fl.system_id is not None
            and fl.eta == 0
            and fl.strength > 0
        ):
            strength[fl.system_id] = strength.get(fl.system_id, 0.0) + fl.strength
    return strength


def enemy_strength_by_system(world: World, faction: str) -> Dict[int, float]:
    """Return total idle enemy strength per system (including garrison if not ours)."""
    strength: Dict[int, float] = {}
    for fl in world.fleets.values():
        if (
            fl.owner == faction
            or fl.system_id is None
            or fl.eta != 0
            or fl.strength <= 0
        ):
            continue
        strength[fl.system_id] = strength.get(fl.system_id, 0.0) + fl.strength
    for sid, sys in world.systems.items():
        if sys.owner != faction:
            strength[sid] = strength.get(sid, 0.0) + sys.garrison
    return strength


def build_candidates(world: World, faction: str, remaining: Dict[int, float], used_origins: set[int]) -> List[CandidateOrder]:
    """Enumerate legal neighbor moves plus a no-op."""
    candidates: List[CandidateOrder] = []
    enemy_strength = enemy_strength_by_system(world, faction)
    for sys_id, sys in world.systems.items():
        if sys.owner != faction or sys_id in used_origins:
            continue
        strength_here = remaining.get(sys_id, 0.0)
        if strength_here < MIN_LAUNCH_STRENGTH:
            continue
        for nid in sys.neighbors:
            target = world.systems[nid]
            owner = target.owner
            enemy_here = enemy_strength.get(nid, 0.0)
            occ_progress = target.occupation_progress / float(max(1, CAPTURE_TICKS))
            occ_ours = 1.0 if target.occupation_faction == faction else 0.0
            features = [
                min(1.0, strength_here / 20.0),  # idle strength at origin
                sys.value / 10.0,  # origin value
                target.value / 10.0,  # target value
                1.0 if owner == faction else 0.0,  # target_owned_self
                1.0 if owner is None else 0.0,  # target_neutral
                1.0 if (owner is not None and owner != faction) else 0.0,  # target_enemy
                min(1.0, target.garrison / max(1.0, GARRISON_MAX)),  # garrison ratio
                min(1.0, target.heat / 3.0),  # recent conflict
                target.stability if target.stability is not None else 0.0,  # stability hint
                min(1.0, enemy_here / 20.0),  # local enemy strength
                occ_progress,  # occupation progress normalized
                occ_ours,  # whether we are occupying
            ]
            candidates.append(
                CandidateOrder(
                    order=Order(
                        faction=faction,
                        origin_id=sys_id,
                        target_id=nid,
                        reason="nn_policy",
                    ),
                    features=features,
                    origin_id=sys_id,
                    attack_strength=strength_here,
                    enemy_strength=enemy_here,
                )
            )

    # Always include a no-op so the policy can decide to hold
    candidates.append(
        CandidateOrder(
            order=None,
            features=[0.0] * 12,
            origin_id=None,
            attack_strength=0.0,
            enemy_strength=0.0,
        )
    )
    return candidates


class PolicyNet(nn.Module):
    """Tiny MLP scoring each candidate action."""

    def __init__(self, input_dim: int = 12, hidden: int = 96):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, input_dim] -> logits [N]
        return self.model(x).squeeze(-1)


class NeuralPolicy:
    """
    Shared policy for all factions (self-play). Selects up to ATTEMPTS_PER_TICK
    orders per faction per tick, choosing among neighbor moves plus a no-op.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.net = PolicyNet().to(self.device)

    def parameters(self):
        return self.net.parameters()

    def act(
        self,
        world: World,
        faction: str,
        max_orders: int = ATTEMPTS_PER_TICK,
        return_metadata: bool = False,
    ) -> Tuple[List[Order], List[torch.Tensor], List[dict]]:
        remaining = idle_strength_by_system(world, faction)
        used_origins: set[int] = set()
        chosen_orders: List[Order] = []
        log_probs: List[torch.Tensor] = []
        meta: List[dict] = []

        for _ in range(max_orders):
            candidates = build_candidates(world, faction, remaining, used_origins)
            if not candidates:
                break
            feats = torch.tensor([c.features for c in candidates], dtype=torch.float32, device=self.device)
            logits = self.net(feats)
            dist = Categorical(logits=logits)
            idx = dist.sample()
            log_probs.append(dist.log_prob(idx))

            selected_idx = idx.item()
            selected = candidates[selected_idx]
            # No-op selected: stop issuing orders this tick
            if selected.order is None:
                break

            chosen_orders.append(selected.order)
            if selected.origin_id is not None:
                used_origins.add(selected.origin_id)
                remaining.pop(selected.origin_id, None)

            if return_metadata:
                meta.append(
                    {
                        "faction": faction,
                        "origin": selected.origin_id,
                        "target": selected.order.target_id if selected.order else None,
                        "features": selected.features,
                        "candidate_index": selected_idx,
                        "attack_strength": selected.attack_strength,
                        "enemy_strength": selected.enemy_strength,
                        "target_owner": world.systems[selected.order.target_id].owner if selected.order else None,
                    }
                )

        return chosen_orders, log_probs, meta


# --------- Self-play trainer ---------


def detect_winner(world: World) -> Optional[str]:
    """Return the sole remaining owner if the map is unified."""
    owners = {sys.owner for sys in world.systems.values() if sys.owner is not None}
    if len(owners) == 1:
        return next(iter(owners))
    return None


@dataclass
class EpisodeResult:
    winner: Optional[str]
    ticks: int
    total_orders: int
    loss_value: float


class SelfPlayTrainer:
    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-3,
        max_ticks: int = 500,
        rollout_log: Optional[TextIO] = None,
        policy: Optional[NeuralPolicy] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.device = device
        self.policy = policy or NeuralPolicy(device)
        self.optimizer = optimizer or torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.max_ticks = max_ticks
        self.rollout_log = rollout_log

    def run_episode(self, episode_idx: int) -> EpisodeResult:
        reset_factions()
        world = create_sector()
        factions = list(FACTION_NAMES.keys())
        trajectories: Dict[str, List[Tuple[torch.Tensor, float]]] = {f: [] for f in factions}
        rollout_entries: List[dict] = []

        winner: Optional[str] = None
        total_orders = 0

        while world.tick < self.max_ticks and winner is None:
            orders: List[Order] = []
            for fid in factions:
                chosen, logs, meta = self.policy.act(world, fid, return_metadata=True)
                orders.extend(chosen)
                total_orders += len(chosen)
                for lp, entry in zip(logs, meta):
                    enemy = float(entry.get("enemy_strength", 0.0))
                    ours = float(entry.get("attack_strength", 0.0))
                    target_owner = entry.get("target_owner")
                    # Shaped reward: favor high-odds/soft targets, penalize low odds
                    base = 0.0
                    if target_owner is None:
                        base += 0.05
                    if target_owner == fid:
                        base -= 0.05  # discourage needless self-hops
                    if ours <= 0.0:
                        base -= 0.05
                    elif enemy <= 0.0:
                        base += 0.15
                    else:
                        ratio = ours / max(0.1, enemy)
                        if ratio >= 1.5:
                            base += 0.1
                        elif ratio >= 1.0:
                            base += 0.05
                        else:
                            base -= min(0.3, 0.1 / max(0.1, ratio))
                    trajectories[fid].append((lp, base))
                    if self.rollout_log is not None:
                        entry["tick"] = world.tick
                        entry["episode"] = episode_idx
                        entry["shaped_reward"] = base
                        rollout_entries.append(entry)

            advance_world(world, orders)
            winner = detect_winner(world)

        # Simple terminal reward: +1 for winner, -1 for everyone else, 0 on draw/time limit
        loss = torch.tensor(0.0, device=self.device)
        for fid, actions in trajectories.items():
            if not actions:
                continue
            terminal = 1.0 if winner == fid else (-1.0 if winner else -0.1)
            for lp, shaped in actions:
                loss = loss - (shaped + terminal) * lp

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.rollout_log is not None and rollout_entries:
            for entry in rollout_entries:
                entry["winner"] = winner
                entry["episode"] = episode_idx
                self.rollout_log.write(json.dumps(entry) + "\n")

        return EpisodeResult(winner=winner, ticks=world.tick, total_orders=total_orders, loss_value=float(loss.item()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural policy via self-play.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of self-play episodes.")
    parser.add_argument("--max-ticks", type=int, default=500, help="Max ticks per episode.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the policy weights.")
    parser.add_argument("--log-rollouts", type=Path, default=None, help="Optional JSONL file to log action rollouts.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory to store run artifacts (defaults to runs/...).")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name. Defaults to selfplay_<timestamp>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    store = RunStore(
        run_type="selfplay",
        root=args.run_dir or Path("runs"),
        name=args.run_name,
        config=store_config,
    )

    # If no explicit rollout path is provided, tuck it under the run directory
    if args.log_rollouts is None:
        args.log_rollouts = store.dir / "rollouts.jsonl"

    log_fh: Optional[TextIO] = None
    if args.log_rollouts:
        args.log_rollouts.parent.mkdir(parents=True, exist_ok=True)
        log_fh = args.log_rollouts.open("w", encoding="utf-8")

    trainer = SelfPlayTrainer(device=device, lr=args.lr, max_ticks=args.max_ticks, rollout_log=log_fh)
    results: List[EpisodeResult] = []

    with store or contextlib.nullcontext() as active_store:  # type: ignore[attr-defined]
        for ep in range(1, args.episodes + 1):
            result = trainer.run_episode(ep)
            results.append(result)
            print(
                f"[episode {ep:03d}] winner={result.winner or 'none':>4} "
                f"ticks={result.ticks:4d} orders={result.total_orders:3d} "
                f"loss={result.loss_value:.3f}"
            )
            if active_store is not None:
                active_store.log(
                    {
                        "event": "episode_end",
                        "episode": ep,
                        "winner": result.winner,
                        "ticks": result.ticks,
                        "orders": result.total_orders,
                        "loss": result.loss_value,
                    }
                )

        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.policy.net.state_dict(), args.save)
            print(f"Saved policy to {args.save}")

        if active_store is not None:
            active_store.save_checkpoint(trainer.policy.net.state_dict(), "final.pt")

        if log_fh:
            log_fh.close()
        if store:
            store.close()


if __name__ == "__main__":
    main()
