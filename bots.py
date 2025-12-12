#!/usr/bin/env python3
from __future__ import annotations

import random
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import SIM_CONFIG
from world import World, StarSystem, Order, FACTION_NAMES, TickSummary

try:
    import torch
except ImportError:  # torch is optional at runtime
    torch = None


@dataclass
class FactionPersonality:
    id: str
    name: str
    aggression: float  # likes attacking enemies
    expansionism: float  # likes unclaimed space
    greed: float  # likes high-value systems
    caution: float  # avoids dangerous fights


@dataclass
class FactionMemory:
    id: str
    battles_won: int = 0
    battles_lost: int = 0
    expansions: int = 0
    confidence: float = 0.5  # 0..1
    war_exhaustion: float = 0.0  # 0..~5


@dataclass
class FactionGoal:
    kind: str  # "attack" | "expand"
    target_system: Optional[int]
    expires_at: int  # world.tick when goal should be reconsidered


# Global attempts per faction per tick (everyone same)
ATTEMPTS_PER_TICK: int = int(SIM_CONFIG.get("attempts_per_tick", 4))
# How long without history events before bots get more aggressive
STAGNATION_TICKS: int = int(SIM_CONFIG.get("stagnation_ticks", 400))
# Minimum idle strength before we bother launching an attack fleet
MIN_LAUNCH_STRENGTH: float = float(SIM_CONFIG.get("min_launch_strength", 1.0))
# Minimum strength needed to flip a neutral (kept in sync with world-side capture rules)
MIN_CAPTURE_STRENGTH: float = float(SIM_CONFIG.get("min_capture_strength", 3.0))
# Overextension threshold (mirror world.py)
OVEREXTENSION_THRESHOLD: int = int(SIM_CONFIG.get("overextension_threshold", 15))
# Penalty for colonizer throughput/reserve per system beyond threshold
COLONIZER_OVEREXT_ORDER_PENALTY: float = float(
    SIM_CONFIG.get("colonizer_overextension_order_penalty", 0.1)
)
COLONIZER_OVEREXT_RESERVE_PER_SYSTEM: float = float(
    SIM_CONFIG.get("colonizer_overextension_reserve_per_system", 1.5)
)
# Bias toward using frontline systems as origins
FRONTLINE_ORIGIN_BONUS: float = float(SIM_CONFIG.get("frontline_origin_bonus", 0.6))
# Extra weight for origins that sit on the path toward the current goal
GOAL_PATH_BONUS: float = float(SIM_CONFIG.get("goal_path_bonus", 0.8))
# Max reachable hop distance for issuing an order (caps endless map crawls)
MAX_REACH_HOPS: int = int(SIM_CONFIG.get("max_reach_hops", 8))

# Colonization tuning
COLONIZER_RESERVE_STRENGTH: float = float(SIM_CONFIG.get("colonizer_reserve_strength", 6.0))
COLONIZER_STRENGTH_PER_ORDER: float = float(
    SIM_CONFIG.get("colonizer_strength_per_order", max(3.0, MIN_CAPTURE_STRENGTH))
)

# Max number of dynamically spawned rebel factions
MAX_DYNAMIC_FACTIONS: int = int(SIM_CONFIG.get("max_dynamic_factions", 6))

# League (coalition of independent worlds)
LEAGUE_FACTION_ID = "L"

# Behavior cloning logging (for later supervised training from heuristic play)
BC_LOG_PATH = SIM_CONFIG.get("bc_log_path")
BC_LOG_EVERY_N = int(SIM_CONFIG.get("bc_log_every_n_ticks", 0))  # 0 disables
_BC_LOG_HANDLE = None

# Minimal neural-policy helpers (shared with training scripts)
def _nn_idle_strength_by_system(world: World, faction: str) -> Dict[int, float]:
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


def _nn_enemy_strength_by_system(world: World, faction: str) -> Dict[int, float]:
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


def _nn_build_candidates(world: World, faction: str, remaining: Dict[int, float], used_origins: set[int]):
    from world import GARRISON_MAX, CAPTURE_TICKS  # late import to avoid circular deps during typing

    candidates = []
    enemy_strength = _nn_enemy_strength_by_system(world, faction)
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
                min(1.0, strength_here / 20.0),
                sys.value / 10.0,
                target.value / 10.0,
                1.0 if owner == faction else 0.0,
                1.0 if owner is None else 0.0,
                1.0 if (owner is not None and owner != faction) else 0.0,
                min(1.0, target.garrison / max(1.0, GARRISON_MAX)),
                min(1.0, target.heat / 3.0),
                target.stability if target.stability is not None else 0.0,
                min(1.0, enemy_here / 20.0),
                occ_progress,
                occ_ours,
            ]
            candidates.append(
                {
                    "order": Order(
                        faction=faction,
                        origin_id=sys_id,
                        target_id=nid,
                        reason="nn_policy",
                    ),
                    "features": features,
                    "origin_id": sys_id,
                }
            )

    candidates.append({"order": None, "features": [0.0] * 12, "origin_id": None})
    return candidates


class _RuntimePolicy:
    """Lightweight runtime policy wrapper for inference-time orders."""

    def __init__(self, weights_path: str):
        if torch is None:
            raise RuntimeError("PyTorch not available; cannot load neural policy.")
        from train_selfplay import PolicyNet  # avoid import if torch missing

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyNet().to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()

    def act(self, world: World, faction: str, max_orders: int = ATTEMPTS_PER_TICK) -> List[Order]:
        remaining = _nn_idle_strength_by_system(world, faction)
        used_origins: set[int] = set()
        chosen: List[Order] = []

        for _ in range(max_orders):
            candidates = _nn_build_candidates(world, faction, remaining, used_origins)
            if not candidates:
                break
            feats = torch.tensor([c["features"] for c in candidates], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.net(feats)
                probs = torch.softmax(logits, dim=0)
                idx = torch.multinomial(probs, num_samples=1).item()
            selected = candidates[idx]
            if selected["order"] is None:
                break
            chosen.append(selected["order"])
            origin_id = selected["origin_id"]
            if origin_id is not None:
                used_origins.add(origin_id)
                remaining.pop(origin_id, None)
        return chosen


# Singleton runtime policy (optional)
_RUNTIME_POLICY: Optional[_RuntimePolicy] = None

# Optional override for order generation (e.g., neural policy inference)
_CUSTOM_ORDER_FN: Optional[Callable[[World], List[Order]]] = None

# Optional neural policy settings
USE_NEURAL_POLICY: bool = bool(SIM_CONFIG.get("use_neural_policy", False))
NEURAL_POLICY_WEIGHTS: Optional[str] = SIM_CONFIG.get("neural_policy_weights")
_WEIGHTS_PATH: Optional[Path] = Path(NEURAL_POLICY_WEIGHTS) if NEURAL_POLICY_WEIGHTS else None
_WEIGHTS_MTIME: Optional[float] = None
# Counter + set for dynamic rebel factions
DYNAMIC_FACTION_COUNTER = 0
DYNAMIC_FACTION_IDS: set[str] = set()

# Colors for factions
FACTION_COLORS: Dict[str, str] = {}
REBEL_COLOR_PALETTE = [
    "#ff79c6",
    "#50fa7b",
    "#bd93f9",
    "#ffb86c",
]
LEAGUE_COLOR = "#7fffd4"


# Load AI config from JSON
_FACTION_CFG: Dict[str, dict] = SIM_CONFIG.get("factions", {})

FACTION_PERSONALITIES: Dict[str, FactionPersonality] = {}
FACTION_MEMORY: Dict[str, FactionMemory] = {}
FACTION_GOALS: Dict[str, FactionGoal] = {}
EXPANSION_ATTEMPTS: Dict[str, int] = {}  # kept for compatibility, not really used
COLONIZER_TARGETS: Dict[str, int] = {}

# Default colors for initial factions
_DEFAULT_COLORS = {
    "E": "#ff5555",
    "P": "#f1fa8c",
    "T": "#8be9fd",
    "M": "#a6e22e",
    "S": "#ff79c6",
}


def _build_initial_state() -> None:
    """Reset faction state back to config defaults (used on sim restart)."""
    FACTION_NAMES.clear()
    for fid, cfg in _FACTION_CFG.items():
        FACTION_NAMES[fid] = cfg.get("name", fid)

    FACTION_PERSONALITIES.clear()
    FACTION_MEMORY.clear()
    EXPANSION_ATTEMPTS.clear()
    COLONIZER_TARGETS.clear()
    FACTION_COLORS.clear()

    for fid, name in FACTION_NAMES.items():
        cfg = _FACTION_CFG.get(fid, {})
        base_attempts = int(cfg.get("base_attempts", ATTEMPTS_PER_TICK))
        p_cfg = cfg.get("personality", {})

        personality = FactionPersonality(
            id=fid,
            name=name,
            aggression=float(p_cfg.get("aggression", 0.5)),
            expansionism=float(p_cfg.get("expansionism", 0.5)),
            greed=float(p_cfg.get("greed", 0.5)),
            caution=float(p_cfg.get("caution", 0.5)),
        )

        FACTION_PERSONALITIES[fid] = personality
        FACTION_MEMORY[fid] = FactionMemory(id=fid)
        EXPANSION_ATTEMPTS[fid] = base_attempts

        FACTION_COLORS[fid] = _DEFAULT_COLORS.get(fid, "#ffffff")


_build_initial_state()


def register_new_faction(parent_id: str | None = None, parent_label: str = "Rebels") -> str:
    """
    Create a new faction dynamically (used by rebellions).
    If we've hit MAX_DYNAMIC_FACTIONS, reuse an existing rebel faction instead
    of creating a brand new one. This prevents unbounded faction explosion.
    """
    global DYNAMIC_FACTION_COUNTER

    # If we already have some rebel factions and hit the cap -> reuse one
    if DYNAMIC_FACTION_IDS and len(DYNAMIC_FACTION_IDS) >= MAX_DYNAMIC_FACTIONS:
        return random.choice(list(DYNAMIC_FACTION_IDS))

    # Otherwise, spawn a genuinely new rebel faction
    DYNAMIC_FACTION_COUNTER += 1
    fid = f"R{DYNAMIC_FACTION_COUNTER}"
    name = f"{parent_label} Rebels {DYNAMIC_FACTION_COUNTER}"

    parent_p = FACTION_PERSONALITIES.get(parent_id) if parent_id else None
    personality = FactionPersonality(
        id=fid,
        name=name,
        aggression=float(parent_p.aggression) if parent_p else 0.6,
        expansionism=float(parent_p.expansionism) if parent_p else 0.7,
        greed=float(parent_p.greed) if parent_p else 0.6,
        caution=float(parent_p.caution) if parent_p else 0.5,
    )
    FACTION_PERSONALITIES[fid] = personality
    FACTION_MEMORY[fid] = FactionMemory(id=fid)
    EXPANSION_ATTEMPTS[fid] = ATTEMPTS_PER_TICK

    # Track dynamic rebel factions
    DYNAMIC_FACTION_IDS.add(fid)

    # Color for this rebel faction
    color = REBEL_COLOR_PALETTE[
        (DYNAMIC_FACTION_COUNTER - 1) % len(REBEL_COLOR_PALETTE)
    ]
    FACTION_COLORS[fid] = color

    # Update global faction names (shared with world)
    FACTION_NAMES[fid] = name
    return fid


def reset_factions() -> None:
    """Reset dynamic rebel state and rebuild faction data from config."""
    global DYNAMIC_FACTION_COUNTER, DYNAMIC_FACTION_IDS
    DYNAMIC_FACTION_COUNTER = 0
    DYNAMIC_FACTION_IDS = set()
    _build_initial_state()
    FACTION_GOALS.clear()
    COLONIZER_TARGETS.clear()


def set_custom_order_fn(fn: Optional[Callable[[World], List[Order]]]) -> None:
    """
    Plug in an external order generator (e.g., neural net inference).
    If None, fall back to the built-in heuristic AI.
    """
    global _CUSTOM_ORDER_FN
    _CUSTOM_ORDER_FN = fn


def _load_neural_policy(strict: bool) -> None:
    """
    Load weights into the runtime policy. If strict, raise on any failure.
    """
    global _RUNTIME_POLICY, _WEIGHTS_MTIME
    if torch is None:
        if strict:
            raise RuntimeError("Neural policy requested but PyTorch is not available.")
        return
    if _WEIGHTS_PATH is None:
        if strict:
            raise RuntimeError("Neural policy requested but no weights path provided.")
        return
    if not _WEIGHTS_PATH.exists():
        if strict:
            raise RuntimeError(f"Neural policy weights not found: {_WEIGHTS_PATH}")
        return
    try:
        _RUNTIME_POLICY = _RuntimePolicy(str(_WEIGHTS_PATH))
        _WEIGHTS_MTIME = _WEIGHTS_PATH.stat().st_mtime
    except Exception as exc:
        if strict:
            raise
        print(f"Failed to load neural policy '{_WEIGHTS_PATH}': {exc}")
        _RUNTIME_POLICY = None
        return

    def _nn_order_fn(world: World) -> List[Order]:
        if _RUNTIME_POLICY is None:
            raise RuntimeError("Neural policy unavailable; refusing to fall back.")
        orders: List[Order] = []
        for fid in FACTION_NAMES.keys():
            orders.extend(_RUNTIME_POLICY.act(world, fid))
        return orders

    set_custom_order_fn(_nn_order_fn)
    print(f"Neural policy enabled from weights: {_WEIGHTS_PATH}")


def enable_neural_policy_from_config() -> None:
    """
    If configured, load a neural policy and route generate_orders through it.
    This is strict: when USE_NEURAL_POLICY is true we refuse to fall back.
    """
    if not USE_NEURAL_POLICY:
        return
    _load_neural_policy(strict=True)


def ensure_league_faction(world: World) -> None:
    """Add the league faction for independent worlds if missing."""
    if LEAGUE_FACTION_ID in FACTION_NAMES:
        world.faction_build_stock.setdefault(LEAGUE_FACTION_ID, 0.0)
        return

    name = "League of Independents"
    FACTION_NAMES[LEAGUE_FACTION_ID] = name
    FACTION_PERSONALITIES[LEAGUE_FACTION_ID] = FactionPersonality(
        id=LEAGUE_FACTION_ID,
        name=name,
        aggression=0.65,
        expansionism=0.35,
        greed=0.6,
        caution=0.65,
    )
    FACTION_MEMORY[LEAGUE_FACTION_ID] = FactionMemory(id=LEAGUE_FACTION_ID)
    EXPANSION_ATTEMPTS[LEAGUE_FACTION_ID] = ATTEMPTS_PER_TICK
    FACTION_COLORS[LEAGUE_FACTION_ID] = LEAGUE_COLOR
    world.faction_build_stock.setdefault(LEAGUE_FACTION_ID, 0.0)


# ---------- Helpers ----------

# Try to enable neural policy immediately if configured
enable_neural_policy_from_config()


def _owned_by_faction(world: World) -> Dict[str, List[StarSystem]]:
    owned_by: Dict[str, List[StarSystem]] = {f: [] for f in FACTION_NAMES.keys()}
    for sys in world.systems.values():
        if sys.owner in owned_by:
            owned_by[sys.owner].append(sys)
    return owned_by


def _idle_fleets_by_system(world: World) -> Dict[str, Dict[int, int]]:
    """Count idle (not enroute) fleets per system for each faction."""
    idle: Dict[str, Dict[int, int]] = {f: {} for f in FACTION_NAMES.keys()}
    for fl in world.fleets.values():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        sys_map = idle.setdefault(fl.owner, {})
        sys_map[fl.system_id] = sys_map.get(fl.system_id, 0) + 1
    return idle


def _idle_strength_by_system(world: World) -> Dict[str, Dict[int, float]]:
    """Total idle strength per system per faction."""
    strengths: Dict[str, Dict[int, float]] = {f: {} for f in FACTION_NAMES.keys()}
    for fl in world.fleets.values():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        sys_map = strengths.setdefault(fl.owner, {})
        sys_map[fl.system_id] = sys_map.get(fl.system_id, 0.0) + fl.strength
    return strengths


def _is_frontline(world: World, faction: str, system: StarSystem) -> bool:
    """A system is frontline if it has at least one neighbor not owned by the faction."""
    for nid in system.neighbors:
        neigh = world.systems.get(nid)
        if neigh and neigh.owner != faction:
            return True
    return False


def _kind_bonus(sys: StarSystem) -> float:
    if sys.kind == "relic":
        return 0.8
    if sys.kind == "forge":
        return 0.5
    if sys.kind == "hive":
        return 0.3
    return 0.0


def _system_econ_value(sys: StarSystem) -> float:
    # For debug-only economy display (simpler than world-side version)
    base = float(sys.value)
    if sys.kind == "relic":
        base *= 1.8
    elif sys.kind == "forge":
        base *= 1.4
    elif sys.kind == "hive":
        base *= 1.2
    return base


def _compute_econ_power(world: World) -> Dict[str, float]:
    econ_power = {f: 0.0 for f in FACTION_NAMES.keys()}
    for sys in world.systems.values():
        owner = sys.owner
        if owner in econ_power:
            econ_power[owner] += _system_econ_value(sys)
    return econ_power


def _bc_log_enabled() -> bool:
    return BC_LOG_PATH is not None and BC_LOG_EVERY_N > 0


def _bc_log_world(world: World, orders: List[Order]) -> None:
    """Append a compact JSONL record of the heuristic decisions for behavior cloning."""
    global _BC_LOG_HANDLE
    if not _bc_log_enabled():
        return
    if world.tick % BC_LOG_EVERY_N != 0:
        return
    try:
        if _BC_LOG_HANDLE is None:
            path = Path(BC_LOG_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            _BC_LOG_HANDLE = path.open("a", encoding="utf-8")
        systems = [
            {
                "id": s.id,
                "owner": s.owner,
                "value": s.value,
                "kind": s.kind,
                "stability": s.stability,
                "garrison": s.garrison,
                "heat": s.heat,
                "neighbors": list(s.neighbors),
                "occupation_faction": s.occupation_faction,
                "occupation_progress": s.occupation_progress,
            }
            for s in world.systems.values()
        ]
        fleets = [
            {
                "id": f.id,
                "owner": f.owner,
                "system_id": f.system_id,
                "strength": f.strength,
                "enroute_from": f.enroute_from,
                "enroute_to": f.enroute_to,
                "eta": f.eta,
            }
            for f in world.fleets.values()
            if f.strength > 0
        ]
        record = {
            "tick": world.tick,
            "orders": [
                {
                    "faction": o.faction,
                    "origin": o.origin_id,
                    "target": o.target_id,
                    "reason": getattr(o, "reason", None),
                }
                for o in orders
            ],
            "systems": systems,
            "fleets": fleets,
        }
        _BC_LOG_HANDLE.write(json.dumps(record, separators=(",", ":")) + "\n")
        _BC_LOG_HANDLE.flush()
    except Exception as exc:
        print(f"[bc_log] failed to log tick {world.tick}: {exc}")


def _pick_goal_target(world: World, faction: str) -> Optional[int]:
    """
    Choose a strategic target for this faction:
    prefer enemy-owned and high-value/special systems.
    """
    candidates: List[tuple[float, int]] = []
    for sys in world.systems.values():
        if sys.owner == faction:
            continue
        score = (sys.value / 10.0) + _kind_bonus(sys)
        score += 0.3 if sys.owner else 0.15  # enemy > neutral
        score += random.uniform(-0.05, 0.05)
        candidates.append((score, sys.id))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _goal_for_faction(world: World, faction: str) -> Optional[FactionGoal]:
    goal = FACTION_GOALS.get(faction)
    if goal:
        target = world.systems.get(goal.target_system) if goal.target_system is not None else None
        if goal.expires_at > world.tick and target and target.owner != faction:
            return goal
    target_id = _pick_goal_target(world, faction)
    if target_id is None:
        FACTION_GOALS[faction] = FactionGoal(kind="attack", target_system=None, expires_at=world.tick + 60)
        return FACTION_GOALS[faction]
    target_owner = world.systems[target_id].owner
    kind = "expand" if target_owner is None else "attack"
    FACTION_GOALS[faction] = FactionGoal(kind=kind, target_system=target_id, expires_at=world.tick + 80)
    return FACTION_GOALS[faction]


def _next_hop_toward(world: World, start_id: int, target_id: int) -> Optional[int]:
    """Return neighbor of start_id that is on a shortest path toward target_id."""
    if start_id == target_id:
        return None
    adj = {sid: sys.neighbors for sid, sys in world.systems.items()}
    seen = {start_id}
    q = deque([(start_id, None)])
    parents: Dict[int, Optional[int]] = {start_id: None}
    while q:
        node, _ = q.popleft()
        if node == target_id:
            break
        for neigh in adj.get(node, []):
            if neigh in seen:
                continue
            seen.add(neigh)
            parents[neigh] = node
            q.append((neigh, node))
    if target_id not in parents:
        return None
    # walk back from target to find the neighbor after start
    cur = target_id
    prev = parents[cur]
    while prev is not None and prev != start_id:
        cur = prev
        prev = parents.get(cur)
    return cur if prev == start_id else None


def _next_hop_owned_path(world: World, faction: str, start_id: int, target_id: int) -> Optional[int]:
    """
    Move through owned territory toward target. If target is not owned, we only step
    into the first non-owned system adjacent to our owned region.
    If an enemy system has no defending fleets in orbit, we treat it as passable
    for pathing purposes (raiding through “open” territory).
    """
    if start_id == target_id:
        return None
    target_owner = world.systems[target_id].owner
    parents: Dict[int, Optional[int]] = {start_id: None}
    q = deque([start_id])
    defended: set[int] = {
        fl.system_id
        for fl in world.fleets.values()
        if fl.eta == 0 and fl.system_id is not None and fl.strength > 0 and fl.owner != faction
    }

    def reconstruct(end: int) -> Optional[int]:
        cur = end
        prev = parents[cur]
        while prev is not None and prev != start_id:
            cur = prev
            prev = parents.get(cur)
        return cur if prev == start_id else None

    while q:
        node = q.popleft()
        if target_owner == faction and node == target_id:
            return reconstruct(node)
        if target_owner != faction and target_id in world.systems[node].neighbors:
            return reconstruct(node)
        for neigh in world.systems[node].neighbors:
            if neigh in parents:
                continue
            neigh_owner = world.systems[neigh].owner
            # allow stepping through our space or neutral/independent space.
            # Enemy space is also passable if it has no defending fleets in orbit.
            if neigh_owner not in (faction, None) and neigh in defended and neigh != target_id:
                continue
            parents[neigh] = node
            q.append(neigh)
    return None


def _owned_path_distance(world: World, faction: str, start_id: int, target_id: int) -> Optional[int]:
    """
    Shortest hop distance through owned territory toward target.
    If target is not owned, we allow the final step into the target from an owned or neutral neighbor.
    Enemy systems with no defending fleets in orbit are treated as passable for routing.
    """
    if start_id == target_id:
        return 0

    target_owner = world.systems[target_id].owner
    defended: set[int] = {
        fl.system_id
        for fl in world.fleets.values()
        if fl.eta == 0 and fl.system_id is not None and fl.strength > 0 and fl.owner != faction
    }
    q = deque([(start_id, 0)])
    seen = {start_id}
    while q:
        node, dist = q.popleft()
        for neigh in world.systems[node].neighbors:
            if neigh in seen:
                continue
            neigh_owner = world.systems[neigh].owner

            if target_owner == faction:
                if neigh_owner not in (faction, None) and neigh in defended and neigh != target_id:
                    continue
            else:
                if neigh != target_id and neigh_owner not in (faction, None) and neigh in defended:
                    continue

            if neigh == target_id:
                return dist + 1
            if neigh_owner in (faction, None) or neigh not in defended:
                seen.add(neigh)
                q.append((neigh, dist + 1))
    return None


def _graph_distance(world: World, start_id: int, target_id: int) -> Optional[int]:
    """Unrestricted shortest-path distance (any ownership)."""
    if start_id == target_id:
        return 0
    seen = {start_id}
    q = deque([(start_id, 0)])
    while q:
        node, dist = q.popleft()
        for neigh in world.systems[node].neighbors:
            if neigh in seen:
                continue
            if neigh == target_id:
                return dist + 1
            seen.add(neigh)
            q.append((neigh, dist + 1))
    return None


def _score_target(world: World, faction: str, target: StarSystem) -> float:
    """
    Compute a desirability score for attacking/expanding into target for this faction.
    Higher = more attractive.
    """
    p = FACTION_PERSONALITIES[faction]
    owned_count = sum(1 for sys in world.systems.values() if sys.owner == faction)
    score = 0.0

    # Prefer high-value systems
    score += p.greed * (target.value / 10.0)

    # Special worlds get a bonus
    score += _kind_bonus(target)

    if target.owner is None:
        score += p.expansionism * 1.0
        # When tiny, strongly prefer nearby unclaimed worlds to grow a base
        if owned_count <= 3:
            score += 2.0
        elif owned_count <= 5:
            score += 1.0
    elif target.owner != faction:
        score += p.aggression * 1.5
        score -= p.caution * 0.3
        # Small empires should temper aggression until they have a foothold
        if owned_count <= 3:
            score -= 0.6
    else:
        score -= 999.0

    # Slight bias against very unstable systems (rebellion risk)
    score -= (1.0 - target.stability) * 0.2

    # Tiny jitter
    score += random.uniform(-0.05, 0.05)
    return score


def _choose_target(world: World, faction: str, origin: StarSystem) -> int | None:
    if not origin.neighbors:
        return None

    best_id = None
    best_score = float("-inf")
    owned_neighbor: int | None = None

    for nid in origin.neighbors:
        neigh = world.systems[nid]
        if neigh.owner == faction:
            owned_neighbor = nid  # fallback for repositioning
            continue

        s = _score_target(world, faction, neigh)
        if s > best_score:
            best_score = s
            best_id = nid

    # If no enemy/neutral neighbors, allow moving through owned space
    if best_id is None:
        return owned_neighbor
    return best_id


def _choose_target_with_goal(
    world: World, faction: str, origin: StarSystem, goal: Optional[FactionGoal]
) -> int | None:
    """
    Prefer moving toward a strategic goal if we have one; otherwise fall back.
    Movement is constrained to own territory until stepping into the first non-owned system.
    """
    if goal and goal.target_system is not None:
        hop = _next_hop_owned_path(world, faction, origin.id, goal.target_system)
        if hop is not None:
            # Only move if the hop meaningfully reduces distance to the goal to avoid ping-pong.
            curr_dist = _owned_path_distance(world, faction, origin.id, goal.target_system)
            next_dist = _owned_path_distance(world, faction, hop, goal.target_system)
            if curr_dist is None or (next_dist is not None and next_dist < curr_dist):
                return hop
            # If it's a lateral move with no progress, stay put instead of oscillating.
            return None

    target_id = _choose_target(world, faction, origin)
    if target_id is None:
        return None

    # If we'd only move to another owned backline system, don't oscillate—just hold.
    target_sys = world.systems[target_id]
    if (
        target_sys.owner == faction
        and not _is_frontline(world, faction, origin)
        and not _is_frontline(world, faction, target_sys)
    ):
        return None

    return target_id


def _generate_colonizer_orders(
    world: World,
    faction: str,
    idle_strength_map: Dict[str, Dict[int, float]],
    owned_count: int,
) -> List[Order]:
    """
    Dispatch small claim fleets to nearby neutral systems, scaling with available idle strength
    and stopping once a reserve threshold is reached.
    """
    local_strength = dict(idle_strength_map.get(faction, {}))
    total_idle = sum(local_strength.values())

    extra_over = max(0, owned_count - OVEREXTENSION_THRESHOLD)
    dynamic_reserve = COLONIZER_RESERVE_STRENGTH + extra_over * COLONIZER_OVEREXT_RESERVE_PER_SYSTEM
    if total_idle <= dynamic_reserve:
        return []

    available = total_idle - dynamic_reserve
    base_orders = int(available // COLONIZER_STRENGTH_PER_ORDER)
    penalty_factor = 1.0 + extra_over * COLONIZER_OVEREXT_ORDER_PENALTY
    max_orders = int(base_orders / penalty_factor)
    if max_orders <= 0:
        return []

    min_needed = max(MIN_LAUNCH_STRENGTH, MIN_CAPTURE_STRENGTH)

    # If we already have a colonizer target, stick to it while it's still neutral and reachable
    pinned_target = COLONIZER_TARGETS.get(faction)
    pinned_origin = None
    pinned_dist = None
    if pinned_target is not None:
        tgt_sys = world.systems.get(pinned_target)
        if not tgt_sys or tgt_sys.owner is not None:
            COLONIZER_TARGETS.pop(faction, None)
            pinned_target = None
        else:
            for origin_id, strength in local_strength.items():
                if strength < min_needed:
                    continue
                dist = _owned_path_distance(world, faction, origin_id, pinned_target)
                if dist is None:
                    continue
                if pinned_dist is None or dist < pinned_dist or (
                    dist == pinned_dist and strength > local_strength.get(pinned_origin, 0.0)
                ):
                    pinned_dist = dist
                    pinned_origin = origin_id
            if pinned_origin is None:
                COLONIZER_TARGETS.pop(faction, None)
                pinned_target = None

    neutral_targets: List[tuple[int, float, int, int]] = []
    for sys in world.systems.values():
        if sys.owner is not None:
            continue
        best_origin = None
        best_dist = None
        for origin_id, strength in local_strength.items():
            if strength < min_needed:
                continue
            dist = _owned_path_distance(world, faction, origin_id, sys.id)
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_origin = origin_id
            elif dist == best_dist and strength > local_strength.get(best_origin, 0.0):
                best_origin = origin_id
        if best_origin is None or best_dist is None:
            continue
        neutral_targets.append((best_dist, -float(sys.value), sys.id, best_origin))

    if not neutral_targets:
        return []

    neutral_targets.sort(key=lambda t: (t[0], t[1]))

    # Stickiness: if we have a pinned target, put it first so we finish it before new ones
    if pinned_target is not None:
        neutral_targets = [t for t in neutral_targets if t[2] == pinned_target] + [
            t for t in neutral_targets if t[2] != pinned_target
        ]
    else:
        # If no pinned target, set one from the best available neutral
        top = neutral_targets[0]
        COLONIZER_TARGETS[faction] = top[2]
        pinned_target = top[2]

    # Re-evaluate pinned data if we just set it
    if pinned_target is not None and pinned_origin is None:
        for dist, _neg, tid, origin_id in neutral_targets:
            if tid != pinned_target:
                continue
            pinned_origin = origin_id
            pinned_dist = dist
            break

    orders: List[Order] = []
    remaining = available
    for dist, _neg_value, target_id, origin_id in neutral_targets:
        if len(orders) >= max_orders:
            break
        origin_strength = local_strength.get(origin_id, 0.0)
        if origin_strength < min_needed:
            continue
        hop = _next_hop_owned_path(world, faction, origin_id, target_id)
        if hop is None:
            # If we're adjacent, go straight in instead of oscillating
            if dist == 1:
                hop = target_id
            else:
                continue
        orders.append(
            Order(
                faction=faction,
                origin_id=origin_id,
                target_id=hop,
                reason=f"colonize neutral #{target_id} (dist {dist})",
            )
        )
        local_strength[origin_id] = origin_strength - COLONIZER_STRENGTH_PER_ORDER
        if local_strength[origin_id] <= 0:
            local_strength.pop(origin_id, None)
        remaining -= COLONIZER_STRENGTH_PER_ORDER
        if remaining < COLONIZER_STRENGTH_PER_ORDER:
            break

    return orders


# ---------- Public bot API ----------


def generate_orders(world: World) -> List[Order]:
    """
    Decide what each faction wants to try this tick.
    We try all reasonable origins (within reach), rather than a hard per-tick cap.
    """
    if USE_NEURAL_POLICY and _CUSTOM_ORDER_FN is None:
        raise RuntimeError("Neural policy enabled but no order generator is active.")
    if _CUSTOM_ORDER_FN is not None:
        return _CUSTOM_ORDER_FN(world)

    orders: List[Order] = []

    # If nothing meaningful happened for a long while, loosen the safety rails
    last_event_tick = world.history[-1].tick if world.history else 0
    stagnation = world.tick - last_event_tick > STAGNATION_TICKS

    owned_by = _owned_by_faction(world)
    systems_per_faction = {f: len(systems) for f, systems in owned_by.items()}
    idle_map = _idle_fleets_by_system(world)
    idle_strength_map = _idle_strength_by_system(world)
    # per-system total idle strength by owner
    system_strengths: Dict[int, Dict[str, float]] = {}
    for fl in world.fleets.values():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        sys_map = system_strengths.setdefault(fl.system_id, {})
        sys_map[fl.owner] = sys_map.get(fl.owner, 0.0) + fl.strength
    factions = list(owned_by.keys())
    random.shuffle(factions)

    dominant_faction: Optional[str] = None
    sorted_counts = sorted(systems_per_faction.items(), key=lambda kv: kv[1], reverse=True)
    if sorted_counts:
        top_faction, top_count = sorted_counts[0]
        second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        if top_count >= max(8, 2 * max(second_count, 1)):
            dominant_faction = top_faction

    for faction in factions:
        goal = _goal_for_faction(world, faction)
        is_dominant = faction == dominant_faction
        min_launch_threshold = MIN_LAUNCH_STRENGTH * (0.5 if is_dominant else 1.0)
        systems = []
        idle_counts = idle_map.get(faction, {})
        idle_strength_for = idle_strength_map.get(faction, {})

        # Proactive neutral claiming scaled by available idle strength
        colonizer_orders = _generate_colonizer_orders(
            world,
            faction,
            idle_strength_map,
            systems_per_faction.get(faction, 0),
        )
        orders.extend(colonizer_orders)

        frontline_filtered = []
        for sys in owned_by[faction]:
            idle_here = idle_counts.get(sys.id, 0)
            strength_here = idle_strength_for.get(sys.id, 0.0)
            if idle_here == 0 or strength_here <= 0.0:
                continue
            # Avoid stripping a frontline system of its only fleet unless we are stagnating
            if (
                _is_frontline(world, faction, sys)
                and strength_here < min_launch_threshold
                and not (stagnation or is_dominant)
            ):
                # allow if there is at least one undefended or neutral neighbor to grab
                has_soft_neighbor = False
                for nid in sys.neighbors:
                    neigh = world.systems.get(nid)
                    if not neigh:
                        continue
                    if neigh.owner != faction:
                        enemy_idle = sum(
                            strength
                            for owner, strength in system_strengths.get(nid, {}).items()
                            if owner != faction
                        )
                        if enemy_idle == 0:
                            has_soft_neighbor = True
                            break
                if not has_soft_neighbor:
                    continue
            frontline_filtered.append(sys)

        # Also allow stacked backline systems to send ships forward
        backline_stacks = [
            sys
            for sys in owned_by[faction]
            if not _is_frontline(world, faction, sys)
            and idle_strength_for.get(sys.id, 0.0) >= min_launch_threshold
        ]

        # Deduplicate while preserving order preference (frontline first)
        seen_ids = set()
        systems = []
        for sys in frontline_filtered + backline_stacks:
            if sys.id in seen_ids:
                continue
            seen_ids.add(sys.id)
            systems.append(sys)

        # If we have no candidates after the guard, allow fallback (so 1-system empires can move)
        if not systems:
            systems = [
                sys for sys in owned_by[faction]
                if idle_counts.get(sys.id, 0) > 0
            ]

        if not systems:
            continue

        # Defensive reactions: try to break sieges on owned systems
        def _is_besieged(s: StarSystem) -> bool:
            if s.occupation_faction and s.occupation_faction != faction:
                return True
            for fl in world.fleets.values():
                if (
                    fl.system_id == s.id
                    and fl.strength > 0
                    and fl.eta == 0
                    and fl.owner != faction
                ):
                    return True
            return False

        defense_orders: List[Order] = []
        threatened = [s for s in owned_by[faction] if _is_besieged(s)]
        rally_orders: List[Order] = []

        for threat in threatened:
            enemy_strength = sum(
                fl.strength
                for fl in world.fleets.values()
                if fl.system_id == threat.id and fl.eta == 0 and fl.owner != faction and fl.strength > 0
            )

            # Best immediate neighbor defense
            neighbor_candidates = []
            for nid in threat.neighbors:
                neigh = world.systems[nid]
                if neigh.owner != faction:
                    continue
                idle_here = idle_counts.get(neigh.id, 0)
                strength_here = idle_strength_for.get(neigh.id, 0.0)
                if idle_here > 0 and strength_here > 0:
                    neighbor_candidates.append((strength_here, neigh))
            neighbor_candidates.sort(key=lambda t: t[0], reverse=True)
            best_neighbor = neighbor_candidates[0] if neighbor_candidates else None

            # Global rally: allow distant stacks to move toward the siege if locals are absent/weak
            best_rally = None
            best_score = 0.0
            for sys in owned_by[faction]:
                if sys.id == threat.id:
                    continue
                idle_here = idle_counts.get(sys.id, 0)
                if idle_here == 0:
                    continue
                rally_strength = idle_strength_for.get(sys.id, 0.0)
                if rally_strength <= 0:
                    continue
                dist = _owned_path_distance(world, faction, sys.id, threat.id)
                if dist is None or dist <= 0:
                    continue
                score = rally_strength / dist
                if score > best_score:
                    best_score = score
                    best_rally = (sys, rally_strength, dist)

            # If neighbor strong enough, defend directly; otherwise rally the best distant stack
            if best_neighbor and (enemy_strength == 0 or best_neighbor[0] >= 0.8 * enemy_strength):
                origin_strength, origin_sys = best_neighbor
                defense_orders.append(
                    Order(
                        faction=faction,
                        origin_id=origin_sys.id,
                        target_id=threat.id,
                        reason=(
                            f"reinforce besieged system #{threat.id} "
                            f"(enemy {enemy_strength:.1f} vs our {origin_strength:.1f})"
                        ),
                    )
                )
                if len(defense_orders) >= 3:
                    break
            elif best_rally:
                sys, rally_strength, dist = best_rally
                hop = _next_hop_owned_path(world, faction, sys.id, threat.id)
                if hop is not None:
                    rally_orders.append(
                        Order(
                            faction=faction,
                            origin_id=sys.id,
                            target_id=hop,
                            reason=(
                                f"rally toward besieged system #{threat.id} "
                                f"(enemy {enemy_strength:.1f} vs our {rally_strength:.1f}, dist {dist})"
                            ),
                        )
                    )

        orders.extend(defense_orders)
        orders.extend(rally_orders[:3])

        # Build weighted origins to bias toward strong stacks on goal paths/frontlines
        goal_target = goal.target_system if goal else None
        weighted_origins: List[tuple[StarSystem, float]] = []
        for sys in systems:
            idle_here = idle_counts.get(sys.id, 0)
            if idle_here == 0:
                continue
            my_str = idle_strength_map.get(faction, {}).get(sys.id, 0.0)
            # Keep very weak stacks at home unless we're stagnating
            if my_str < min_launch_threshold and not (stagnation or is_dominant):
                continue

            frontline = _is_frontline(world, faction, sys)
            best_neighbor_adv = 0.0
            for nid in sys.neighbors:
                neigh = world.systems.get(nid)
                if not neigh:
                    continue
                enemy_strength = 0.0
                for owner, strength in system_strengths.get(nid, {}).items():
                    if owner != faction:
                        enemy_strength += strength
                if enemy_strength > 0:
                    ratio = my_str / (enemy_strength + 1e-6)
                    best_neighbor_adv = max(best_neighbor_adv, ratio)
                elif neigh.owner != faction:
                    # Treat undefended enemy/neutral as juicy
                    best_neighbor_adv = max(best_neighbor_adv, my_str + 1.0)

            goal_bonus = 0.0
            if goal_target is not None:
                hop = _next_hop_owned_path(world, faction, sys.id, goal_target)
                if hop is not None:
                    goal_bonus = GOAL_PATH_BONUS

            adv_bonus = min(1.5, best_neighbor_adv)
            score = my_str * (1.0 + goal_bonus + (FRONTLINE_ORIGIN_BONUS if frontline else 0.0))
            score += 0.35 * adv_bonus
            weighted_origins.append((sys, score))

        # If everything was too weak, allow the strongest few to try (heavily de-weighted)
        if not weighted_origins and not stagnation:
            for sys in systems:
                my_str = idle_strength_map.get(faction, {}).get(sys.id, 0.0)
                if my_str <= 0:
                    continue
                weighted_origins.append((sys, 0.25 * my_str))

        weighted_origins.sort(key=lambda t: t[1], reverse=True)

        for origin, _ in weighted_origins:
            target_id = _choose_target_with_goal(world, faction, origin, goal)
            if target_id is None:
                continue

            reach = _graph_distance(world, origin.id, target_id)
            if reach is None or reach > MAX_REACH_HOPS:
                continue

            target_sys = world.systems[target_id]
            goal_desc = None
            if goal and goal.target_system is not None:
                goal_target_sys = world.systems.get(goal.target_system)
                goal_owner = None
                if goal_target_sys:
                    goal_owner = goal_target_sys.owner
                owner_label = (
                    FACTION_NAMES.get(goal_owner, "unclaimed") if goal_owner else "unclaimed"
                )
                goal_desc = f"{goal.kind} goal #{goal.target_system} ({owner_label})"

            if target_sys.owner == faction:
                action_reason = "reposition within territory"
                if goal_desc:
                    action_reason = f"reposition along path toward {goal_desc}"
            elif target_sys.owner is None:
                action_reason = "expand into neutral neighbor"
                if goal_desc:
                    action_reason += f" toward {goal_desc}"
            else:
                action_reason = f"attack neighbor held by {FACTION_NAMES.get(target_sys.owner, target_sys.owner)}"
                if goal_desc:
                    action_reason += f" toward {goal_desc}"

            orders.append(
                Order(
                    faction=faction,
                    origin_id=origin.id,
                    target_id=target_id,
                    reason=action_reason,
                )
            )

    _bc_log_world(world, orders)
    return orders


def update_bot_memory_and_personality(summary: TickSummary) -> None:
    """
    Update each faction's memory & personality based on the last tick.
    """
    for fid, mem in FACTION_MEMORY.items():
        won = summary.battles_won.get(fid, 0)
        lost = summary.battles_lost.get(fid, 0)
        exp = summary.expansions.get(fid, 0)

        mem.battles_won += won
        mem.battles_lost += lost
        mem.expansions += exp

        # Confidence: based on recent tick delta, with decay toward 0.5
        delta = (won + 0.5 * exp) - lost
        mem.confidence += 0.05 * delta
        mem.confidence += 0.01 * (0.5 - mem.confidence)  # decay toward neutral
        mem.confidence = max(0.0, min(1.0, mem.confidence))

        # War exhaustion: grows from losses and (a bit) wins, decays slowly
        mem.war_exhaustion *= 0.97
        mem.war_exhaustion += 0.07 * lost + 0.03 * won
        mem.war_exhaustion = max(0.0, min(5.0, mem.war_exhaustion))

        # Personality drift based on confidence and exhaustion
        p = FACTION_PERSONALITIES[fid]
        shift = mem.confidence - 0.5  # -0.5..+0.5

        # Confidence pushes aggression/caution
        p.aggression = max(0.1, min(1.0, p.aggression + 0.06 * shift))
        p.caution = max(0.1, min(1.0, p.caution - 0.06 * shift))

        # War exhaustion makes them more cautious, slightly less aggressive
        p.caution = max(0.1, min(1.0, p.caution + 0.02 * mem.war_exhaustion))
        p.aggression = max(0.1, min(1.0, p.aggression - 0.015 * mem.war_exhaustion))


# ---------- Debug helper ----------


def get_ai_debug_state(world: World) -> dict:
    """
    Build a JSON-serializable snapshot of AI state for debugging UI.
    Includes economy.
    """
    owned_count = {fid: 0 for fid in FACTION_NAMES.keys()}
    stability_sum = {fid: 0.0 for fid in FACTION_NAMES.keys()}

    for sys in world.systems.values():
        owner = sys.owner
        if owner in owned_count:
            owned_count[owner] += 1
            stability_sum[owner] += sys.stability

    econ_power = _compute_econ_power(world)
    total_econ = sum(econ_power.values()) or 1.0

    factions_debug = []
    for fid, name in FACTION_NAMES.items():
        mem = FACTION_MEMORY.get(fid, FactionMemory(id=fid))
        pers = FACTION_PERSONALITIES.get(
            fid, FactionPersonality(fid, name, 0.5, 0.5, 0.5, 0.5)
        )

        n_owned = owned_count.get(fid, 0)
        avg_stability = stability_sum.get(fid, 0.0) / n_owned if n_owned > 0 else 0.0

        econ = econ_power.get(fid, 0.0)
        econ_share = econ / total_econ if total_econ > 0 else 0.0

        base_attempts = ATTEMPTS_PER_TICK
        eff_attempts = ATTEMPTS_PER_TICK  # equal for everyone now

        factions_debug.append(
            {
                "id": fid,
                "name": name,
                "systems_owned": n_owned,
                "avg_stability": avg_stability,
                "battles_won": mem.battles_won,
                "battles_lost": mem.battles_lost,
                "expansions": mem.expansions,
                "confidence": mem.confidence,
                "war_exhaustion": mem.war_exhaustion,
                "econ_power": econ,
                "econ_share": econ_share,
                "base_attempts": base_attempts,
                "effective_attempts": eff_attempts,
                "color": FACTION_COLORS.get(fid, "#ffffff"),
                "personality": {
                    "aggression": pers.aggression,
                    "expansionism": pers.expansionism,
                    "greed": pers.greed,
                    "caution": pers.caution,
                },
            }
        )

    return {
        "tick": world.tick,
        "factions": factions_debug,
    }
