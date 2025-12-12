#!/usr/bin/env python3
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import SIM_CONFIG

# World-level config from JSON
NUM_SYSTEMS: int = int(SIM_CONFIG.get("num_systems", 40))
LANES_PER_SYSTEM: int = int(SIM_CONFIG.get("lanes_per_system", 3))

# Specials config
SPECIAL_CFG = SIM_CONFIG.get("special_systems", {})
NUM_RELIC = int(SPECIAL_CFG.get("relic", 0))
NUM_FORGE = int(SPECIAL_CFG.get("forge", 0))
NUM_HIVE = int(SPECIAL_CFG.get("hive", 0))

# Faction IDs and names from JSON
FACTION_CONFIG: Dict[str, dict] = SIM_CONFIG.get("factions", {})
FACTION_NAMES: Dict[str, str] = {
    fid: fcfg.get("name", fid) for fid, fcfg in FACTION_CONFIG.items()
}

# Rebellion tuning
REBELLION_STABILITY_THRESHOLD = 0.08  # only VERY unstable worlds can revolt
REBELLION_CHANCE = 0.25  # 25% chance when under threshold
NEW_FACTION_CHANCE = 0.2  # 20% of rebellions join/create a rebel faction
UPKEEP_IDLE = 0.01  # per-tick idle fleet upkeep (drain)
OVEREXTENSION_THRESHOLD = int(SIM_CONFIG.get("overextension_threshold", 15))
OVEREXTENSION_RISK_PER_EXTRA = float(SIM_CONFIG.get("overextension_risk_per_extra", 0.02))
ECON_OVEREXT_PENALTY_PER_EXTRA = float(SIM_CONFIG.get("econ_overextension_penalty_per_extra", 0.04))
ECON_MATURITY_TICKS = int(SIM_CONFIG.get("econ_maturity_ticks", 15))
ECON_MATURITY_MIN_FACTOR = float(SIM_CONFIG.get("econ_maturity_min_factor", 0.2))
LEAGUE_FACTION_ID = "L"
LEAGUE_TECH_BONUS = float(SIM_CONFIG.get("league_tech_bonus", 0.3))
LEAGUE_MILITIA_STRENGTH = float(SIM_CONFIG.get("league_militia_strength", 6.0))
CAPTURE_TICKS = int(SIM_CONFIG.get("capture_ticks", 3))  # ticks required to flip undefended system
MIN_CAPTURE_STRENGTH = float(SIM_CONFIG.get("min_capture_strength", 3.0))  # min idle strength to capture
GARRISON_MAX = float(SIM_CONFIG.get("garrison_max_strength", 3.0))  # max passive home guard
GARRISON_REGEN = float(SIM_CONFIG.get("garrison_regen_per_tick", 0.2))  # regen per tick

# Fleet travel config
TRAVEL_TICKS = 2  # movement duration between neighboring systems
BUILD_RATE = 0.005  # econ -> build stock per tick
BUILD_COST = 5.0  # stock needed for a small fleet
NEW_FLEET_STRENGTH = 5.0
REPAIR_RATE = 0.02
DAMAGE_FACTOR = 0.7


@dataclass
class StarSystem:
    id: int
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    owner: Optional[str] = None  # faction id or None
    neighbors: List[int] = field(default_factory=list)
    value: int = 1  # resource / importance, 1–10
    stability: float = 1.0  # 0..1, low = rebellion likely
    kind: str = "normal"  # "normal" | "relic" | "forge" | "hive"
    heat: float = 0.0  # recent conflict, for visuals
    unrest: float = 0.0  # builds up on frequent ownership changes
    reclaim_cooldown: int = 0  # ticks before auto-claim can occur after rebellion
    occupation_faction: Optional[str] = None  # who is occupying
    occupation_progress: int = 0  # ticks of continuous occupation
    garrison: float = 0.0  # passive defense that regrows when owned
    econ_maturity: int = 0  # ticks since owned (caps at ECON_MATURITY_TICKS)


@dataclass
class Fleet:
    id: int
    owner: str
    system_id: Optional[int]  # None when enroute
    strength: float
    max_strength: float
    experience: float = 0.0
    enroute_from: Optional[int] = None
    enroute_to: Optional[int] = None
    eta: int = 0  # ticks remaining to arrival; 0 means idle at system


@dataclass
class HistoricalEvent:
    tick: int
    kind: str  # "expansion", "fleet_battle_win", "rebellion_*", ...
    systems: List[int]
    factions: List[str]
    text: str


@dataclass
class World:
    tick: int
    systems: Dict[int, StarSystem]
    lanes: List[Tuple[int, int]]
    events: List[str]
    last_event_systems: List[int] = field(default_factory=list)
    history: List[HistoricalEvent] = field(default_factory=list)

    # Fleets & production
    fleets: Dict[int, Fleet] = field(default_factory=dict)
    next_fleet_id: int = 0
    faction_build_stock: Dict[str, float] = field(default_factory=dict)


@dataclass
class Order:
    """A generic order issued by a faction for this tick."""

    faction: str  # e.g. "E", "P", "T"
    origin_id: int  # system id issuing the order
    target_id: int  # neighbor system id being targeted
    reason: Optional[str] = None  # why this move was chosen (for history logging)


@dataclass
class TickSummary:
    """Aggregated outcome of a single tick, for AI adaptation."""

    battles_won: Dict[str, int]
    battles_lost: Dict[str, int]
    expansions: Dict[str, int]


# ---------- Sector generation ----------


def create_sector(num_systems: int = NUM_SYSTEMS) -> World:
    systems: Dict[int, StarSystem] = {}

    # Random positions & values
    for i in range(num_systems):
        x = random.random()
        y = random.random()
        value = random.randint(1, 10)
        systems[i] = StarSystem(id=i, x=x, y=y, value=value, econ_maturity=ECON_MATURITY_TICKS)

    # Lanes: connect each system to LANES_PER_SYSTEM nearest neighbors
    lane_set = set()
    ids = list(systems.keys())

    def dist2(a: StarSystem, b: StarSystem) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return dx * dx + dy * dy

    for i in ids:
        sys_i = systems[i]
        distances = []
        for j in ids:
            if i == j:
                continue
            sys_j = systems[j]
            distances.append((dist2(sys_i, sys_j), j))
        distances.sort(key=lambda t: t[0])
        nearest = [j for _, j in distances[:LANES_PER_SYSTEM]]
        for j in nearest:
            a, b = sorted((i, j))
            lane_set.add((a, b))

    # Ensure the graph is fully connected by linking components with shortest bridges
    def components(edges: set[tuple[int, int]]) -> List[List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in ids}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        seen = set()
        comps: List[List[int]] = []
        for node in ids:
            if node in seen:
                continue
            stack = [node]
            curr = []
            seen.add(node)
            while stack:
                n = stack.pop()
                curr.append(n)
                for nxt in adj[n]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            comps.append(curr)
        return comps

    comp_list = components(lane_set)
    while len(comp_list) > 1:
        # connect the smallest component to the largest (or first)
        comp_list.sort(key=len, reverse=True)
        main = comp_list[0]
        other = comp_list[-1]
        best = None
        best_d2 = None
        for a_id in main:
            for b_id in other:
                d2 = dist2(systems[a_id], systems[b_id])
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = (min(a_id, b_id), max(a_id, b_id))
        if best:
            lane_set.add(best)
        comp_list = components(lane_set)

    for a, b in lane_set:
        systems[a].neighbors.append(b)
        systems[b].neighbors.append(a)

    # Assign special system types
    available = ids.copy()
    random.shuffle(available)

    def pop_n(n: int) -> List[int]:
        chosen = available[:n]
        del available[:n]
        return chosen

    for sid in pop_n(min(NUM_RELIC, len(available))):
        systems[sid].kind = "relic"
    for sid in pop_n(min(NUM_FORGE, len(available))):
        systems[sid].kind = "forge"
    for sid in pop_n(min(NUM_HIVE, len(available))):
        systems[sid].kind = "hive"

    # Starting systems for each (initial) faction
    faction_ids = list(FACTION_NAMES.keys())
    if len(faction_ids) > len(ids):
        raise ValueError("More factions defined than systems in sector.")
    starting_ids = random.sample(ids, k=len(faction_ids))
    for faction, sid in zip(faction_ids, starting_ids):
        systems[sid].owner = faction

    world = World(
        tick=0,
        systems=systems,
        lanes=list(lane_set),
        events=[],
        last_event_systems=[],
        history=[],
    )

    # Initialize build stock
    world.faction_build_stock = {fid: 0.0 for fid in faction_ids}
    # Initialize garrisons
    for sys in systems.values():
        sys.garrison = GARRISON_MAX

    # Starting fleets: one per faction at its home system
    def spawn_fleet(owner: str, system_id: int, strength: float):
        fid = world.next_fleet_id
        world.next_fleet_id += 1
        world.fleets[fid] = Fleet(
            id=fid,
            owner=owner,
            system_id=system_id,
            strength=strength,
            max_strength=strength,
            enroute_from=None,
            enroute_to=None,
            eta=0,
        )

    for faction, sid in zip(faction_ids, starting_ids):
        spawn_fleet(faction, sid, strength=10.0)

    return world


# ---------- Economic power helper ----------


def _system_econ_value(sys: StarSystem) -> float:
    """Effective economic contribution of a single system."""
    base = float(sys.value)
    # Special worlds are worth more economically
    if sys.kind == "relic":
        base *= 1.8
    elif sys.kind == "forge":
        base *= 1.4
    elif sys.kind == "hive":
        base *= 1.2
    # Stability matters: unstable worlds don't contribute fully
    stability_factor = 0.5 + 0.5 * sys.stability
    maturity = min(1.0, sys.econ_maturity / float(max(1, ECON_MATURITY_TICKS)))
    maturity_factor = ECON_MATURITY_MIN_FACTOR + (1.0 - ECON_MATURITY_MIN_FACTOR) * maturity
    return base * stability_factor * maturity_factor


def _compute_econ_power(world: World, faction_ids: List[str]) -> Dict[str, float]:
    econ_power = {f: 1.0 for f in faction_ids}  # baseline so nobody is 0
    sizes: Dict[str, int] = {f: 0 for f in faction_ids}
    for sys in world.systems.values():
        owner = sys.owner
        if owner in econ_power:
            sizes[owner] = sizes.get(owner, 0) + 1
            econ_power[owner] += _system_econ_value(sys)

    for fid, size in sizes.items():
        extra = max(0, size - OVEREXTENSION_THRESHOLD)
        if extra > 0:
            penalty = 1.0 / (1.0 + ECON_OVEREXT_PENALTY_PER_EXTRA * extra)
            econ_power[fid] *= penalty
    return econ_power


def _tech_multiplier(faction: Optional[str]) -> float:
    if faction == LEAGUE_FACTION_ID:
        return 1.0 + LEAGUE_TECH_BONUS
    return 1.0


# ---------- Simulation (fleets, orders, rebellions, sprawl) ----------


def advance_world(world: World, orders: List[Order]) -> TickSummary:
    """
    Advance the world by one tick by applying a list of Orders.
    The 'AI' deciding these orders lives in bots.py.
    Returns a TickSummary so bots can adapt.
    """
    world.tick += 1
    events: List[str] = []
    history = world.history
    highlight_ids = set()

    faction_ids = list(FACTION_NAMES.keys())

    # Counters for summary (can grow with new factions via setdefault)
    wins: Dict[str, int] = {f: 0 for f in faction_ids}
    losses: Dict[str, int] = {f: 0 for f in faction_ids}
    expansions: Dict[str, int] = {f: 0 for f in faction_ids}

    owned_by: Dict[str, List[StarSystem]] = {f: [] for f in faction_ids}
    for sys in world.systems.values():
        if sys.owner in owned_by:
            owned_by[sys.owner].append(sys)

    # Economic power snapshot at the start of the tick
    econ_power = _compute_econ_power(world, faction_ids)

    # Ensure build stock for any new factions
    for f in faction_ids:
        world.faction_build_stock.setdefault(f, 0.0)

    def describe_system(sys: StarSystem) -> str:
        if sys.kind == "relic":
            return f"relic world #{sys.id}"
        if sys.kind == "forge":
            return f"forge world #{sys.id}"
        if sys.kind == "hive":
            return f"hive world #{sys.id}"
        return f"system #{sys.id}"

    def log_event(kind: str, systems_ids: List[int], factions_ids: List[str], text: str):
        events.append(text)
        history.append(
            HistoricalEvent(
                tick=world.tick,
                kind=kind,
                systems=systems_ids,
                factions=factions_ids,
                text=text,
            )
        )

    def check_rebellion(sys: StarSystem):
        """
        If stability is very low, the system may rebel:
        - either becomes independent (owner None)
        - or (rarely) spawns / joins a rebel faction (splinter state).
        """
        from bots import register_new_faction, ensure_league_faction  # late import to avoid cyclic import

        if sys.owner is None:
            return

        old_owner = sys.owner

        # Only consider VERY unstable systems (affected by unrest)
        effective_stability = sys.stability - 0.2 * sys.unrest
        if effective_stability > REBELLION_STABILITY_THRESHOLD:
            return

        # Only a fraction of such systems actually rebel this tick (higher with unrest/overextension)
        unrest_factor = 1 + sys.unrest
        base_chance = REBELLION_CHANCE * unrest_factor
        owner_size = sum(1 for s in world.systems.values() if s.owner == old_owner)
        if owner_size > OVEREXTENSION_THRESHOLD:
            extra = owner_size - OVEREXTENSION_THRESHOLD
            base_chance *= 1.0 + OVEREXTENSION_RISK_PER_EXTRA * extra
            base_chance = min(base_chance, 0.9)
        if random.random() > base_chance:
            return

        old_owner = sys.owner
        base_label = FACTION_NAMES.get(old_owner, "Rebels")

        roll = random.random()
        if roll > NEW_FACTION_CHANCE:
            # Most rebellions just go independent
            ensure_league_faction(world)
            sys.owner = LEAGUE_FACTION_ID
            sys.stability = 0.7
            sys.reclaim_cooldown = 6  # give the league a moment to secure
            sys.occupation_faction = None
            sys.occupation_progress = 0
            sys.garrison = min(GARRISON_MAX, GARRISON_MAX + 1.0)
            sys.econ_maturity = 0
            # spawn a militia fleet with tech edge
            fid = world.next_fleet_id
            world.next_fleet_id += 1
            world.fleets[fid] = Fleet(
                id=fid,
                owner=LEAGUE_FACTION_ID,
                system_id=sys.id,
                strength=LEAGUE_MILITIA_STRENGTH,
                max_strength=LEAGUE_MILITIA_STRENGTH,
                enroute_from=None,
                enroute_to=None,
                eta=0,
            )
            world.faction_build_stock[LEAGUE_FACTION_ID] = world.faction_build_stock.get(
                LEAGUE_FACTION_ID, 0.0
            )
            text = (
                f"t={world.tick}: Rebellion in {describe_system(sys)}! "
                f"{FACTION_NAMES[old_owner]} lost control; the world joined the {FACTION_NAMES[LEAGUE_FACTION_ID]}."
            )
            log_event("rebellion_independence", [sys.id], [old_owner, LEAGUE_FACTION_ID], text)
        else:
            # A minority actually form / join a rebel faction
            new_fid = register_new_faction(old_owner, base_label)
            sys.owner = new_fid
            sys.stability = 0.8
            sys.reclaim_cooldown = 3
            sys.occupation_faction = None
            sys.occupation_progress = 0
            sys.garrison = GARRISON_MAX
            sys.econ_maturity = 0
            # give the new rebels an initial defensive fleet
            fid = world.next_fleet_id
            world.next_fleet_id += 1
            world.fleets[fid] = Fleet(
                id=fid,
                owner=new_fid,
                system_id=sys.id,
                strength=8.0,
                max_strength=8.0,
                enroute_from=None,
                enroute_to=None,
                eta=0,
            )
            # seed some build stock so they can reinforce soon
            world.faction_build_stock[new_fid] = world.faction_build_stock.get(new_fid, 0.0) + BUILD_COST
            text = (
                f"t={world.tick}: Secession in {describe_system(sys)}! "
                f"A splinter of {FACTION_NAMES[old_owner]} joined the rebel faction "
                f"{FACTION_NAMES[new_fid]}."
            )
            expansions.setdefault(new_fid, 0)
            expansions[new_fid] += 1
            log_event("rebellion_secession", [sys.id], [old_owner, new_fid], text)

        highlight_ids.add(sys.id)
        losses.setdefault(old_owner, 0)
        losses[old_owner] += 1

    # Handle in-flight fleets (arrivals happen before new orders)
    arrivals: List[int] = []
    for fid, fl in world.fleets.items():
        if fl.eta > 0:
            fl.eta -= 1
            if fl.eta <= 0:
                arrivals.append(fid)

    def resolve_arrival(fleet: Fleet):
        # Put fleet at destination
        dest_id = fleet.enroute_to
        if dest_id is None:
            return
        fleet.system_id = dest_id
        fleet.enroute_from = None
        fleet.enroute_to = None
        fleet.eta = 0

        target_sys = world.systems[dest_id]
        old_owner = target_sys.owner

        defenders = [
            fl
            for fl in world.fleets.values()
            if fl.system_id == dest_id and fl.owner != fleet.owner and fl.strength > 0 and fl.eta == 0
        ]
        defender_factions = {fl.owner for fl in defenders}

        def _complete_capture(instantly: bool = False):
            target_sys.owner = fleet.owner
            target_sys.occupation_faction = None
            target_sys.occupation_progress = 0
            target_sys.unrest = min(1.0, target_sys.unrest + 0.25)
            target_sys.stability = max(0.0, target_sys.stability - 0.05 - 0.1 * target_sys.unrest)
            target_sys.heat = min(3.0, target_sys.heat + 1.5)
            target_sys.garrison = min(GARRISON_MAX, target_sys.garrison + 0.5 * GARRISON_MAX)
            target_sys.econ_maturity = 0

            expansions.setdefault(fleet.owner, 0)
            expansions[fleet.owner] += 1
            if old_owner:
                losses.setdefault(old_owner, 0)
                losses[old_owner] += 1
                wins.setdefault(fleet.owner, 0)
                wins[fleet.owner] += 1

            text = (
                f"t={world.tick}: Fleet of {FACTION_NAMES[fleet.owner]} secured "
                f"{describe_system(target_sys)} {'via rapid occupation' if instantly else 'after occupation.'}"
            )
            log_event("fleet_capture_no_defense", [target_sys.id], [fleet.owner], text)
            highlight_ids.add(target_sys.id)
            check_rebellion(target_sys)

        # No defenders
        if not defenders:
            # reset occupation if returning owner
            if old_owner == fleet.owner:
                target_sys.occupation_faction = None
                target_sys.occupation_progress = 0
                return

            # tally idle attacking strength at this system
            attackers_here = [
                fl
                for fl in world.fleets.values()
                if fl.system_id == dest_id and fl.owner == fleet.owner and fl.strength > 0 and fl.eta == 0
            ]
            total_attack = sum(fl.strength for fl in attackers_here)

            # Garrison soaks damage first
            if target_sys.garrison > 0:
                absorbed = min(target_sys.garrison, total_attack)
                target_sys.garrison -= absorbed
                total_attack -= absorbed
                if total_attack <= 0:
                    target_sys.occupation_faction = None
                    target_sys.occupation_progress = 0
                    return

            # If there's no garrison left, flip immediately with any remaining attackers
            if target_sys.garrison <= 0 and total_attack >= MIN_CAPTURE_STRENGTH:
                _complete_capture(instantly=True)
                return

            if total_attack >= MIN_CAPTURE_STRENGTH:
                if target_sys.occupation_faction == fleet.owner:
                    target_sys.occupation_progress += 1
                else:
                    target_sys.occupation_faction = fleet.owner
                    target_sys.occupation_progress = 1
            else:
                target_sys.occupation_faction = None
                target_sys.occupation_progress = 0
                return

            if target_sys.occupation_progress >= CAPTURE_TICKS:
                target_sys.owner = fleet.owner
                target_sys.occupation_faction = None
                target_sys.occupation_progress = 0
                target_sys.unrest = min(1.0, target_sys.unrest + 0.25)
                target_sys.stability = max(
                    0.0, target_sys.stability - 0.05 - 0.1 * target_sys.unrest
                )
                target_sys.heat = min(3.0, target_sys.heat + 1.5)
                target_sys.garrison = min(GARRISON_MAX, target_sys.garrison + 0.5 * GARRISON_MAX)
                target_sys.econ_maturity = 0

                _complete_capture()
            return

        # Attack/defend with local fleets
        attackers_here = [
            fl
            for fl in world.fleets.values()
            if fl.system_id == dest_id and fl.owner == fleet.owner and fl.strength > 0 and fl.eta == 0
        ]
        total_attack = sum(fl.strength for fl in attackers_here) * _tech_multiplier(fleet.owner)
        def_owner = defenders[0].owner if defenders else None
        total_defense = sum(fl.strength for fl in defenders) * _tech_multiplier(def_owner)
        if total_attack <= 0 or total_defense <= 0:
            return

        target_sys.occupation_faction = None
        target_sys.occupation_progress = 0

        target_sys.stability = max(0.0, target_sys.stability - 0.15)
        target_sys.heat = min(3.0, target_sys.heat + 2.0)

        base = 0.5
        effective_defense = total_defense + target_sys.garrison * _tech_multiplier(def_owner)
        local_adv = (total_attack - effective_defense) / (total_attack + effective_defense)
        p_attack_wins = base + 0.3 * local_adv
        if target_sys.reclaim_cooldown > 0:
            p_attack_wins -= 0.15  # defending rebels get a brief edge
        p_attack_wins = max(0.2, min(0.8, p_attack_wins))

        attacker_wins = random.random() < p_attack_wins

        if attacker_wins:
            total_losers = total_defense
            remaining = max(1.0, total_attack - DAMAGE_FACTOR * total_losers)
            target_sys.garrison = 0.0  # spent in defense

            for fl in defenders:
                fl.strength = 0.0

            if attackers_here:
                main = max(attackers_here, key=lambda fl: fl.strength)
                main.strength = remaining
                main.max_strength = max(main.max_strength, remaining)
                for fl in attackers_here:
                    if fl is not main:
                        fl.strength = 0.0

            old_owner = target_sys.owner
            if old_owner != fleet.owner:
                target_sys.owner = fleet.owner
                target_sys.unrest = min(1.0, target_sys.unrest + 0.25)
                target_sys.stability = max(
                    0.0, target_sys.stability - 0.05 - 0.1 * target_sys.unrest
                )
                target_sys.econ_maturity = 0
                expansions.setdefault(fleet.owner, 0)
                expansions[fleet.owner] += 1
                if old_owner:
                    losses.setdefault(old_owner, 0)
                    losses[old_owner] += 1

            wins.setdefault(fleet.owner, 0)
            wins[fleet.owner] += 1

            text = (
                f"t={world.tick}: Fleet of {FACTION_NAMES[fleet.owner]} won a battle in "
                f"{describe_system(target_sys)} (local odds {p_attack_wins:.2f}) and took control."
            )
            log_event("fleet_battle_win", [target_sys.id], [fleet.owner] + list(defender_factions), text)
            highlight_ids.add(target_sys.id)
        else:
            total_losers = total_attack
            remaining = max(1.0, total_defense - DAMAGE_FACTOR * total_losers)
            target_sys.garrison = max(0.0, target_sys.garrison - total_attack)

            for fl in attackers_here:
                fl.strength = 0.0

            defenders_alive = [fl for fl in defenders if fl.strength > 0]
            if defenders_alive:
                main = max(defenders_alive, key=lambda fl: fl.strength)
                main.strength = remaining
                main.max_strength = max(main.max_strength, remaining)
                for fl in defenders_alive:
                    if fl is not main:
                        fl.strength = 0.0

            for df in defender_factions:
                wins.setdefault(df, 0)
                wins[df] += 1

            losses.setdefault(fleet.owner, 0)
            losses[fleet.owner] += 1

            text = (
                f"t={world.tick}: Fleet of {FACTION_NAMES[fleet.owner]} failed to take "
                f"{describe_system(target_sys)} (local odds {p_attack_wins:.2f})."
            )
            log_event("fleet_battle_lose", [target_sys.id], [fleet.owner] + list(defender_factions), text)
            highlight_ids.add(target_sys.id)

        check_rebellion(target_sys)

    for fid in arrivals:
        fl = world.fleets.get(fid)
        if fl:
            resolve_arrival(fl)

    # Safety: if a system has fleets present and no owner, auto-resolve control
    # (unless in reclaim cooldown after rebellion).
    for sys in world.systems.values():
        # decay occupation if no occupying fleets remain
        if sys.occupation_faction:
            occupying_present = any(
                fl.owner == sys.occupation_faction and fl.system_id == sys.id and fl.strength > 0 and fl.eta == 0
                for fl in world.fleets.values()
            )
            if not occupying_present:
                sys.occupation_faction = None
                sys.occupation_progress = 0

        if sys.owner is not None or sys.reclaim_cooldown > 0:
            continue
        fleets_here = [
            fl for fl in world.fleets.values()
            if fl.system_id == sys.id and fl.strength > 0 and fl.eta == 0
        ]
        if not fleets_here:
            continue

        # Pick faction with highest total strength; if tie, leave contested.
        strength_by_owner: Dict[str, float] = {}
        for fl in fleets_here:
            strength_by_owner[fl.owner] = strength_by_owner.get(fl.owner, 0.0) + fl.strength

        if not strength_by_owner:
            continue

        sorted_strength = sorted(strength_by_owner.items(), key=lambda kv: kv[1], reverse=True)
        if len(sorted_strength) > 1 and sorted_strength[0][1] == sorted_strength[1][1]:
            # contested tie; do nothing this tick
            continue

        owner = sorted_strength[0][0]
        sys.owner = owner
        sys.econ_maturity = 0
        sys.unrest = min(1.0, sys.unrest + 0.15)
        sys.stability = max(0.0, sys.stability - 0.05 - 0.05 * sys.unrest)
        sys.reclaim_cooldown = 0
        sys.garrison = min(GARRISON_MAX, sys.garrison + 0.5 * GARRISON_MAX)
        expansions.setdefault(owner, 0)
        expansions[owner] += 1
        text = (
            f"t={world.tick}: {FACTION_NAMES[owner]} claimed empty {describe_system(sys)} "
            f"after securing orbit."
        )
        log_event("auto_claim", [sys.id], [owner], text)
        highlight_ids.add(sys.id)

    # --- Fleet building based on economy ---
    for f in faction_ids:
        stock = world.faction_build_stock.get(f, 0.0)
        stock += econ_power.get(f, 0.0) * BUILD_RATE
        world.faction_build_stock[f] = stock

        while world.faction_build_stock[f] >= BUILD_COST:
            world.faction_build_stock[f] -= BUILD_COST
            owned = [s.id for s in world.systems.values() if s.owner == f]
            if not owned:
                break
            sid = random.choice(owned)
            fid = world.next_fleet_id
            world.next_fleet_id += 1
            world.fleets[fid] = Fleet(
                id=fid,
                owner=f,
                system_id=sid,
                strength=NEW_FLEET_STRENGTH,
                max_strength=NEW_FLEET_STRENGTH,
                enroute_from=None,
                enroute_to=None,
                eta=0,
            )

    # Helper: choose strongest fleet at origin that is idle
    def find_attacking_fleet(faction: str, system_id: int) -> Optional[Fleet]:
        candidates = [
            fl
            for fl in world.fleets.values()
            if fl.owner == faction and fl.system_id == system_id and fl.strength > 0 and fl.eta == 0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda fl: fl.strength)

    # Apply orders using fleets (set them in motion)
    for order in orders:
        attacker = order.faction
        if attacker not in FACTION_NAMES:
            continue
        if order.origin_id not in world.systems or order.target_id not in world.systems:
            continue

        origin_sys = world.systems[order.origin_id]
        target_sys = world.systems[order.target_id]

        # must own the origin system
        if origin_sys.owner != attacker:
            continue
        # must be neighbors
        if target_sys.id not in origin_sys.neighbors:
            continue

        fleet = find_attacking_fleet(attacker, origin_sys.id)
        if fleet is None:
            continue

        # Log the decision to move before launching
        move_reason = getattr(order, "reason", None)
        text = (
            f"t={world.tick}: {FACTION_NAMES[attacker]} launched a fleet "
            f"from system #{origin_sys.id} to system #{target_sys.id} "
            f"(strength {fleet.strength:.1f})."
        )
        if move_reason:
            text += f" Reason: {move_reason}."
        log_event("fleet_move", [origin_sys.id, target_sys.id], [attacker], text)

        # launch fleet
        fleet.enroute_from = origin_sys.id
        fleet.enroute_to = target_sys.id
        fleet.system_id = None
        fleet.eta = TRAVEL_TICKS

    # Remove dead fleets and apply small auto-repair
    to_delete = []
    for fid, fl in world.fleets.items():
        if fl.strength <= 0.0:
            to_delete.append(fid)
            continue

        # upkeep drain to discourage hoarding idle fleets
        if fl.eta == 0:
            fl.strength *= (1.0 - UPKEEP_IDLE)

        if fl.strength <= 0.0:
            to_delete.append(fid)
            continue

        # only repair when idle
        if fl.eta == 0:
            fl.strength = min(fl.max_strength, fl.strength * (1.0 + REPAIR_RATE))

    for fid in to_delete:
        del world.fleets[fid]

    # Merge idle fleets of the same owner at the same system to avoid tiny strays
    merged: Dict[tuple[str, int], Fleet] = {}
    to_remove: List[int] = []
    for fid, fl in world.fleets.items():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        key = (fl.owner, fl.system_id)
        if key not in merged:
            merged[key] = fl
            continue
        primary = merged[key]
        primary.strength += fl.strength
        primary.max_strength = max(primary.max_strength, fl.max_strength, primary.strength)
        to_remove.append(fid)
    for fid in to_remove:
        del world.fleets[fid]

    # Empire size snapshot after this tick (for sprawl / instability)
    systems_per_faction: Dict[str, int] = {}
    for sys in world.systems.values():
        if sys.owner is not None:
            systems_per_faction[sys.owner] = systems_per_faction.get(sys.owner, 0) + 1

    # Global updates: heat decay & stability recovery + empire sprawl penalty
    for sys in world.systems.values():
        sys.heat *= 0.85
        if sys.heat < 0.05:
            sys.heat = 0.0

        # unrest naturally cools off
        sys.unrest *= 0.9

        # reclaim cooldown ticks down
        if sys.reclaim_cooldown > 0:
            sys.reclaim_cooldown -= 1
        # garrison decay/regeneration
        if sys.owner is None:
            sys.garrison = 0.0
            sys.econ_maturity = 0
        else:
            sys.garrison = min(GARRISON_MAX, sys.garrison + GARRISON_REGEN)
            sys.econ_maturity = min(ECON_MATURITY_TICKS, sys.econ_maturity + 1)

        # Base natural recovery
        sys.stability = min(1.0, sys.stability + 0.01)

        # Empire sprawl: big empires are harder to control
        if sys.owner in systems_per_faction:
            size = systems_per_faction[sys.owner]
            # Only kick in if you have more than 10 systems
            if size > 10:
                extra_decay = 0.002 * (size - 10)
                sys.stability = max(0.0, sys.stability - extra_decay)

    world.events.extend(events)
    max_events = 80
    if len(world.events) > max_events:
        world.events = world.events[-max_events:]

    # Cap history so it doesn't grow unbounded
    # Keep full run history (no cap) so completed runs can be dumped intact

    world.last_event_systems = sorted(highlight_ids)

    return TickSummary(
        battles_won=wins,
        battles_lost=losses,
        expansions=expansions,
    )
