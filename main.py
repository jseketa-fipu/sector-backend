#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
import traceback
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from config import SIM_CONFIG
from world import World, create_sector, advance_world, FACTION_NAMES
from bots import (
    generate_orders,
    update_bot_memory_and_personality,
    get_ai_debug_state,
    FACTION_COLORS,
    reset_factions,
    set_custom_order_fn,
)

try:
    import torch
    from train_selfplay import NeuralPolicy, SelfPlayTrainer
except ImportError:
    torch = None  # type: ignore[assignment]
    NeuralPolicy = None  # type: ignore[assignment]
    SelfPlayTrainer = None  # type: ignore[assignment]

TICK_DELAY: float = float(SIM_CONFIG.get("tick_delay", 0.5))

BASE_DIR = Path(__file__).resolve().parent

_simulation_task: asyncio.Task | None = None
_online_training_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _start_online_training()
    await start_simulation()
    try:
        yield
    finally:
        tasks = [t for t in (_simulation_task, _online_training_task) if t]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)

print(">>> Starting sector_sim with TICK_DELAY =", TICK_DELAY)

world: World = create_sector()
world_lock = asyncio.Lock()
RUN_COUNTER = 1
RUN_LOG_DIR = BASE_DIR / "logs" / "runs"
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

ONLINE_TRAIN_POLICY: bool = bool(SIM_CONFIG.get("online_train_policy", False))
ONLINE_MAX_TICKS: int = int(SIM_CONFIG.get("online_train_max_ticks", 300))
ONLINE_LR: float = float(SIM_CONFIG.get("online_train_lr", 5e-4))
ONLINE_SAVE_PATH = Path(SIM_CONFIG.get("online_policy_checkpoint", "models/online_policy.pt"))


class OnlinePolicyManager:
    """
    Runs a shared neural policy and trains it in the background via self-play.
    Inference and training share the same weights behind a lock.
    """

    def __init__(self, device: torch.device):
        if NeuralPolicy is None or torch is None:
            raise RuntimeError("PyTorch not available for online training.")
        self.device = device
        self.policy = NeuralPolicy(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=ONLINE_LR)
        self.trainer = SelfPlayTrainer(
            device=device,
            lr=ONLINE_LR,
            max_ticks=ONLINE_MAX_TICKS,
            policy=self.policy,
            optimizer=self.optimizer,
        )
        self.lock = threading.Lock()
        self.episode = 0

        # load previous checkpoint if available
        if ONLINE_SAVE_PATH.exists():
            state = torch.load(ONLINE_SAVE_PATH, map_location=self.device)
            self.policy.net.load_state_dict(state)
            print(f"[online-train] Loaded policy checkpoint from {ONLINE_SAVE_PATH}")

    def act_all(self, world: World) -> list:
        with self.lock:
            orders = []
            for fid in FACTION_NAMES.keys():
                chosen, _, _ = self.policy.act(world, fid)
                orders.extend(chosen)
            return orders

    def train_one_episode(self) -> None:
        with self.lock:
            self.episode += 1
            result = self.trainer.run_episode(self.episode)
            if ONLINE_SAVE_PATH:
                ONLINE_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.policy.net.state_dict(), ONLINE_SAVE_PATH)
            print(
                f"[online-train] ep={self.episode} winner={result.winner or 'none'} "
                f"ticks={result.ticks} loss={result.loss_value:.3f}"
            )


online_policy_manager: OnlinePolicyManager | None = None


def _compute_siege_state(world: World) -> dict[int, dict]:
    """
    Build a per-system map describing whether it's under siege and by whom.
    A system is considered besieged if:
    - it has an owner and any idle fleet from another faction is present, or
    - it is currently being occupied (occupation_faction set), or
    - it is unowned and multiple factions have fleets present (contested neutral).
    """
    fleets_at: dict[int, list] = {}
    for fl in world.fleets.values():
        if fl.eta != 0 or fl.system_id is None or fl.strength <= 0:
            continue
        fleets_at.setdefault(fl.system_id, []).append(fl)

    siege_state: dict[int, dict] = {}
    for sys_id, sys in world.systems.items():
        present = fleets_at.get(sys_id, [])
        enemy_by_owner: dict[str, float] = {}
        occupants: set[str] = set()
        for fl in present:
            occupants.add(fl.owner)
            if sys.owner is None or fl.owner != sys.owner:
                enemy_by_owner[fl.owner] = enemy_by_owner.get(fl.owner, 0.0) + fl.strength

        enemy_strength = sum(enemy_by_owner.values())
        siege_owner = None
        if enemy_by_owner:
            siege_owner = max(enemy_by_owner.items(), key=lambda kv: kv[1])[0]

        contested_neutral = sys.owner is None and len(occupants) > 1
        is_besieged = bool(sys.occupation_faction) or (
            sys.owner is not None and enemy_strength > 0
        ) or contested_neutral

        siege_state[sys_id] = {
            "is_besieged": is_besieged,
            "siege_owner": siege_owner,
            "siege_strength": enemy_strength,
        }
    return siege_state


async def _online_training_loop() -> None:
    """
    Background coroutine that runs self-play episodes continuously to improve the policy.
    """
    global online_policy_manager
    if online_policy_manager is None:
        return

    loop = asyncio.get_running_loop()
    while True:
        # Run training in a thread to avoid blocking the event loop
        await loop.run_in_executor(None, online_policy_manager.train_one_episode)


def dump_run_history(history: list[dict], factions: dict, winner: str | None, end_tick: int, run_id: int) -> None:
    payload = {
        "winner": winner,
        "end_tick": end_tick,
        "factions": factions,
        "history": history,
    }
    fname = f"run_{run_id:04d}_tick{end_tick}_winner_{winner or 'none'}.json"
    out_path = RUN_LOG_DIR / fname
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _start_online_training() -> None:
    global online_policy_manager, _online_training_task
    if not ONLINE_TRAIN_POLICY:
        return
    if torch is None:
        print("[online-train] PyTorch unavailable; online training disabled.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_policy_manager = OnlinePolicyManager(device=device)

    def _order_fn(world: World):
        if online_policy_manager is None:
            return []
        return online_policy_manager.act_all(world)

    set_custom_order_fn(_order_fn)
    print("[online-train] Enabled online neural policy training.")
    _online_training_task = asyncio.create_task(_online_training_loop())


@app.get("/")
async def index():
    """Lightweight health endpoint for the backend."""
    return JSONResponse({"status": "ok", "service": "sector-backend"})


@app.get("/history")
async def history_endpoint():
    """
    Dump the full structured history as JSON.
    """
    async with world_lock:
        data = [
            {
                "tick": ev.tick,
                "kind": ev.kind,
                "systems": ev.systems,
                "factions": ev.factions,
                "text": ev.text,
            }
            for ev in world.history
        ]
    return JSONResponse(data)


@app.get("/system/{system_id}")
async def system_detail(system_id: int):
    """Return current system stats plus recent history related to it."""
    async with world_lock:
        if system_id not in world.systems:
            return JSONResponse({"error": "not found"}, status_code=404)
        sys = world.systems[system_id]
        siege_state = _compute_siege_state(world).get(system_id, {})

        fleets_here = [
            {
                "id": fl.id,
                "owner": fl.owner,
                "strength": fl.strength,
                "eta": fl.eta,
                "enroute_from": fl.enroute_from,
                "enroute_to": fl.enroute_to,
            }
            for fl in world.fleets.values()
            if fl.system_id == system_id and fl.strength > 0
        ]

        # filter history entries involving this system
        hist = [
            {
                "tick": ev.tick,
                "kind": ev.kind,
                "factions": ev.factions,
                "text": ev.text,
            }
            for ev in world.history
            if system_id in ev.systems
        ]

        data = {
            "id": sys.id,
            "owner": sys.owner,
            "owner_name": FACTION_NAMES.get(sys.owner, sys.owner) if sys.owner else None,
            "value": sys.value,
            "stability": sys.stability,
            "unrest": sys.unrest,
            "heat": sys.heat,
            "kind": sys.kind,
            "reclaim_cooldown": sys.reclaim_cooldown,
            "occupation_faction": sys.occupation_faction,
            "occupation_progress": sys.occupation_progress,
            "is_besieged": siege_state.get("is_besieged", False),
            "siege_owner": siege_state.get("siege_owner"),
            "siege_strength": siege_state.get("siege_strength", 0.0),
            "fleets": fleets_here,
            "history": hist,  # full history for this system (already capped globally)
        }
    return JSONResponse(data)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    print("WS: incoming connection")
    await ws.accept()
    print("WS: client accepted")
    try:
        while True:
            async with world_lock:
                siege_state = _compute_siege_state(world)
                payload = {
                    "tick_delay": TICK_DELAY,
                    "tick_delay_ms": int(TICK_DELAY * 1000),
                    "tick": world.tick,
                    "systems": [
                        {
                            "id": sys.id,
                            "x": sys.x,
                            "y": sys.y,
                            "owner": sys.owner,
                            "value": sys.value,
                            "kind": sys.kind,
                            "heat": sys.heat,
                            "occupation_faction": sys.occupation_faction,
                            "occupation_progress": sys.occupation_progress,
                            "is_besieged": siege_state.get(sys.id, {}).get("is_besieged", False),
                            "siege_owner": siege_state.get(sys.id, {}).get("siege_owner"),
                            "siege_strength": siege_state.get(sys.id, {}).get("siege_strength", 0.0),
                        }
                        for sys in world.systems.values()
                    ],
                    "lanes": [[a, b] for (a, b) in world.lanes],
                    "fleets": [
                        {
                            "id": fl.id,
                            "owner": fl.owner,
                            "system_id": fl.system_id,
                            "strength": fl.strength,
                            "enroute_from": fl.enroute_from,
                            "enroute_to": fl.enroute_to,
                            "eta": fl.eta,
                        }
                        for fl in world.fleets.values()
                    ],
                    "events": world.events[-30:],
                    "highlight_ids": world.last_event_systems,
                    "ai_state": get_ai_debug_state(world),
                    "factions": [
                        {
                            "id": fid,
                            "name": name,
                            "color": FACTION_COLORS.get(fid, "#ffffff"),
                        }
                        for fid, name in FACTION_NAMES.items()
                    ],
                    "history_tail": [
                        {
                            "tick": ev.tick,
                            "kind": ev.kind,
                            "systems": ev.systems,
                            "factions": ev.factions,
                            "text": ev.text,
                        }
                        for ev in world.history[-40:]
                    ],
                }
            await ws.send_json(payload)
            await asyncio.sleep(TICK_DELAY)
    except WebSocketDisconnect:
        print("WS: client disconnected")
        return
    except Exception:
        print("WS: unexpected error in websocket handler:")
        traceback.print_exc()
        return


async def start_simulation() -> None:
    global _simulation_task
    random.seed()
    print(">>> startup: simulation task starting")

    async def run():
        global RUN_COUNTER
        global world
        while True:
            try:
                restart_winner = None
                restart_msg = None
                end_tick = None

                async with world_lock:
                    # prune fully dead fleets to keep owner checks accurate
                    dead_ids = [fid for fid, fl in world.fleets.items() if fl.strength <= 0]
                    for fid in dead_ids:
                        del world.fleets[fid]

                    orders = generate_orders(world)
                    summary = advance_world(world, orders)
                    update_bot_memory_and_personality(summary)

                    # End condition: only one faction with assets (systems or fleets)
                    system_owners = {sys.owner for sys in world.systems.values() if sys.owner}
                    fleet_owners = {fl.owner for fl in world.fleets.values() if fl.strength > 0}

                    active = system_owners | fleet_owners
                    if restart_winner is None and len(system_owners) == 1:
                        restart_winner = next(iter(system_owners))
                        end_tick = world.tick
                        restart_msg = (
                            f"SIM: ending run at tick {end_tick}, winner={FACTION_NAMES.get(restart_winner, restart_winner)}; restarting after delay "
                            "(owns all systems)."
                        )
                    if len(active) <= 1:
                        restart_winner = next(iter(active), None)
                        end_tick = world.tick
                        winner_label = FACTION_NAMES.get(restart_winner, restart_winner) if restart_winner else "none"
                        restart_msg = f"SIM: ending run at tick {end_tick}, winner={winner_label}; restarting after delay."

                    # Fail-safe: if one faction owns all systems and others only have trivial fleet strength, end the run
                    if restart_winner is None and len(system_owners) == 1:
                        dominant = next(iter(system_owners))
                        other_owners = {
                            fl.owner for fl in world.fleets.values() if fl.owner != dominant and fl.strength > 0
                        }
                        if not other_owners:
                            restart_winner = dominant
                            end_tick = world.tick
                            restart_msg = (
                                f"SIM: ending run at tick {end_tick}, winner={FACTION_NAMES.get(dominant, dominant)} "
                                "(no opposing fleets or systems); restarting after delay."
                            )
                        if restart_winner is None:
                            other_strength = sum(
                                fl.strength for fl in world.fleets.values() if fl.owner != dominant and fl.strength > 0
                            )
                            # If no other systems exist and opposing fleets are negligible, call it
                            if other_strength < 2.0:
                                restart_winner = dominant
                                end_tick = world.tick
                                restart_msg = (
                                    f"SIM: ending run at tick {end_tick}, winner={FACTION_NAMES.get(dominant, dominant)} "
                                    "(others <2 fleet strength and no systems); restarting after delay."
                                )

                if restart_winner is not None:
                    # Snapshot run history and factions before reset
                    async with world_lock:
                        history_snapshot = [
                            {
                                "tick": ev.tick,
                                "kind": ev.kind,
                                "systems": ev.systems,
                                "factions": ev.factions,
                                "text": ev.text,
                            }
                            for ev in world.history
                        ]
                        factions_snapshot = dict(FACTION_NAMES)

                    dump_run_history(history_snapshot, factions_snapshot, restart_winner, end_tick or world.tick, RUN_COUNTER)

                    print(restart_msg)
                    await asyncio.sleep(2.0)
                    async with world_lock:
                        reset_factions()
                        world = create_sector()
                    RUN_COUNTER += 1
                    continue
                if world.tick % 20 == 0:
                    print(f"SIM: tick {world.tick}")
                await asyncio.sleep(TICK_DELAY)
            except Exception:
                print("SIM: error in background loop:")
                traceback.print_exc()
                await asyncio.sleep(1.0)

    _simulation_task = asyncio.create_task(run())


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", SIM_CONFIG.get("port", 8000)))
    reload_flag = os.environ.get("RELOAD", "").lower() in {"1", "true", "yes", "on"}
    uvicorn.run("main:app", host=host, port=port, reload=reload_flag)
