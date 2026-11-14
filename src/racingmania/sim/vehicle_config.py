"""Load ``web/assets/vehicle.json``: the one vehicle definition shared by the game and the port.

Global context: the browser game passes this file to cannon-es (``web/js/core/physics.js``); the
port reads the same numbers here, so tuning the car in one place changes it everywhere.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from ..utils.paths import REPO_ROOT

DEFAULT_VEHICLE_FILE = REPO_ROOT / "web" / "assets" / "vehicle.json"


@dataclass(frozen=True)
class VehicleConfig:
    """Plain numbers from vehicle.json plus derived constants computed exactly as cannon-es does."""

    raw: dict  # the parsed JSON, kept for the controls block and for writing reports
    dt: float
    ticks_per_action: int
    gravity: tuple[float, float, float]
    mass: float
    inv_mass: float  # Body.updateMassProperties: 1.0 / mass
    inv_inertia: tuple[float, float, float]  # inverse of Box.calculateInertia(half_extents, mass)
    linear_damping_factor: float  # Math.pow(1 - linearDamping, dt), applied every tick
    angular_damping_factor: float
    wheel: dict
    connections: tuple[tuple[float, float, float], ...]
    front_wheels: tuple[int, ...]
    drive_wheels: tuple[int, ...]
    controls: dict
    spawn_height: float

    def connections_tensor(self, device, dtype) -> torch.Tensor:
        """Wheel connection points in the chassis frame, ``[4, 3]``."""
        return torch.tensor(self.connections, device=device, dtype=dtype)


def box_inertia(half_extents: tuple[float, float, float], mass: float) -> tuple[float, float, float]:
    """Box.calculateInertia, with cannon-es's operation order: ``1/12 * m * (2ey*2ey + 2ez*2ez)`` and cyclic."""
    ex, ey, ez = half_extents
    ix = 1.0 / 12.0 * mass * (2 * ey * 2 * ey + 2 * ez * 2 * ez)
    iy = 1.0 / 12.0 * mass * (2 * ex * 2 * ex + 2 * ez * 2 * ez)
    iz = 1.0 / 12.0 * mass * (2 * ey * 2 * ey + 2 * ex * 2 * ex)
    return ix, iy, iz


def load_vehicle_config(path: str | Path = DEFAULT_VEHICLE_FILE) -> VehicleConfig:
    """Parse vehicle.json and derive the constants the port needs."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ch = raw["chassis"]
    mass = float(ch["mass"])
    inertia = box_inertia(tuple(ch["half_extents"]), mass)
    inv_inertia = tuple(1.0 / i if i > 0 else 0.0 for i in inertia)  # Body.updateMassProperties
    dt = float(raw["dt"])
    return VehicleConfig(
        raw=raw,
        dt=dt,
        ticks_per_action=int(raw["ticks_per_action"]),
        gravity=tuple(float(g) for g in raw["gravity"]),
        mass=mass,
        inv_mass=1.0 / mass,
        inv_inertia=inv_inertia,
        linear_damping_factor=math.pow(1.0 - float(ch["linear_damping"]), dt),  # World.internalStep damping
        angular_damping_factor=math.pow(1.0 - float(ch["angular_damping"]), dt),
        wheel=dict(raw["wheel"]),
        connections=tuple(tuple(float(x) for x in c) for c in raw["wheel_connections"]),
        front_wheels=tuple(raw["front_wheels"]),
        drive_wheels=tuple(raw["drive_wheels"]),
        controls=dict(raw["controls"]),
        spawn_height=float(raw["spawn_height"]),
    )
