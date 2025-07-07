"""Simple spring-mass simulator using NumPy arrays and adjacency matrices.

Each world consists of N point-masses with
‐ positions        : (N,3) array  – metres
‐ velocities       : (N,3) array  – metres / second
‐ masses           : (N,)  array  – kilograms
‐ stiffness(K)     : (N,N) array  – N/m for each (i,j) spring (0 ⇒ no spring)
‐ damping(C)       : (N,N) array  – Ns/m for each (i,j) damper   (0 ⇒ none)
‐ rest_lengths(L0) : (N,N) array  – metres (ignored if no spring)

A single semi-implicit Euler step is performed per call to `main`.
"""

from __future__ import annotations

import numpy as np
from typing import TypedDict, List


class World(TypedDict):
    positions: List[List[float]]
    velocities: List[List[float]] 
    masses: List[float]
    stiffness: List[List[float]]
    damping: List[List[float]]
    rest_length: List[List[float]]


GRAVITY = np.array([0.0, -9.81, 0.0])  # m/s²


def simulate_step(world: World, dt: float = 0.01) -> World:
    """Advance the system by one time-step (semi-implicit Euler)."""

    # Convert to NumPy arrays
    pos = np.asarray(world['positions'], dtype=float)       # (N,3)
    vel = np.asarray(world['velocities'], dtype=float)      # (N,3)
    m = np.asarray(world['masses'], dtype=float)            # (N,)
    k = np.asarray(world['stiffness'], dtype=float)         # (N,N)
    c = np.asarray(world['damping'], dtype=float)           # (N,N)
    L0 = np.asarray(world['rest_length'], dtype=float)     # (N,N)

    N = pos.shape[0]

    # Pairwise displacement vectors ∆x_ij = x_i − x_j
    diff = pos[:, None, :] - pos[None, :, :]             # (N,N,3)
    dist = np.linalg.norm(diff, axis=2) + 1e-12          # (N,N) avoid /0
    direction = diff / dist[:, :, None]                  # (N,N,3)

    # Relative velocity along spring direction
    rel_vel = vel[:, None, :] - vel[None, :, :]          # (N,N,3)
    vel_along = np.sum(rel_vel * direction, axis=2)      # (N,N)

    # Hooke + damping (only where k != 0)
    spring_force = -k * (dist - L0)                      # (N,N)
    damping_force = -c * vel_along                       # (N,N)
    force_mag = spring_force + damping_force             # (N,N)
    np.fill_diagonal(force_mag, 0.0)                     # no self-force

    # Vector forces
    force_vec = direction * force_mag[:, :, None]        # (N,N,3)
    net_force = force_vec.sum(axis=1)                    # (N,3)

    # Gravity
    net_force += m[:, None] * GRAVITY

    # Acceleration and integration
    acc = net_force / m[:, None]                         # (N,3)
    vel_new = vel + acc * dt
    pos_new = pos + vel_new * dt

    # Return new World (lists for JSON-compat)
    return {
        'positions': pos_new.tolist(),
        'velocities': vel_new.tolist(),
        'masses': m.tolist(),
        'stiffness': k.tolist(),
        'damping': c.tolist(),
        'rest_length': L0.tolist(),
    }


def main(initial_state: World):
    """Framework entry: advances the world by one step."""
    return simulate_step(initial_state)