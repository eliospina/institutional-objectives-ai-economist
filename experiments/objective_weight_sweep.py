"""Sweep government equality weights in EconoJax.

This is a small, measurable extension point for AI Economist-style simulations:
instead of treating the government's objective as fixed, expose the equality
weight and compare outcomes under identical simple policies.

The script intentionally does not train a reinforcement-learning policy yet.
It evaluates a reproducible random-action baseline so the first experiment is
cheap, inspectable, and easy to falsify.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from econojax import EconoJax
from econojax.util import get_gini


def _sample_actions(env: EconoJax, key):
    actions = {}
    for agent_id, space in env.action_space.items():
        key, subkey = jax.random.split(key)
        if agent_id == "government":
            actions[agent_id] = jax.random.randint(
                subkey, shape=space.nvec.shape, minval=0, maxval=jnp.asarray(space.nvec)
            )
        else:
            actions[agent_id] = jax.random.randint(
                subkey, shape=(), minval=0, maxval=space.n
            )
    return actions, key


def run_episode(equality_weight: float, seed: int, steps: int) -> dict[str, float | int]:
    key = jax.random.PRNGKey(seed)
    env = EconoJax(
        seed=seed,
        equality_weight=equality_weight,
        max_steps_in_episode=steps,
        tax_period_length=max(2, min(100, steps // 2)),
    )
    key, reset_key = jax.random.split(key)
    _, state = env.reset_env(reset_key)

    for _ in range(steps):
        key, action_key, step_key = jax.random.split(key, 3)
        actions, key = _sample_actions(env, action_key)
        _, state = env.step_env(step_key, state, actions)

    coin = state.inventory_coin + state.escrow_coin
    utility_values = jnp.asarray(
        [state.utility[f"a{i:0{env.pop_str_width}}"] for i in range(env.num_population)]
    )
    return {
        "equality_weight": equality_weight,
        "seed": seed,
        "steps": steps,
        "productivity": float(state.productivity),
        "equality": float(state.equality),
        "gini": float(get_gini(coin)),
        "mean_agent_utility": float(jnp.mean(utility_values)),
        "mean_coin": float(jnp.mean(coin)),
        "coin_std": float(jnp.std(coin)),
        "government_utility": float(state.utility["government"]),
        "mean_tax_rate": float(jnp.mean(state.tax_rates)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", default="results/objective_weight_sweep.csv")
    args = parser.parse_args()

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = [
        run_episode(weight, seed, args.steps)
        for weight in weights
        for seed in range(args.seeds)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
