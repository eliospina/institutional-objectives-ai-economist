"""Compare simple tax rules against a minimum social floor objective.

This experiment is deliberately modest. It does not claim to train an optimal
AI planner. Instead, it asks a falsifiable first question:

Can simple tax-and-transfer rules reduce the share of agents below a minimum
economic security threshold without collapsing productivity?
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from econojax import EconoJax
from econojax.util import get_gini


Policy = Callable[[EconoJax, object, float], jnp.ndarray]


def no_tax(env: EconoJax, _state: object, _social_floor: float) -> jnp.ndarray:
    return jnp.zeros(len(env.tax_bracket_cutoffs) - 1, dtype=jnp.int32)


def low_progressive_tax(
    env: EconoJax, _state: object, _social_floor: float
) -> jnp.ndarray:
    return _fit_brackets(env, [0, 2, 4])


def high_progressive_tax(
    env: EconoJax, _state: object, _social_floor: float
) -> jnp.ndarray:
    return _fit_brackets(env, [0, 5, 10])


def social_floor_feedback(
    env: EconoJax, state: object, social_floor: float
) -> jnp.ndarray:
    coin = state.inventory_coin + state.escrow_coin
    share_below_floor = float(jnp.mean(coin < social_floor))

    if share_below_floor > 0.50:
        return _fit_brackets(env, [0, 6, 12])
    if share_below_floor > 0.25:
        return _fit_brackets(env, [0, 4, 8])
    if share_below_floor > 0:
        return _fit_brackets(env, [0, 2, 5])
    return _fit_brackets(env, [0, 1, 2])


POLICIES: dict[str, Policy] = {
    "no_tax": no_tax,
    "low_progressive_tax": low_progressive_tax,
    "high_progressive_tax": high_progressive_tax,
    "social_floor_feedback": social_floor_feedback,
}


def _fit_brackets(env: EconoJax, actions: list[int]) -> jnp.ndarray:
    """Return tax actions with one entry per tax bracket.

    EconoJax encodes government tax actions in 5 percentage-point increments:
    action 2 means 10%, action 10 means 50%, and so on.
    """
    bracket_count = len(env.tax_bracket_cutoffs) - 1
    clipped = actions[:bracket_count]
    if len(clipped) < bracket_count:
        clipped += [clipped[-1] if clipped else 0] * (bracket_count - len(clipped))
    return jnp.asarray(clipped, dtype=jnp.int32)


def production_actions(env: EconoJax, state: object) -> jnp.ndarray:
    """A simple deterministic production rule for the population agents."""
    resources = np.asarray(state.inventory_resources)
    actions = []
    for agent_id, agent_resources in enumerate(resources):
        can_craft = (
            np.sum(agent_resources >= env.craft_num_resource_required)
            >= env.craft_diff_resources_required
        )
        if can_craft:
            actions.append(env.craft_action_index)
        else:
            resource_id = int((np.argmin(agent_resources) + agent_id) % env.num_resources)
            actions.append(env.gather_resource_action_index(resource_id))
    return jnp.asarray(actions, dtype=jnp.int32)


def floor_metrics(env: EconoJax, state: object, social_floor: float) -> dict[str, float]:
    coin = state.inventory_coin + state.escrow_coin
    floor_gap = jnp.mean(jnp.maximum(social_floor - coin, 0))
    normalized_floor_gap = floor_gap / max(social_floor, 1e-6)
    return {
        "productivity": float(state.productivity),
        "equality": float(state.equality),
        "gini": float(get_gini(coin)),
        "mean_coin": float(jnp.mean(coin)),
        "minimum_coin": float(jnp.min(coin)),
        "coin_std": float(jnp.std(coin)),
        "share_below_floor": float(jnp.mean(coin < social_floor)),
        "floor_gap": float(floor_gap),
        "floor_security": float(jnp.clip(1 - normalized_floor_gap, 0, 1)),
        "government_utility": float(state.utility["government"]),
        "mean_tax_rate": float(jnp.mean(state.tax_rates)),
    }


def run_episode(
    policy_name: str,
    social_floor: float,
    seed: int,
    steps: int,
    tax_period_length: int,
) -> dict[str, float | int | str]:
    key = jax.random.PRNGKey(seed)
    env = EconoJax(
        seed=seed,
        equality_weight=0.0,
        social_floor=social_floor,
        social_floor_weight=1.0,
        max_steps_in_episode=steps,
        tax_period_length=tax_period_length,
    )
    key, reset_key = jax.random.split(key)
    _, state = env.reset_env(reset_key)
    policy = POLICIES[policy_name]

    for _ in range(steps):
        key, step_key = jax.random.split(key)
        population_action = production_actions(env, state)
        government_action = policy(env, state, social_floor)
        actions = {
            f"a{i:0{env.pop_str_width}}": population_action[i]
            for i in range(env.num_population)
        }
        actions["government"] = government_action
        _, state = env.step_env(step_key, state, actions)

    return {
        "policy": policy_name,
        "social_floor": social_floor,
        "seed": seed,
        "steps": steps,
        "tax_period_length": tax_period_length,
    } | floor_metrics(env, state, social_floor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--social-floor", type=float, default=100.0)
    parser.add_argument("--tax-period-length", type=int, default=20)
    parser.add_argument(
        "--out", default="results/social_floor_policy_comparison.csv"
    )
    args = parser.parse_args()

    rows = [
        run_episode(
            policy_name=policy_name,
            social_floor=args.social_floor,
            seed=seed,
            steps=args.steps,
            tax_period_length=args.tax_period_length,
        )
        for policy_name in POLICIES
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
