# Experiments

## Minimum Social Floor Policy Comparison

`social_floor_policy_comparison.py` is the main experiment for this fork.

The research question is:

```text
Can simple tax-and-transfer rules reduce the share of agents below a minimum
economic security threshold without collapsing productivity?
```

This deliberately avoids framing the planner objective as income equality. The
goal is not to make agents equal. The goal is to test whether institutions can
keep agents from falling below a minimum social floor.

```bash
python experiments/social_floor_policy_comparison.py
```

Output:

```text
results/social_floor_policy_comparison.csv
```

The default social floor is `100` coins. This is intentionally above the
starting coin endowment, so the experiment measures whether policy rules can
move agents toward a nontrivial economic security threshold.

The experiment compares four explicit tax rules:

- `no_tax`
- `low_progressive_tax`
- `high_progressive_tax`
- `social_floor_feedback`

The reported metrics include:

- `share_below_floor`
- `floor_gap`
- `floor_security`
- `minimum_coin`
- `productivity`
- `mean_coin`
- `government_utility`
- `gini`

`gini` is retained as a diagnostic metric, not as the central target.

## Interpretation Guardrail

This is not yet a trained AI planner. It is a deterministic policy comparison
inside an AI Economist-style environment. That makes the first result easier to
inspect and falsify, but it does not prove that reinforcement learning would
discover the same policy.

The next research step is to train a PPO planner under the social-floor utility
function and compare learned policies against the deterministic baselines.

## Social Floor Sensitivity

`social_floor_sensitivity.py` repeats the policy comparison across several
minimum social floor thresholds:

```bash
python experiments/social_floor_sensitivity.py
```

Outputs:

```text
results/social_floor_sensitivity.csv
results/social_floor_sensitivity_summary.csv
```

The default thresholds are:

```text
50, 100, 150, 200
```

This tests whether the baseline result is specific to one arbitrary floor or
whether the policy ranking is stable as the minimum economic security threshold
becomes more demanding.

## Legacy Objective Weight Sweep

`objective_weight_sweep.py` makes the AI Economist-style government objective
explicitly testable.

Original EconoJax fixes the government objective at:

```text
government utility = productivity * equality
```

This extension exposes `equality_weight`:

```text
government utility = productivity * (w * equality + (1 - w))
```

where:

- `w = 0`: productivity-only objective
- `w = 1`: productivity times equality objective

The first script runs a reproducible random-action baseline. This is not yet a
trained AI planner. It is a cheap sanity check and a measurable starting point
before adding PPO training.

```bash
python experiments/objective_weight_sweep.py
```

Output:

```text
results/objective_weight_sweep.csv
```

The default run is deliberately small (`50` steps, `2` seeds) so it can serve as
a reproducibility smoke test. Longer sweeps should be run after the transition
function is vectorized or JIT-compiled across episodes.

## Legacy Interpretation Guardrail

The random-action baseline is expected to show that `equality_weight` changes
the government's utility calculation, not necessarily the realized economy. The
population and government actions are sampled randomly, so policy behavior does
not adapt to the new objective.

To test whether the institutional objective changes economic outcomes, the next
step must be one of:

1. train a PPO planner for each equality weight, or
2. implement a deterministic tax rule that explicitly uses the equality weight.

Only then can the experiment compare productivity, equality, Gini, and agent
utility as outcome variables.
