# Pre-Analysis Plan

## Project Status

This repository is an experimental scaffold for studying institutional
objectives in an AI Economist-style simulation. Existing outputs should be read
as debugging baselines, not as substantive economic findings.

This pre-analysis plan defines the next experiment before producing new figures
or interpreting additional results.

## Research Question

Can a minimum social-floor metric identify severe deprivation in cases where a
distributional inequality metric, such as Gini, is not sufficient?

More narrowly:

```text
Do social-floor outcomes and Gini outcomes move differently under the same
simulated tax-and-transfer rules?
```

## Conceptual Distinction

This project does not treat equality as the primary policy goal.

- `Gini` measures relative inequality in the distribution.
- `social floor` measures whether agents fall below a minimum economic security
  threshold.

A low Gini value can describe equal outcomes around a low material level. A
social floor instead asks whether agents avoid severe deprivation.

## Hypotheses

### H1: Social-floor metrics capture deprivation more directly than Gini.

Policies may reduce `floor_gap` or `share_below_floor` even when `gini` changes
only weakly.

### H0: Social-floor metrics do not add meaningful information beyond Gini.

If `gini`, `floor_gap`, and `share_below_floor` move together across policies
and thresholds, then the social-floor metric may not identify a distinct
institutional outcome in this simulation.

## Unit of Analysis

The unit of analysis is:

```text
one simulated economy episode under one policy rule, one social floor threshold,
and one random seed
```

The planned panel unit for the next experiment is:

```text
one tax period within one simulated episode
```

## Treatments

The initial deterministic policy rules are:

- `no_tax`
- `low_progressive_tax`
- `high_progressive_tax`
- `social_floor_feedback`

These are not learned AI planners. They are deterministic baselines used to
test whether the measurement design is coherent.

## Thresholds

The social floor thresholds are measured in model coin units:

```text
50, 100, 150, 200
```

These are internal model units, not real-world money, GDP, wages, or official
poverty lines.

## Primary Outcomes

The primary outcomes are:

- `floor_gap`: average shortfall below the social floor, measured in model coin
  units
- `share_below_floor`: share of agents below the social floor

These outcomes measure minimum economic security, not equality.

## Secondary Outcomes

The secondary outcomes are:

- `gini`: inequality index
- `productivity`: mean total coin per agent
- `minimum_coin`: minimum agent coin holdings
- `mean_coin`: mean agent coin holdings
- `mean_tax_rate`: average tax rate selected by the policy rule

`gini` is a contrast metric, not the target objective.

## Required Time-Series Design

The next valid figure must use measured time-series data from the simulation:

```text
X = simulation timestep or tax period
Y = floor_gap in model coin units
```

A second valid figure may use:

```text
X = simulation timestep or tax period
Y = share_below_floor
```

No figure should use time, years, money, income, or money supply unless those
units are explicitly recorded or defined by the simulation.

## Evidence Criteria

Evidence for a distinct social-floor metric would be:

```text
floor_gap or share_below_floor changes across policy rules while gini remains
comparatively stable
```

Evidence against the distinctiveness of the social-floor metric would be:

```text
gini, floor_gap, and share_below_floor move together across policy rules,
thresholds, and tax periods
```

Evidence that the baseline rules are too weak would be:

```text
all policies produce similar trajectories for floor_gap, share_below_floor,
gini, and productivity
```

## What Would Contradict the Project Intuition

The project intuition would be weakened if:

- equality-oriented metrics predict floor outcomes well;
- Gini and social-floor outcomes move together in nearly all settings;
- social-floor policies reduce deprivation only by sharply reducing
  productivity;
- deterministic policy rules do not materially change any measured outcome;
- the simulation environment is too stylized to generate meaningful variation.

These outcomes should be treated as valid findings, not failures.

## Reporting Rules

Until time-series outcomes are recorded:

- do not present figures as substantive evidence;
- report current CSV outputs only as preliminary deterministic baselines;
- label all internal units as model coin units;
- avoid terms such as real income, GDP, poverty line, money supply, or years
  unless the model explicitly defines them.

## Next Implementation Step

Create a trajectory experiment that records, at each tax period:

- `policy`
- `social_floor`
- `seed`
- `tax_period`
- `timestep`
- `floor_gap`
- `share_below_floor`
- `gini`
- `productivity`
- `minimum_coin`
- `mean_coin`
- `mean_tax_rate`

The first acceptable plot should be generated only after this table exists.
