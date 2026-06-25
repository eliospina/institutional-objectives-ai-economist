# Institutional Objectives in AI Economist-Style Simulations

This fork extends EconoJax to test **minimum social floor** objectives in an
AI Economist-style simulation.

The central question is not whether artificial planners can make agents equal.
The question is whether institutional rules can prevent agents from falling
below a minimum economic security threshold while preserving productivity.

The first experiment compares explicit tax-and-transfer rules:

```bash
python experiments/social_floor_policy_comparison.py
```

Results are written to:

```text
results/social_floor_policy_comparison.csv
```

This is an early laboratory baseline, not a trained AI planner. The next step is
to train a PPO government agent using the social-floor objective and compare the
learned policy against the deterministic baselines.

---

# EconoJax: A Fast & Scalable Economic Simulation in JAX

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

EconoJax is (loosely) a reimplementation of [The AI Economist](https://www.science.org/doi/10.1126/sciadv.abk2607) in JAX with a 1D observation space rather than the original 2D visual space.
With GPU support, EconoJax's transition function is over 100x times faster and agents converge over **2000x** times faster.

---

## 📦 Installation

For those using [uv](https://docs.astral.sh/uv/getting-started/installation/), it is possible to run a standard PPO implementation with default settings by directly running `uv run main.py`.

```bash
git clone git@github.com:ponseko/econojax.git
cd econojax
uv run main.py
```

Alternatively, install the project as an editable package in your favourite virtual environment software. E.g. using conda:

```bash
git clone git@github.com:ponseko/econojax.git
cd econojax
conda create -n econojax python=3.11
conda activate econojax
pip install -e .

python main.py
```

for CUDA support, additionally run `pip install jax[cuda]`.

---

## 📑 Citing

If you use EconoJax in your research or projects, please cite:

```bibtex
@article{ponse2024econojax,
  title={EconoJax: A Fast \& Scalable Economic Simulation in Jax},
  author={Ponse, Koen and Plaat, Aske and van Stein, Niki and Moerland, Thomas M},
  journal={arXiv preprint arXiv:2410.22165},
  year={2024}
}
```
