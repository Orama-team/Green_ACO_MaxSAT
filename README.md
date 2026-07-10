# Green ACO for MAX-SAT

Implementation of **Green ACO**, an energy-aware Ant Colony Optimisation
solver for MAX-SAT, developed for TP5 ("Green ACO pour MAX-SAT",
équipe **Orama**). This repository contains the actual solver
(`method/`) that the analysis notebook `TP5_presentation_finale.ipynb`
(included under `examples/`) profiles, ablates, tunes, and compares
against the TP4 baselines (GA / ACO variants).

It also adds a `budget_predictor/` component: an XGBoost model that
recommends the energy budget to allocate to a new, unseen MAX-SAT
instance, based on cheap structural features — replacing the manual
`[400, 1000, 2000] J` sweep used throughout the notebook with a
per-instance prediction.

---

## 1. Method

### 1.1 Reminder — ACO for MAX-SAT

Ant Colony Optimisation maintains a **pheromone matrix** `tau` where
`tau[v][b]` encodes the desirability of assigning boolean value `b` to
variable `v`. At every iteration, an operator modifies the current
assignment and the pheromone is updated to reinforce good assignments.
(`method/pheromone.py`)

### 1.2 Green ACO

Green ACO's central idea is to **dynamically select the operator that
maximizes the quality gain per Joule consumed**.

#### 1.2.1 The EI/J scheduler (Expected Improvement per Joule)

Implemented in `method/selection.py` (`EIJScheduler`). The scheduler
learns which operator is most efficient:

```
EI/J(o) = E[Δf(o)] / E[E(o)]
```

Statistics are maintained with an **EWMA** (Exponentially Weighted
Moving Average, smoothing factor `alpha_ewma`):

```
mu_Δf  <- alpha * mu_Δf  + (1 - alpha) * Δf_t
mu_lnE <- alpha * mu_lnE + (1 - alpha) * ln(E_t)
```

**Thompson Sampling** injects controlled exploration — instead of using
the mean directly, we sample from the learned distribution:

```
Δf~ ~ N(mu_Δf, sigma²_Δf)          E~ ~ LogNormal(mu_lnE, sigma²_lnE)
```

A **budget penalty** discourages expensive operators when the
remaining budget is low:

```
pi(o, B) = B / (B + E[E~(o)])
```

Final selection score:

```
Phi(o) = (Δf~ / E~) * pi(o, B)     =>     o* = argmax_o Phi(o)
```

#### 1.2.2 Other improvements (all implemented in `method/`)

| Improvement | Where | Description |
|---|---|---|
| Dual-regime profiling | `method/benchmark.py::run_profiling` | Every operator is profiled before any run, on `easy` (q0=0.5) and `hard` (q0=0.95) regimes |
| Sparse pheromone update | `method/pheromone.py`, `config.use_sparse_update` | O(k) update over the variables touched by the last move, instead of O(n) |
| Pheromone caching / warm-start | `method/pheromone.py::save/load`, `config.use_pheromone_cache` | Reuse tau learned on budget B1 to warm-start budget B2 |
| Adaptive evaporation | `method/solver.py` | `rho_base` in normal regime, `rho_stagnant` during stagnation |
| Frugal skip | `method/pheromone.py::update` | Skip reinforcement of moves that barely progress (`stagnation_epsilon`) |
| Anti-overrun guard | `method/solver.py` | Vetoes an operator call whose estimated cost exceeds `max_overrun_factor * remaining_budget` |
| Early stopping | `method/solver.py` | `stagnation_window`, `best_stagnation_limit`, `plateau_tolerance_pct` |

### 1.3 Operator pool (`method/operators.py`)

| Operator | Mechanism |
|---|---|
| **WalkSAT** | Targets unsatisfied clauses: greedy flip (max satisfied clauses) or random flip with probability `p_noise` |
| **Focused VNS** | Ranks variables by (# unsatisfied clauses involved) × (pheromone signal), flips the top-`k` most problematic |
| **Clause Restart Greedy** | Guided partial reconstruction: greedily reassigns the variables most involved in unsatisfied clauses |

### 1.4 Statistical testing (`method/metrics.py`)

Kruskal-Wallis, pairwise Mann-Whitney U, and Wilcoxon signed-rank tests
on the Δf produced by each operator during profiling — reproducing the
notebook's "Tests Statistiques" section exactly.

---

## 2. Experiments

### 2.1 Benchmark instances

The 4 instances used throughout the notebook and this repository's
examples (not redistributed here — bring your own `.cnf` files with
matching names):

| # | Alias | Full name | Vars | Clauses | Best known |
|---|---|---|---|---|---|
| 1 | soybean | `decision-tree-soybean-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree` | 10 503 | 89 879 | 22 |
| 2 | min-fill | `min-fill-MinFill_R0_myciel5` | 15 416 | 109 371 | 196 |
| 3 | car | `decision-tree-car-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree` | 5 959 | 110 235 | 143 |
| 4 | vote | `decision-tree-vote-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree` | 9 973 | 78 069 | 6 |

### 2.2 Hyperparameter tuning (`method/benchmark.py::run_optuna_search`)

Optuna (TPE sampler, 50 trials by default), maximizing the
quality/joule ratio, over the search space in `method/config.py`:

| Hyperparameter | Range | Role |
|---|---|---|
| `alpha_ewma` | [0.5, 0.99] | EWMA smoothing factor of the scheduler |
| `max_overrun_factor` | [1.0, 3.0] | Per-operator budget-overrun tolerance |
| `stagnation_window` | [5, 20] | Stagnation detection window |
| `stagnation_epsilon` | [0.1, 2.0] | Stagnation threshold (mean Δf) |
| `best_stagnation_limit` | [5, 30] | Max iterations without improving the incumbent |
| `plateau_tolerance_pct` | [0.001, 0.02] | Early-stop tolerance |
| `rho_base` | [0.01, 0.3] | Pheromone evaporation (normal regime) |
| `rho_stagnant` | [0.005, 0.1] | Pheromone evaporation (stagnation regime) |

If `optuna_results.csv` is unavailable, `method/config.py` falls back
to the empirical TP5 defaults (`alpha_ewma=0.9`, `max_overrun_factor=1.8`,
`stagnation_window=10`, `stagnation_epsilon=0.5`,
`best_stagnation_limit=15`, `plateau_tolerance_pct=0.005`,
`rho_base=0.1`, `rho_stagnant=0.02`).

### 2.3 Profiling

`method/benchmark.py::run_profiling` measures, per operator, per
instance, per regime (`n_runs=10` by default): mean energy cost (J)
and mean quality gain (Δf) — written to `operator_profiles.csv`.

### 2.4 Ablation study

`method/benchmark.py::run_ablation` evaluates every component of Green
ACO by toggling it off, across budgets `[400, 1000, 2000]` J:

| Configuration | WalkSAT | FocusedVNS | ClauseRestart | Caching | Profiling | Sparse update | Overrun fix |
|---|---|---|---|---|---|---|---|
| **full_v4** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| no_walksat | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| no_focused_vns | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| no_greedy_restart | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| no_caching | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| no_profiling | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| no_sparse_ph | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| no_overrun_fix | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

Results are written to `ablation_results.csv`, consumed directly by the
notebook's z-score visualization cell.

---

## 3. Baselines (TP4)

The GA (Classic / Adaptive / Knowledge-Compilation) and ACO
(AS-SAT / AS-SAT Élitiste / MMAS / ACS) baselines compared against in
the notebook (section 3) come from a separate TP4 codebase and are
**out of scope** for this repository — the notebook only *reads*
`green_metrics_aco_methods.csv` / `green_metrics_ga_methods.csv` to
plot the comparison.

---

## 4. Results & interpretation

`method/benchmark.py::run_best_params` runs Green ACO v4 with the
best/default hyperparameters and writes `best_params_results.csv`,
which feeds the notebook's comparison bar charts (4.1), convergence
curves (4.2), and final ranking table (5.1).

Key conclusions from the ablation study, reproduced by design in this
implementation:

- **`clause_restart_greedy` is the riskiest component without a
  guard.** It produces large quality jumps (+1000 to +3700 Δf per
  call) but at high cost (800–3600 J). Without `max_overrun_factor`,
  a single call can massively overrun the budget — this is exactly
  what `solver.py`'s anti-overrun veto prevents.
- **The EI/J scheduler earns its cost on constrained budgets** (400 J),
  where operator cost profiles are most heterogeneous.
- **Removing WalkSAT** hurts most on instances with very few
  unsatisfied clauses near the optimum (e.g. `vote`, best known = 6),
  exactly WalkSAT's regime of surgical clause-targeting.
- **Frugal skip + adaptive evaporation** save 17–27% of pheromone
  updates during stagnation without measurable quality loss.

---

## 5. Budget predictor (`budget_predictor/`)

A complementary XGBoost regressor that predicts the energy budget
(Joules) an unseen instance needs to reach a target quality, from
structural features alone (variable/clause counts, clause-length and
variable-degree distributions, Horn-clause fraction, etc. — see
`budget_predictor/feature_extractor.py`), trained on the
benchmark pickle `Bechmarks/final_merged_100_per_benchmark_3cat.pck`.
The workflow is split into two phases: first export a tabular CSV from
the pickle, then train on that CSV. The export step derives the target
budget by running Green ACO on each instance at the standard budget
grid and selecting the smallest budget that reaches the requested
quality threshold.

```bash
# phase 1: export the tabular dataset from the pickle
python examples/build_budget_tabular_dataset.py \
  --pck-path Bechmarks/final_merged_100_per_benchmark_3cat.pck \
  --out-csv models/budget_tabular_dataset.csv \
  --limit 200 \
  --flush-every 100

# --limit samples across categories/benchmarks by default;
# use --limit-strategy contiguous for the old first-N-records behavior.
# Completed rows are appended to the CSV every --flush-every instances,
# so Ctrl+C leaves a usable partial dataset on disk.
# Re-running the command resumes from existing rows by default; pass
# --no-resume if you want to overwrite the CSV.

# phase 2: train the model from the exported CSV (default test split: 20%)
python -m budget_predictor.trainer \
  --dataset-csv models/budget_tabular_dataset.csv \
  --model-dir models

# optional: train + predict a new CNF instance in one step
python examples/train_and_predict_budget.py \
  models/budget_tabular_dataset.csv \
  path/to/new_instance.cnf

# predict a budget for a new instance
python -m budget_predictor.inference --cnf path/to/new_instance.cnf --snap-to-tested
```

With only 4 profiled instances (as in the notebook), this model is a
proof of concept — accuracy improves as more instances are profiled
and added to the training set.

---

## Repository structure

```
method/
  solver.py       # main Green ACO loop (GreenACOSolver)
  operators.py    # WalkSAT, Focused VNS, Clause Restart Greedy
  selection.py    # EI/J scheduler (EWMA + Thompson sampling + budget penalty)
  energy.py       # Joule accounting + CO2 footprint
  pheromone.py    # tau matrix, evaporation, sparse update, warm-start cache
  parallel.py     # multi-process helpers for profiling/ablation/Optuna
  metrics.py      # Kruskal-Wallis / Mann-Whitney / Wilcoxon, ranking table
  benchmark.py    # profiling, ablation, best-params, Optuna harnesses
  parser.py       # DIMACS CNF parser
  utils.py        # assignment helpers
  config.py       # hyperparameters, ablation configs, search space
budget_predictor/
  cnf_parser.py, feature_extractor.py, preprocessing.py, dataset.py,
  trainer.py, predictor.py, model.py, inference.py
models/           # trained XGBoost model artifacts (generated)
examples/         # runnable end-to-end scripts + the original notebook
README.md
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python examples/run_single_instance.py path/to/instance.cnf --budget 1000
```
