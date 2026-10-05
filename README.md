# Green ACO for MAX-SAT

An energy-aware ant colony optimisation solver for MAX-SAT, and the reproducible
pipeline behind the experiments reported with it.

Green ACO does not choose operators round-robin. At each iteration an
**EI/J scheduler** — Expected Improvement per Joule — picks the operator with
the best ratio of expected quality gain to expected energy cost, biased against
expensive operators as the remaining budget shrinks. Energy is the binding
constraint of the whole method, so the choice of operator *is* the method.

---

## Quick start

```bash
pip install -r requirements.txt

# Build every figure from the results shipped with the repository (seconds)
python reproduce_all.py --use-shipped --subset core

# Reproduce the experiments, then the figures
python reproduce_all.py --subset core
```

The full benchmark is 54 instances and a long job:

```bash
python reproduce_all.py --subset all --stages prepare,profile,finetune,final
```

Every run is **resumable**: interrupt it and start it again, and it continues
from the last completed task rather than starting over.

---

## What is reproducible, and what is not

This matters more than anything else here, so it comes first.

Energy is **estimated, not metered**: CodeCarbon reports the power the CPU
drew, and on this platform it does so in **TDP-based estimation mode** — it
multiplies CPU utilisation by the CPU's TDP and integrates over execution time.
Verified on this machine: `tracking_mode: machine`, `cpu_power ≈ 4.1 W` on an
idle-ish sample. There is no RAPL hardware counter here, so the reported
joules are a *model* of consumption rather than a direct measurement.

That has a consequence worth stating plainly: **energy is essentially
proportional to wall-clock time**, so how fast the code runs directly changes
how much energy a run consumes, and therefore how many iterations fit inside a
fixed budget. See `docs/BACKENDS.md`.

Being an estimate also means the figures are comparative proxies — valid
across algorithms, instances and budgets because all were measured under the
same methodology, but not absolute statements about electricity drawn.

| Quantity | Reproducible? |
|---|---|
| Solution quality for a given seed | ✅ yes, exactly |
| Operator choice, and therefore every ranking | ✅ yes |
| Quality per Joule, energy, CO2 in absolute terms | ❌ machine-dependent |
| Energy per iteration at a given `--n-jobs` | ⚠️ depends on CPU contention |

Contention is not a footnote. A contended core spends more joules per unit of
work, so **fewer iterations fit inside the same budget** and the result
changes. Measured on `ramsey-ram_k3_n12`, budget 400 J, seed 42:

| background load | violated clauses | iterations | J/iteration |
|---|---|---|---|
| none | 12 | 30 | 13.8 |
| 1 process | 12 | 28 | 14.3 |
| 3 processes | 13 | 24 | 17.1 |
| 7 processes | 13 | 10 | 41.9 |

Low-load runs match the solo baseline exactly; divergence starts from a few
concurrent workers upward. `--n-jobs` is therefore **recorded in every run
manifest** rather than left to an automatic default. Reproduce the effect with
`python tests/probe_cpu_contention.py solo` / `loaded 7`.

Because energy tracks wall-clock time, **implementation speed is part of the
method's behaviour under an energy budget**: a faster implementation completes
more iterations inside the same budget and therefore returns a different, often
better, solution. See `docs/BACKENDS.md`.

---

## The benchmark

Defined entirely by `data/manifest.csv` — 54 instances. No stage of the
pipeline adds, drops, filters or reorders instances; the manifest is read as
given and validated, not derived.

| column | meaning |
|---|---|
| `benchmark` | instance name; the join key in every result file |
| `file` | filename under `data/mse24/` |
| `vars`, `clauses` | parsed sizes, cross-checked at load time |
| `best_known` | best known violated clauses (MSE 2024 300 s unweighted) |
| `mandatory` | the 4 original instances forming the `core` subset |

The corpus is **weighted** MAX-SAT (`.wcnf.xz`, with hard clauses and soft
weights). The experimental protocol is **unweighted**: every clause counts 1,
and the parser discards weights and `h` markers accordingly. This matches the
instances the original measurements were computed on, and
`scripts/prepare_data.py` verifies all 54 parse to exactly the sizes the
manifest declares.

### Subsets

- `--subset all` — all 54 instances.
- `--subset core` — the 4 original instances (`car`, `soybean`, `vote`,
  `min-fill`/`myciel5`). These are the instances the comparison methods were
  run on, so the method comparison is scoped to them. Also the fast path for a
  first run.
---

## Pipeline

```
prepare -> profile -> finetune -> final -> ablation -> comparison
        -> figures -> verify
```

Each stage is its own script and runs standalone; `reproduce_all.py` is a
convenience wrapper, not the only way in.

| script | writes | purpose |
|---|---|---|
| `prepare_data.py` | — | validate the manifest and every instance file |
| `run_profiling.py` | `operator_profiles_<subset>.csv` | measure each operator in isolation, per regime |
| `run_finetuning.py` | `finetune_d_best_params_<subset>.csv` | reuse (`--mode frozen`) or re-derive (`--mode full`) the hyper-parameters |
| `run_final.py` | `final_runs_<subset>.csv`, `final_summary_<subset>.csv` | the main sweep: instances × budgets × repeats |
| `run_ablation.py` | `ablation_operators_<subset>.csv`, `ablation_mechanisms_<subset>.csv` | remove one component at a time |
| `run_comparison.py` | `comparison_runs_<subset>.csv`, `comparison_ranking_<subset>.csv` | the comparison methods, plus a like-for-like Green ACO row |
| `make_figures.py` | `figures/*` | **reads** results, never computes |
| `verify_results.py` | — | schema, coverage and internal-consistency checks |

Artifact names encode their scope (`_all54`, `_core`) so results from different
subsets never silently overwrite one another.

### The comparison set

Seven methods, all re-run by this repository under the same meter and the same
green metrics — so every row of the comparison table comes from one machine in
one session, rather than mixing fresh numbers with historical rows:

`AG Classique` · `AG Adaptatif` · `AG + KC` · `AS-SAT` · `AS-SAT Elitiste` ·
`MMAS` · `ACS`

### Figures

| figure | reads |
|---|---|
| `fig_operator_profiles.png` | energy share, cost by regime, gain per Joule |
| `fig_operator_tests.png` | Kruskal-Wallis / Mann-Whitney on measured gains |
| `fig_ablation.png` | ablation, z-scored within each instance and budget |
| `fig_convergence.png` | search depth and quality against budget |
| `fig_method_comparison.png` | Green ACO against the comparison methods |
| `table_summary_ranking.md` | the ranking table |

---

## Layout

```
src/greenaco/
  wcnf.py       WCNF parsing (weighted format -> unweighted problem)
  data.py       manifest handling, subset selection, instance loading
  config.py     run configuration and the tuned hyper-parameters
  operators.py  the three operators; reference and indexed backends
  scheduler.py  the EI/J scheduler (EWMA, Thompson sampling, budget penalty)
  solver.py     the Green ACO search loop
  energy.py     CodeCarbon measurement and CO2 accounting
  profile.py    dual-regime operator profiling
  metrics.py    statistical tests, ranking, context standardisation
  runner.py     parallelism, checkpointing, provenance manifests
  comparison/   the GA and ACO variants used for the comparison
scripts/        one script per stage, plus the figure builder
data/           manifest, instance files, comparison calibration artifacts
configs/        tuned parameters
results/        generated CSVs; results/shipped/ holds the recorded results
figures/        generated figures and tables
tests/          the test suite, run with `pytest`
docs/           BACKENDS.md — why the two backends exist
```

### One file per result, not two

`results/shipped/` stores each result **once**, at full scope. There is no
separate `*_core.csv` copy next to a `*_all54.csv`: when a figure needs the
core subset and no core-scoped file exists, the full-benchmark file is loaded
and filtered to the four core instances listed in the manifest. The subset is
therefore a *view* derived at load time, which is what stops the two copies
drifting apart as results change.

The same reasoning applies to the ablation: the operator and mechanism splits
(`ablation_operators_core.csv`, `ablation_mechanisms_core.csv`) together
reconstruct the unsplit eight-configuration run exactly, so the unsplit file is
not stored.

### What each experiment covers

Two experiments were run on the four core instances only, and the repository
does not pretend otherwise:

| Artifact | Scope |
|---|---|
| `ablation_*_core.csv` | 4 core instances, 8 configurations x 3 budgets |
| `comparison_runs_core.csv` | 4 core instances, all methods |
| `final_runs_all54.csv`, `final_summary_all54.csv` | 54 instances |
| `operator_profiles_all54.csv` | 54 instances |

`make_figures.py --subset all` therefore reports the ablation and method
comparison as skipped rather than silently substituting core numbers into a
figure captioned for the full benchmark.
---

## Hyper-parameters

`configs/best_params.json` holds the tuned values, obtained by the finetuning
stage and recorded in `results/shipped/finetune_d_best_params.csv`:

| parameter | value |
|---|---|
| `alpha_ewma` | 0.632961 |
| `max_overrun_factor` | 1.176985 |
| `stagnation_window` | 8 |
| `stagnation_epsilon` | 0.185932 |
| `best_stagnation_limit` | 13 |
| `plateau_tolerance_pct` | 0.008385 |
| `rho_base` | 0.250334 |
| `rho_stagnant` | 0.038892 |

`--mode full` re-runs the search (Optuna TPE, seeded). It searches, so it may
land on different values; what was actually used is always recorded.

---

## Parallelism

Default `min(4, cpu_count - 1)` workers. Set `--n-jobs` explicitly and keep it
fixed for runs that are meant to be comparable. Instances above 400 000 clauses
run **serially**: they are the slowest and the most memory-hungry, and running
several at once risks exhausting RAM. Each task writes an atomic JSON
checkpoint, so an interrupted run resumes rather than restarting.

---

## Tests

```bash
pytest -q
```

The tests that matter most:

- `test_operators.py::test_backends_agree_exactly` — the indexed backend is
  required to return **identical assignments and gains** under a fixed seed.
- `test_energy.py::test_green_metrics_match_hand_computation` — reproduces a
  known row from the original measurements exactly.
- `test_comparison.py` — every comparison method solves, and its reported
  quality is confirmed by an independent recount.
- `test_solver.py::test_same_seed_gives_same_quality` — seed reproducibility.
- `test_profiling.py` — manifest integrity and shipped artifacts.

`scripts/prepare_data.py` is the end-to-end guard: all 54 instances must parse
to the sizes the manifest declares, or it fails.

---

## Assumptions, stated plainly

- **Energy is measured.** Absolute Joule and CO2 figures are properties of the
  machine that produced them. They are not comparable across machines, and not
  comparable across different `--n-jobs` on the same machine.
- **The problem is unweighted.** Weights and hard-clause markers in the source
  `.wcnf.xz` files are discarded, matching the original measurements.
- **`results/shipped/` are historical records.** They were produced on the
  original hardware. Re-running regenerates `results/`; the shipped copies are
  kept so figures can be built without a multi-hour sweep, and so the recorded
  numbers remain inspectable.
- **`results/checkpoints/` is disposable.** Delete it to force a clean re-run.
- **`data/mse24/` is git-ignored.** Fetch the instances from the MaxSAT
  Evaluation 2024 distribution and place the 54 files named in the manifest.
