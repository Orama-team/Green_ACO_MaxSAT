# Green ACO for MAX-SAT

An energy-aware ant colony optimisation solver for MAX-SAT, and the reproducible
pipeline behind the experiments reported with it.

Green ACO does not choose operators round-robin. At each iteration an
**EI/J scheduler** (Expected Improvement per Joule) picks the operator with
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

The full benchmark is 50 instances and a long job:

```bash
python reproduce_all.py --subset all --stages prepare,profile,finetune,final
```

Every run is **resumable**: interrupt it and start it again, and it continues
from the last completed task rather than starting over.

---

## Reproducibility and Environmental Measurement

Energy is estimated, not metered: CodeCarbon reports the power the CPU
drew, and on this platform it does so in **TDP-based estimation mode**; it
multiplies CPU utilisation by the CPU's TDP and integrates over execution time.
Verified on this machine: `tracking_mode: machine`, `cpu_power ≈ 4.1 W` on an
idle-ish sample. 

As a result, energy is essentially proportional to wall-clock time, so how fast the code runs directly changes
how much energy a run consumes, and therefore how many iterations fit inside a
fixed budget.










| Quantity | Reproducible? |
|---|---|
| Solution quality for a given seed |  yes |
| Operator choice, and therefore every ranking |  yes |
| Quality per Joule, energy, CO2 in absolute terms |  machine-dependent |
| Energy per iteration at a given `--n-jobs` |  depends on CPU contention |



Low-load runs match the solo baseline exactly; divergence starts from a few
concurrent workers upward. `--n-jobs` is therefore recorded in every run
manifest rather than left to an automatic default. Reproduce the effect with
`python tests/probe_cpu_contention.py solo` / `loaded 7`.


---

## The benchmark

Defined entirely by `data/manifest.csv`: 50 instances. 

| column | meaning |
|---|---|
| `benchmark` | instance name; the join key in every result file |
| `file` | filename under `data/mse24/` |
| `vars`, `clauses` | parsed sizes, cross-checked at load time |
| `best_known` | best known violated clauses (MSE 2024 300 s unweighted) |
| `mandatory` | the 4 original instances forming the `core` subset |

The corpus is weighted MAX-SAT (`.wcnf.xz`, with hard clauses and soft
weights). The experimental protocol is unweighted: every clause counts 1. 
`scripts/prepare_data.py` verifies all 50 parse to exactly the sizes the
manifest declares.

### Subsets

- `--subset all` — all 50 instances.
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
| `make_figures.py` | `figures/*` | reads results, never computes |
| `verify_results.py` | — | schema, coverage and internal-consistency checks |

Artifact names encode their scope (`_all50`, `_core`).

### The comparison set

Seven methods, all re-run by this repository under the same meter and the same
green metrics:

`AG Classique` · `AG Adaptatif` · `AG + KC` · `AS-SAT` · `AS-SAT Elitiste` ·
`MMAS` · `ACS`

### Figures

| figure | reads |
|---|---|
| `fig_operator_profiles.png` | energy share, cost by regime, gain per Joule |
| `fig_operator_tests.png` | Kruskal-Wallis / Mann-Whitney on measured gains |
| `fig_ablation.png` | ablation, z-scored within each instance and budget |
| `fig_method_comparison.png` | Green ACO against the comparison methods |
| `table_summary_ranking.md` | the ranking table |

---

## Layout

```
src/greenaco/
  wcnf.py       WCNF parsing (weighted format -> unweighted problem)
  data.py       manifest handling, subset selection, instance loading
  config.py     run configuration and the tuned hyper-parameters
  operators.py  the three operators; reference backend
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
```

### One file per result, not two

`results/shipped/` stores each result once, at full scope. There is no
separate `*_core.csv` copy next to a `*_all50.csv`: when a figure needs the
core subset and no core-scoped file exists, the full-benchmark file is loaded
and filtered to the four core instances listed in the manifest. The subset is
therefore a *view* derived at load time.

The same reasoning applies to the ablation: the operator and mechanism splits
(`ablation_operators_core.csv`, `ablation_mechanisms_core.csv`) together
reconstruct the unsplit eight-configuration run exactly, so the unsplit file is
not stored.

### What each experiment covers

The ablation and comparison experiments were conducted specifically on the four core instances to focus the analysis:

| Artifact | Scope |
|---|---|
| `ablation_*_core.csv` | 4 core instances, 8 configurations x 3 budgets |
| `comparison_runs_core.csv` | 4 core instances, all methods |
| `final_runs_all50.csv`, `final_summary_all50.csv` | 50 instances |
| `operator_profiles_all50.csv` | 50 instances |

`make_figures.py --subset all` therefore reports the ablation and method comparison as skipped when scoped to the full benchmark. 

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

`--mode full` re-runs the search (Optuna TPE, seeded). 

---

## Parallelism & Sequential Execution

The pipeline runs in parallel by default using `min(4, cpu_count - 1)` workers. 

You control the exact number of parallel workers using the `--n-jobs` flag. 
- **Sequential Execution:** If you want to force a strict sequential run (no parallelism), simply use `--n-jobs 1`.
- **Parallel Execution:** Use `--n-jobs 4`, --n-jobs 8, etc. to speed up the experiments.

*Note:* Instances above 400,000 clauses automatically run serially regardless of `--n-jobs` to prevent memory exhaustion.

---

## Tests

```bash
pytest -q
```

The tests that matter most:


- `test_energy.py::test_green_metrics_match_hand_computation` — reproduces a
  known row from the original measurements exactly.
- `test_comparison.py` — every comparison method solves, and its reported
  quality is confirmed by an independent recount.
- `test_solver.py::test_same_seed_gives_same_quality` — seed reproducibility.
- `test_profiling.py` — manifest integrity and shipped artifacts.

`scripts/prepare_data.py` is the end-to-end guard: all 50 instances must parse
to the sizes the manifest declares, or it fails.

---

## Reproducibility Notes


- **Shipped Records:** The files in 
esults/shipped/ are the authoritative snapshot used to generate the paper's figures. This allows reviewers to reproduce the plots instantly without computing them from scratch.
- **Resuming:** Checkpoints are saved to 
esults/checkpoints/ dynamically. Delete this folder if you want to force a clean re-run.
- **Providing the Dataset:** The 50 source instances (MaxSAT Evaluation 2024) are expected inside data/mse24/.


## Citation

If you use this code or the Green ACO framework in your research, please cite the corresponding paper:

```bibtex
@article{greenaco2026,
  title={Green ACO: An Energy-Aware Ant Colony Optimisation for the MAX-SAT Problem with EI/J Scheduling and Thompson Sampling},
  author={Nazim Abderrahmane Aouni and Nour El Imane Elbar and Adriane Anis Khaled and Billel Moussous and Maroua Ogab and Rayane Rahmat Errahmane Smara and Zakaria Soualah Mohammed and Malika Bessedik},
  year={2026}
}
```
