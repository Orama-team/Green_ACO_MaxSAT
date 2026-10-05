# Backend equivalence and the energy-measurement subtlety

Two evaluation backends are available, selected by `config.backend`:

| backend | behaviour | cost |
|---|---|---|
| `rescan` (default) | the reference implementation | full formula rescan per candidate |
| `indexed` | **bit-identical results** | uses a variable→clauses index and incremental score deltas |

`tests/test_operators.py` asserts the two agree exactly under a fixed seed, and
this was verified directly: on a 150-variable/600-clause synthetic instance all
three operators returned identical assignments and identical `delta_f`.

## Measured cost

`clause_restart_greedy` on `min-fill-MinFill_R0_myciel5` (15 416 variables,
109 371 clauses), one operator call:

| operator | rescan | indexed | speedup |
|---|---|---|---|
| walksat | 2.77 s | 2.19 s | 1.3× |
| focused_vns | 0.31 s | 0.35 s | 0.9× |
| clause_restart_greedy | 214.1 s | 0.36 s | **592×** |

The reference cost implies **≈12 hours** for a 200-step profiling pass on that
one operator/instance pair, which is why the pipeline cannot complete on large
instances without the indexed path.

## The caveat that matters

**The backends are logically identical, but a full run still gives different
results between them.** This is not an inconsistency — it follows from energy
being *measured from wall-clock CPU draw*:

| backend | iterations | energy | J/iteration | violated clauses |
|---|---|---|---|---|
| rescan | 2 | 762.7 J | 381.3 | 101 |
| indexed | 17 | 283.2 J | 16.7 | 1 |

The slower backend burns the 400 J budget in two operator calls and stops; the
faster one completes seventeen. The budget loop is driven by measured energy, so
**a change in implementation speed is a change in the method's behaviour under
the energy constraint.**

Consequences for the reported work:

- `rescan` is the faithful reproduction of the original measurements, which were
  produced with the reference implementation on this class of hardware.
- `indexed` is the same algorithm, and is the only tractable option on the full
  54-instance benchmark, but its energy figures reflect a faster implementation
  and are **not** comparable with the original `rescan`-based numbers.

Which backend is "the method" is a scientific decision, not an engineering one,
and is recorded in `results/MANIFEST.json` for every run.