"""
data.py
=======
Manifest handling and instance loading.

The benchmark is defined entirely by ``data/manifest.csv``. This module reads
that file, validates it, selects a subset, and parses the corresponding WCNF
instances. It deliberately performs **no** instance selection or filtering
logic of its own: the manifest is the single source of truth, and every stage
of the pipeline consumes exactly the rows it is given.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from . import paths
from .wcnf import Instance, parse_wcnf

MANIFEST_COLUMNS = [
    "domain",
    "category",
    "benchmark",
    "file",
    "vars",
    "clauses",
    "best_known",
    "mandatory",
]

# The 4 original benchmark instances. `core` is the subset on which the method
# comparison is reported, and doubles as the fast smoke-test configuration.
CORE_INSTANCES = (
    "decision-tree-car-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree",
    "decision-tree-soybean-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree",
    "decision-tree-vote-un-formula_0.8_2021_atleast_15_max-3_reduced_incomplete_tree",
    "min-fill-MinFill_R0_myciel5",
)


@dataclass
class Benchmark:
    """A manifest row paired with its parsed formula."""

    benchmark: str
    domain: str
    category: str
    file: str
    n_vars: int
    n_clauses: int
    best_known: float
    mandatory: bool
    instance: Instance

    @property
    def label(self) -> str:
        """Short display label, e.g. ``dt-car``."""
        return short_label(self.benchmark)


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    """Read and validate the benchmark manifest."""
    path = Path(path) if path else paths.manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark manifest not found at {path}. Run scripts/prepare_data.py."
        )

    df = pd.read_csv(path)

    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing manifest column(s): {missing}")

    if df["benchmark"].duplicated().any():
        dupes = df.loc[df["benchmark"].duplicated(), "benchmark"].tolist()
        raise ValueError(f"{path}: duplicate benchmark rows: {dupes}")

    if df["best_known"].isna().any():
        bad = df.loc[df["best_known"].isna(), "benchmark"].tolist()
        raise ValueError(f"{path}: missing best_known for: {bad}")

    return df


def resolve_instance_path(row: dict, instances_dir: Path | None = None) -> Path:
    """Locate the WCNF file for a manifest row."""
    base = Path(instances_dir) if instances_dir else paths.INSTANCES_DIR
    candidate = base / str(row["file"])
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Instance file not found for {row['benchmark']}: {candidate}\n"
        f"Run scripts/prepare_data.py to fetch the benchmark instances."
    )


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA256 of a file, used to record input provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_benchmark(
    subset: str = "all",
    manifest: Path | None = None,
    instances_dir: Path | None = None,
    verify: bool = True,
) -> List[Benchmark]:
    """Load the requested subset, parsing each instance from disk.

    When ``verify`` is set (the default) each parsed instance is checked against
    the ``vars`` and ``clauses`` recorded in the manifest, and a mismatch
    raises. This guards against silently running on a different corpus than the
    one the shipped results were produced on.
    """
    df = select_subset(load_manifest(manifest), subset)

    out: List[Benchmark] = []
    for row in df.to_dict("records"):
        path = resolve_instance_path(row, instances_dir)
        inst = parse_wcnf(path, name=row["benchmark"])
        inst.strip_empty_clauses()

        if verify:
            if inst.n_vars != int(row["vars"]):
                raise ValueError(
                    f"{row['benchmark']}: parsed {inst.n_vars} vars but manifest "
                    f"declares {row['vars']}"
                )
            if inst.n_clauses != int(row["clauses"]):
                raise ValueError(
                    f"{row['benchmark']}: parsed {inst.n_clauses} clauses but "
                    f"manifest declares {row['clauses']}"
                )

        out.append(
            Benchmark(
                benchmark=row["benchmark"],
                domain=str(row["domain"]),
                category=str(row["category"]),
                file=str(row["file"]),
                n_vars=inst.n_vars,
                n_clauses=inst.n_clauses,
                best_known=float(row["best_known"]),
                mandatory=bool(row["mandatory"]),
                instance=inst,
            )
        )
    return out


def short_label(name: str) -> str:
    """Compact label used in figure axes and console output."""
    if name.startswith("decision-tree-"):
        name = "dt-" + name[len("decision-tree-"):]
    if name.startswith("min-fill-MinFill_R0_"):
        name = "minfill-" + name[len("min-fill-MinFill_R0_"):]
    return name.split("-un-formula")[0].split(".wcnf")[0][:28]


def select_subset(df: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    """Return the manifest rows belonging to ``subset``.

    ``all``  -- every instance in the manifest (the full 54-instance benchmark).
    ``core`` -- only the 4 original benchmark instances.

    Any other value is treated as a comma-separated list of benchmark names,
    which keeps ad-hoc debugging runs possible without touching the manifest.
    """
    if subset == "all":
        return df.reset_index(drop=True)
    if subset == "core":
        out = df[df["benchmark"].isin(CORE_INSTANCES)].reset_index(drop=True)
        missing = set(CORE_INSTANCES) - set(out["benchmark"])
        if missing:
            raise ValueError(f"Manifest is missing core instance(s): {sorted(missing)}")
        return out
    wanted = [s for s in subset.split(",") if s]
    out = df[df["benchmark"].isin(wanted)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No manifest rows matched subset {subset!r}")
    return out
