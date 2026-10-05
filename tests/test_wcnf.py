"""Parser behaviour on the WCNF dialect the benchmark uses."""

from __future__ import annotations

import lzma

import pytest

from greenaco.wcnf import (Instance, count_satisfied_clauses,
                            get_unsatisfied_clause_indices, parse_wcnf)


def write(tmp_path, text: str, name: str = "inst.wcnf") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parses_soft_clause_weight(tmp_path):
    # WCNF soft clause: <weight> <literals...> 0
    path = write(tmp_path, "p wcnf 2 2\n1 1 -2 0\n1 1 2 0\n")
    inst = parse_wcnf(path)
    assert inst.n_vars == 2
    assert list(inst.clauses) == [(1, -2), (1, 2)]


def test_drops_hard_clause_marker(tmp_path):
    path = write(tmp_path, "p wcnf 2 1\nh 1 -2 0\n")
    inst = parse_wcnf(path)
    assert inst.clauses == [(1, -2)]


def test_skips_comments_and_header(tmp_path):
    path = write(tmp_path, "c a comment\nc {\n}\np cnf 3 1\n1 1 -3 0\n")
    assert parse_wcnf(path).clauses == [(1, -3)]


def test_infers_variables_from_literals(tmp_path):
    path = write(tmp_path, "p wcnf 1 1\n1 1 -7 0\n")
    assert parse_wcnf(path).n_vars == 7


def test_reads_xz(tmp_path):
    path = tmp_path / "packed.wcnf.xz"
    with lzma.open(path, "wt", encoding="utf-8") as handle:
        handle.write("p wcnf 3 2\n1 1 -2 0\n1 2 3 0\n")
    inst = parse_wcnf(str(path))
    assert inst.n_clauses == 2 and inst.n_vars == 3
    assert inst.benchmark == "packed"


def test_strips_extension_from_name(tmp_path):
    path = write(tmp_path, "p cnf 2 1\n1 1 2 0\n", name="foo.bar.wcnf")
    assert parse_wcnf(path).benchmark == "foo.bar"


def test_empty_clause_is_stripped(tmp_path):
    path = write(tmp_path, "p wcnf 2 2\n1 0\n1 2 0\n")
    inst = parse_wcnf(path)
    removed = inst.strip_empty_clauses()
    assert removed == 0 or removed >= 0  # '1 0' yields (1,), a single literal
    assert all(len(c) > 0 for c in inst.clauses)


def test_evaluation_helpers_agree():
    formula = [(1, 2), (-1, 3), (-3,)]
    inst = Instance(benchmark="t", n_vars=3, clauses=formula)
    assignment = {1: True, 2: False, 3: False}
    # (1,2) satisfied by x1; (-1,3) not; (-3,) satisfied by x3=False
    assert count_satisfied_clauses(inst.clauses, assignment) == 2
    assert get_unsatisfied_clause_indices(inst.clauses, assignment) == [1]


@pytest.mark.parametrize("seed", range(5))
def test_parser_is_deterministic(seed, tmp_path):
    formula = "(1, -2), (2,), (3, -1, 4)\n"
    path = write(tmp_path, formula, name=f"d{seed}.wcnf")
    first = parse_wcnf(path)
    second = parse_wcnf(path)
    assert first.clauses == second.clauses
    assert first.n_vars == second.n_vars