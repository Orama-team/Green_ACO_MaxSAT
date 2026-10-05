"""Tests for the DIMACS CNF reader used by the optional budget predictor."""

import pytest

from greenaco.cnf import CNFInstance, build_var_clause_index, parse_dimacs_cnf


def write(tmp_path, text, name="instance.cnf"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parses_header_and_clauses(tmp_path):
    inst = parse_dimacs_cnf(
        write(tmp_path, "c comment\np cnf 3 4\n1 -2 0\n2 3 0\n-1 -3 0\n2 0\n"))
    assert inst.n_vars == 3
    assert inst.n_clauses == 4
    assert inst.clauses == [(1, -2), (2, 3), (-1, -3), (2,)]


def test_clause_may_span_several_lines(tmp_path):
    """DIMACS terminates a clause with 0, not with a newline."""
    inst = parse_dimacs_cnf(write(tmp_path, "p cnf 2 2\n1\n2 0\n-1 0\n"))
    assert inst.clauses == [(1, 2), (-1,)]


def test_comment_lines_are_ignored_anywhere(tmp_path):
    inst = parse_dimacs_cnf(
        write(tmp_path, "p cnf 2 2\n1 0\nc mid-file comment\n2 0\n"))
    assert inst.clauses == [(1,), (2,)]


def test_missing_header_is_rejected(tmp_path):
    """A file with no p line is not a CNF; silently returning nothing hides it."""
    with pytest.raises(ValueError, match="p cnf"):
        parse_dimacs_cnf(write(tmp_path, "1 -2 0\n"))


def test_name_defaults_to_file_stem(tmp_path):
    inst = parse_dimacs_cnf(write(tmp_path, "p cnf 1 1\n1 0\n", name="foo.cnf"))
    assert inst.name == "foo"


def test_var_clause_index_maps_var_to_clause_positions():
    inst = CNFInstance(name="x", n_vars=3, n_clauses=3,
                       clauses=[(1, -2), (2, 3), (-1, -3)])
    assert build_var_clause_index(inst) == {1: [0, 2], 2: [0, 1], 3: [1, 2]}


def test_index_omits_variables_that_do_not_occur():
    """Declared but unmentioned variables are absent, so counts reflect the formula."""
    inst = CNFInstance(name="x", n_vars=10, n_clauses=1, clauses=[(3,)])
    assert set(build_var_clause_index(inst)) == {3}


def test_index_treats_repeated_literals_as_one_occurrence():
    inst = CNFInstance(name="x", n_vars=1, n_clauses=1, clauses=[(1, 1)])
    assert build_var_clause_index(inst) == {1: [0]}