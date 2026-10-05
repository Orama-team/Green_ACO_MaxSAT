"""
comparison/ga.py
================
Genetic-algorithm variants used in the method comparison.

Ported from the earlier GA codebase so that every method in the comparison
table is produced by this repository and measured under identical conditions.

All four share the same interface: a population of bit strings (0-indexed,
variable ``v`` at position ``v - 1``), evolved until a generation, quality or
time limit is reached, and they return ``(best_bits, best_fitness, quality,
elapsed)``.

  ClassicGA   binary tournament, 1-point crossover, bit-flip, elitism
  AdaptiveGA  Pc/Pm adapted to the convergence state (Fu et al., 2018)
  KCBasedGA   semantic distance + crowding replacement, incremental fitness

``TUNED_PARAMS`` below holds the calibrated settings used for the comparison,
taken from the GA calibration artefacts in ``data/comparison/``.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Dict

# Calibrated settings for the comparison. Source: data/comparison/
# ga_best_params_summary.csv, which mirrors ga_best_params_per_method.pkl.
#
# `mutation_prob: null` means the 1/n default is used: a literal NaN would make
# the mutation test `random() < NaN` always false, silently disabling mutation.
TUNED_PARAMS: Dict[str, Dict] = {
    "AG Classique": {"pop_size": 50, "max_gen": 500, "crossover_prob": 0.95,
                     "mutation_prob": None},
    "AG Adaptatif": {"pop_size": 50, "max_gen": 500, "P1": 0.8, "P2": 0.05},
    "AG + KC": {"pop_size": 50, "max_gen": 500, "crossover_prob": 0.75,
                "mutation_prob": None},
}


def build_ga(method_name: str, timeout: int = 300, seed: int = 42):
    """Instantiate the GA class named ``method_name`` with its tuned settings."""
    params = dict(TUNED_PARAMS[method_name])
    params.update({"timeout": timeout, "seed": seed})
    return GA_CLASS_MAP[method_name](**params)


GA_CLASS_MAP: Dict[str, type] = {}  # populated below, after the classes exist


def make_ga_solver(ga_class, **ga_kwargs):
    """Adapt a GA class to the ``fn(formula, n_vars)`` signature.

    Returns a callable producing ``(assignment_dict, stats_dict)`` with the
    1-based variable keys the rest of the pipeline uses.
    """

    def _solver(formula, n_vars, _shared=None):
        n_clauses = len(formula)
        adapter = MaxSATInstanceAdapter(formula, n_vars, n_clauses)
        ga = ga_class(**ga_kwargs)

        bits, best_fit, _quality, elapsed = ga.solve(adapter)
        assignment = {v + 1: bits[v] for v in range(n_vars)}

        stats = {
            "qualite_solution": best_fit,
            "temps_exec": elapsed,
            "nbr_mouvements_ameliorants": 0,
            "nbr_mouvements_deteriorants": 0,
            "nbr_mouvements_neutres": 0,
            "stagnation_moyenne": 0.0,
        }
        if _shared is not None:
            _shared["best_score"] = best_fit
            _shared["best_assignment"] = assignment
            _shared["stats"] = stats
        return assignment, stats

    return _solver

class MaxSATInstanceAdapter:
    """
    Encapsule (formula, n_vars, n_clauses) dans l'interface attendue par les classes AG.
    formula : liste de clauses, chaque clause est une liste de littéraux (int).
    Variables 1-indexed dans formula, bits 0-indexed dans les AG.
    """
    def __init__(self, formula, n_vars, n_clauses):
        self.clauses    = [tuple(c) for c in formula]
        self.n_vars     = n_vars
        self.n_clauses  = n_clauses
        self.var_to_clauses = defaultdict(list)
        for idx, clause in enumerate(self.clauses):
            for lit in clause:
                self.var_to_clauses[abs(lit)].append(idx)

    def evaluate(self, assignment):
        """assignment : liste de bool 0-based (assignment[0] = variable 1)."""
        count = 0
        for clause in self.clauses:
            for lit in clause:
                var = abs(lit) - 1
                val = assignment[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    count += 1
                    break
        return count

    def quality(self, assignment):
        return self.evaluate(assignment) / self.n_clauses

    def random_assignment(self):
        return [random.random() < 0.5 for _ in range(self.n_vars)]


# ── Méthode AG-1 : AG Classique (Holland, 1975) ───────────────────────────────

class ClassicGA:
    def __init__(self, pop_size=100, crossover_prob=0.85,
                 mutation_prob=None, max_gen=5000, timeout=300, seed=42):
        self.pop_size       = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob  = mutation_prob
        self.max_gen        = max_gen
        self.timeout        = timeout
        self.seed           = seed

    def _init_population(self, n):
        return [[random.random() < 0.5 for _ in range(n)]
                for _ in range(self.pop_size)]

    def _tournament_select(self, pop, fits):
        i1, i2 = random.randrange(len(pop)), random.randrange(len(pop))
        return pop[i1] if fits[i1] >= fits[i2] else pop[i2]

    def _crossover(self, p1, p2):
        k = random.randint(1, len(p1) - 1)
        return p1[:k] + p2[k:], p2[:k] + p1[k:]

    def _mutate(self, ind, pm):
        return [not b if random.random() < pm else b for b in ind]

    def solve(self, instance):
        random.seed(self.seed)
        n  = instance.n_vars
        pm = self.mutation_prob if self.mutation_prob else 1.0 / n

        pop  = self._init_population(n)
        fits = [instance.evaluate(ind) for ind in pop]

        best_idx             = max(range(len(fits)), key=lambda i: fits[i])
        best_ind, best_fit   = pop[best_idx][:], fits[best_idx]
        start                = time.time()

        for _ in range(self.max_gen):
            if time.time() - start > self.timeout or best_fit == instance.n_clauses:
                break
            new_pop, new_fits = [best_ind[:]], [best_fit]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fits)
                p2 = self._tournament_select(pop, fits)
                if random.random() < self.crossover_prob:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1, c2 = self._mutate(c1, pm), self._mutate(c2, pm)
                new_pop.extend([c1, c2])
                new_fits.extend([instance.evaluate(c1), instance.evaluate(c2)])
            pop, fits = new_pop[:self.pop_size], new_fits[:self.pop_size]
            idx = max(range(len(fits)), key=lambda i: fits[i])
            if fits[idx] > best_fit:
                best_fit, best_ind = fits[idx], pop[idx][:]

        elapsed = time.time() - start
        return best_ind, best_fit, best_fit / instance.n_clauses, elapsed



# ── Méthode AG-2 : AG Adaptatif (Fu et al., 2018) ────────────────────────────

class AdaptiveGA:
    LAMBDA = 1e-9

    def __init__(self, pop_size=100, max_gen=5000, timeout=300,
                 P1=0.9, P2=0.1, seed=42):
        self.pop_size = pop_size
        self.max_gen  = max_gen
        self.timeout  = timeout
        self.P1       = P1
        self.P2       = P2
        self.seed     = seed

    def _detect_premature(self, fits, f_max, f_mean, f_min):
        ratio = (f_max - f_mean) / (f_mean - f_min + self.LAMBDA)
        M1    = sum(1 for f in fits if f > f_mean)
        return (ratio < 1.0) and (M1 > len(fits) - M1)

    def _adaptive_pc(self, gen, f_max, f_mean, f_min, f_p, f_opt, premature):
        early = gen <= 0.75 * self.max_gen
        L = self.LAMBDA
        if premature:
            if early:
                return min(0.8 * (f_max - f_mean) / (f_max - f_min + L), self.P1)
            return min(0.8 * (f_opt - f_max) / (f_opt - f_mean + L), self.P1) if f_opt > f_max else self.P1
        if early:
            return max(0.3 * (f_p - f_min) / (f_max - f_min + L), 0.3)
        return max(0.3 * (f_opt - f_max) / (f_opt - f_p + L), 0.3) if f_opt > f_max else 0.3

    def _adaptive_pm(self, gen, f_max, f_mean, f_min, f_ind, f_opt, premature):
        early = gen <= 0.75 * self.max_gen
        L = self.LAMBDA
        if premature:
            if early:
                return min(0.1 * (f_max - f_mean) / (f_max - f_min + L), self.P2)
            return min(0.1 * (f_opt - f_max) / (f_opt - f_mean + L), self.P2) if f_opt > f_max else self.P2
        if early:
            return max(0.09 * (f_ind - f_min) / (f_max - f_min + L), 0.01)
        return max(0.09 * (f_opt - f_max) / (f_opt - f_ind + L), 0.01) if f_opt > f_max else 0.01

    def _crossover(self, p1, p2):
        k = random.randint(1, len(p1) - 1)
        return p1[:k] + p2[k:], p2[:k] + p1[k:]

    def _mutate(self, ind, pm):
        return [not b if random.random() < pm else b for b in ind]

    def _greedy_improve(self, ind, instance, fit):
        best_gain, best_var = 0, -1
        for var in range(instance.n_vars):
            ind[var] = not ind[var]
            gain = instance.evaluate(ind) - fit
            if gain > best_gain:
                best_gain, best_var = gain, var
            ind[var] = not ind[var]
        if best_var >= 0:
            ind[best_var] = not ind[best_var]
            return ind, fit + best_gain
        return ind, fit

    def solve(self, instance):
        random.seed(self.seed)
        n, f_opt = instance.n_vars, instance.n_clauses

        pop  = [[random.random() < 0.5 for _ in range(n)] for _ in range(self.pop_size)]
        fits = [instance.evaluate(ind) for ind in pop]

        best_idx             = max(range(len(fits)), key=lambda i: fits[i])
        best_ind, best_fit   = pop[best_idx][:], fits[best_idx]
        start                = time.time()

        for gen in range(1, self.max_gen + 1):
            if time.time() - start > self.timeout or best_fit == f_opt:
                break
            f_max, f_min = max(fits), min(fits)
            f_mean       = sum(fits) / len(fits)
            premature    = self._detect_premature(fits, f_max, f_mean, f_min)

            sorted_idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
            half = self.pop_size // 2
            for i in range(half):
                pop[sorted_idx[half + i]]  = pop[sorted_idx[i]][:]
                fits[sorted_idx[half + i]] = fits[sorted_idx[i]]

            new_pop, new_fits = [best_ind[:]], [best_fit]
            while len(new_pop) < self.pop_size:
                i1, i2 = random.randrange(self.pop_size), random.randrange(self.pop_size)
                p1, f_p1 = (pop[i1], fits[i1]) if fits[i1] >= fits[i2] else (pop[i2], fits[i2])
                i3, i4 = random.randrange(self.pop_size), random.randrange(self.pop_size)
                p2 = pop[i3] if fits[i3] >= fits[i4] else pop[i4]

                pc = self._adaptive_pc(gen, f_max, f_mean, f_min, f_p1, f_opt, premature)
                pm = self._adaptive_pm(gen, f_max, f_mean, f_min, f_p1, f_opt, premature)

                if random.random() < pc:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1, c2 = self._mutate(c1, pm), self._mutate(c2, pm)
                new_pop.extend([c1, c2])
                new_fits.extend([instance.evaluate(c1), instance.evaluate(c2)])

            pop, fits = new_pop[:self.pop_size], new_fits[:self.pop_size]

            idx = max(range(len(fits)), key=lambda i: fits[i])
            if fits[idx] >= best_fit:
                candidate, new_fit = self._greedy_improve(pop[idx][:], instance, fits[idx])
                if new_fit > best_fit:
                    best_ind, best_fit = candidate, new_fit

        elapsed = time.time() - start
        return best_ind, best_fit, best_fit / instance.n_clauses, elapsed



# ── Méthode AG-3 : AG + Compilation de Connaissances (Berden et al., 2022) ───

class _Individual:
    __slots__ = ['bits', 'sat_vec', 'fitness']
    def __init__(self, bits, sat_vec=None, fitness=None):
        self.bits, self.sat_vec, self.fitness = bits, sat_vec, fitness


class KCBasedGA:
    def __init__(self, pop_size=100, crossover_prob=0.85,
                 mutation_prob=None, max_gen=5000, timeout=300, seed=42):
        self.pop_size       = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob  = mutation_prob
        self.max_gen        = max_gen
        self.timeout        = timeout
        self.seed           = seed

    def _full_eval(self, bits, instance):
        sat_vec, fitness = [], 0
        for clause in instance.clauses:
            sat = any((lit > 0 and bits[abs(lit)-1]) or (lit < 0 and not bits[abs(lit)-1])
                      for lit in clause)
            sat_vec.append(sat)
            if sat:
                fitness += 1
        return sat_vec, fitness

    def _incremental_update(self, ind, var_idx, instance):
        ind.bits[var_idx] = not ind.bits[var_idx]
        for cidx in instance.var_to_clauses[var_idx + 1]:
            old_sat = ind.sat_vec[cidx]
            new_sat = any((lit > 0 and ind.bits[abs(lit)-1]) or
                          (lit < 0 and not ind.bits[abs(lit)-1])
                          for lit in instance.clauses[cidx])
            ind.sat_vec[cidx] = new_sat
            if new_sat and not old_sat:
                ind.fitness += 1
            elif not new_sat and old_sat:
                ind.fitness -= 1

    def _crossover_2pt(self, p1, p2, instance):
        n  = len(p1.bits)
        k1 = random.randint(0, n - 2)
        k2 = random.randint(k1 + 1, n - 1)
        b1 = p1.bits[:k1] + p2.bits[k1:k2] + p1.bits[k2:]
        b2 = p2.bits[:k1] + p1.bits[k1:k2] + p2.bits[k2:]
        sv1, f1 = self._full_eval(b1, instance)
        sv2, f2 = self._full_eval(b2, instance)
        return _Individual(b1, sv1, f1), _Individual(b2, sv2, f2)

    def _mutate_incremental(self, ind, pm, instance):
        ind = _Individual(ind.bits[:], ind.sat_vec[:], ind.fitness)
        for var_idx in range(len(ind.bits)):
            if random.random() < pm:
                self._incremental_update(ind, var_idx, instance)
        return ind

    def _semantic_distance(self, a, b):
        return sum(x != y for x, y in zip(a.sat_vec, b.sat_vec))

    def _tournament_select(self, pop):
        i1, i2 = random.randrange(len(pop)), random.randrange(len(pop))
        return (i1, pop[i1]) if pop[i1].fitness >= pop[i2].fitness else (i2, pop[i2])

    def _crowding_replace(self, pop, child, p1_idx, p2_idx):
        d1  = self._semantic_distance(child, pop[p1_idx])
        d2  = self._semantic_distance(child, pop[p2_idx])
        tgt = p1_idx if d1 <= d2 else p2_idx
        if child.fitness >= pop[tgt].fitness:
            pop[tgt] = child

    def solve(self, instance):
        random.seed(self.seed)
        n  = instance.n_vars
        pm = self.mutation_prob if self.mutation_prob else 1.0 / n

        pop = []
        for _ in range(self.pop_size):
            bits = [random.random() < 0.5 for _ in range(n)]
            sv, fit = self._full_eval(bits, instance)
            pop.append(_Individual(bits, sv, fit))

        best    = max(pop, key=lambda ind: ind.fitness)
        best_ind = _Individual(best.bits[:], best.sat_vec[:], best.fitness)
        start   = time.time()

        for _ in range(self.max_gen):
            if time.time() - start > self.timeout or best_ind.fitness == instance.n_clauses:
                break
            for _ in range(self.pop_size // 2):
                p1_idx, p1 = self._tournament_select(pop)
                p2_idx, p2 = self._tournament_select(pop)
                if random.random() < self.crossover_prob:
                    c1, c2 = self._crossover_2pt(p1, p2, instance)
                else:
                    c1 = _Individual(p1.bits[:], p1.sat_vec[:], p1.fitness)
                    c2 = _Individual(p2.bits[:], p2.sat_vec[:], p2.fitness)
                c1 = self._mutate_incremental(c1, pm, instance)
                c2 = self._mutate_incremental(c2, pm, instance)
                self._crowding_replace(pop, c1, p1_idx, p2_idx)
                self._crowding_replace(pop, c2, p2_idx, p1_idx)

            cur = max(pop, key=lambda ind: ind.fitness)
            if cur.fitness > best_ind.fitness:
                best_ind = _Individual(cur.bits[:], cur.sat_vec[:], cur.fitness)

        elapsed = time.time() - start
        return best_ind.bits, best_ind.fitness, best_ind.fitness / instance.n_clauses, elapsed


GA_CLASS_MAP.update({
    "AG Classique": ClassicGA,
    "AG Adaptatif": AdaptiveGA,
    "AG + KC": KCBasedGA,
})
