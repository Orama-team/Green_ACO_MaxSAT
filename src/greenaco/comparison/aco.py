"""
comparison/aco.py
=================
Ant-colony variants used in the method comparison.

Ported from the earlier ACO codebase, so every method in the comparison table is
produced by this repository and measured under identical conditions.

All four share the signature ``fn(formula, n_vars) -> (assignment, stats)`` and
differ in how the pheromone is exploited and updated:

  aco_sat           Ant System: every ant deposits on its own solution
  aco_sat_elitist   plus extra reinforcement of the iteration's best solution
  mmas_sat          MAX-MIN Ant System: only the best ant updates, pheromone
                    clipped to [tau_min, tau_max], reset on stagnation
  acs_sat           Ant Colony System: pseudo-random proportional rule with
                    q0, online local update and a single global update

``TUNED_PARAMS`` holds the calibrated settings used for the comparison.
"""

from __future__ import annotations

from typing import Dict

# Calibrated settings for the comparison, shared across instances.
# Source: data/comparison/method_params.yaml
ACO_SHARED = {
    "n_ants": 10,
    "n_iter": 50,
    "alpha": 2.0,
    "beta": 3.0,
    "rho": 0.2,
}

TUNED_PARAMS: Dict[str, Dict] = {
    "AS-SAT": {"tau0": 1.0, "Q": 1.0},
    "AS-SAT Elitiste": {"tau0": 0.5, "e": 5.0},
    "MMAS": {"tau_min": 0.05, "tau_max": 1.0, "stagnation_limit": 10},
    "ACS": {"tau0": 1.0, "q0": 0.95, "xi": 0.1, "cl": 5.0},
}


def satisfied_clauses(formula, assignment):
    count = 0
    for clause in formula:
        for lit in clause:
            var = abs(lit)
            if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                count += 1
                break
    return count


def aco_sat(formula, n_vars,
            n_ants=20, n_iter=100,
            alpha=1.0, beta=2.0,
            rho=0.1, Q=1.0,
            tau0=0.5):
    """
    AS-SAT: plain Ant System (Dorigo et al.) adapté à MaxSAT.
    """
    import time, random, numpy as np

    start_time = time.time()
    n_clauses  = len(formula)

    pheromone = [[tau0, tau0] for _ in range(n_vars)]

    # Precompute heuristic eta[i][v]: number of clauses satisfied
    # by setting variable (i+1) to bool(v), static across all ants/iterations.
    eta = [[0.0, 0.0] for _ in range(n_vars)]
    for i in range(n_vars):
        var = i + 1
        for clause in formula:
            for lit in clause:
                if abs(lit) == var:
                    for v in range(2):
                        val = (v == 1)
                        if (lit > 0 and val) or (lit < 0 and not val):
                            eta[i][v] += 1.0
                    break

    best_score      = -1
    best_assignment = None
    visited_nodes   = 0
    improving = 0; deteriorating = 0; neutral = 0
    stagnation_lengths = []; stagnation_counter = 0
    total_possible = n_ants * n_iter * n_vars  # each ant visits n_vars nodes

    for iteration in range(n_iter):
        solutions = []

        for ant in range(n_ants):
            visited_nodes += n_vars  # ant traverses one node per variable
            assignment = {}

            for i in range(n_vars):
                var = i + 1
                scores = [
                    (pheromone[i][v] ** alpha) * ((eta[i][v] + 1e-6) ** beta)
                    for v in range(2)
                ]
                total = sum(scores)
                prob_true = scores[1] / total if total > 0 else 0.5
                assignment[var] = random.random() < prob_true

            score = satisfied_clauses(formula, assignment)
            solutions.append((assignment, score))

            if score > best_score:
                best_score = score
                best_assignment = assignment.copy()
                improving += 1
                stagnation_lengths.append(stagnation_counter)
                stagnation_counter = 0
            elif score == best_score:
                neutral += 1
                stagnation_counter += 1
            else:
                deteriorating += 1
                stagnation_counter += 1

        # AS update: evaporate then deposit from all ants (formula 3)
        for i in range(n_vars):
            pheromone[i][0] *= (1 - rho)
            pheromone[i][1] *= (1 - rho)

        for assignment, score in solutions:
            delta = Q * score / n_clauses
            for i in range(n_vars):
                var = i + 1
                v = 1 if assignment[var] else 0
                pheromone[i][v] += delta

        if best_score == n_clauses:
            break

    exec_time = time.time() - start_time
    stagnation_lengths.append(stagnation_counter)

    stats = {
        "qualite_solution":             best_score,
        "temps_exec":                   exec_time,
        "noeuds_explores":              visited_nodes,
        "noeuds_elagues":               0,
        "couverture_noeuds":            100.0 * visited_nodes / total_possible,
        "nbr_mouvements_ameliorants":   improving,
        "nbr_mouvements_deteriorants":  deteriorating,
        "nbr_mouvements_neutres":       neutral,
        "stagnation_moyenne":           np.mean(stagnation_lengths) if stagnation_lengths else 0,
    }
    return best_assignment, stats


def aco_sat_elitist(formula, n_vars,
                    n_ants=20, n_iter=100,
                    alpha=1.0, beta=2.0,
                    rho=0.1, Q=1.0,
                    tau0=0.5, e=3):
    """
    ACO-SAT Elitiste (Dorigo et al. 1996) adapté à MaxSAT.
    """
    import time, random, numpy as np

    start_time = time.time()
    n_clauses  = len(formula)

    pheromone = [[tau0, tau0] for _ in range(n_vars)]

    # Precompute eta once
    eta = [[0.0, 0.0] for _ in range(n_vars)]
    for i in range(n_vars):
        var = i + 1
        for clause in formula:
            for lit in clause:
                if abs(lit) == var:
                    for v in range(2):
                        val = (v == 1)
                        if (lit > 0 and val) or (lit < 0 and not val):
                            eta[i][v] += 1.0
                    break

    best_score      = -1
    best_assignment = None
    visited_nodes   = 0
    improving = 0; deteriorating = 0; neutral = 0
    stagnation_lengths = []; stagnation_counter = 0
    total_possible = n_ants * n_iter * n_vars

    for iteration in range(n_iter):
        solutions = []
        iter_best_score      = -1
        iter_best_assignment = None

        for ant in range(n_ants):
            visited_nodes += n_vars
            assignment = {}

            for i in range(n_vars):
                var = i + 1
                scores = [
                    (pheromone[i][v] ** alpha) * ((eta[i][v] + 1e-6) ** beta)
                    for v in range(2)
                ]
                total = sum(scores)
                prob_true = scores[1] / total if total > 0 else 0.5
                assignment[var] = random.random() < prob_true

            score = satisfied_clauses(formula, assignment)
            solutions.append((assignment, score))

            if score > iter_best_score:
                iter_best_score      = score
                iter_best_assignment = assignment.copy()

            if score > best_score:
                best_score      = score
                best_assignment = assignment.copy()
                improving += 1
                stagnation_lengths.append(stagnation_counter)
                stagnation_counter = 0
            elif score == best_score:
                neutral += 1
                stagnation_counter += 1
            else:
                deteriorating += 1
                stagnation_counter += 1

        for i in range(n_vars):
            pheromone[i][0] *= (1 - rho)
            pheromone[i][1] *= (1 - rho)

        for assignment, score in solutions:
            delta = Q * score / n_clauses
            for i in range(n_vars):
                var = i + 1
                v = 1 if assignment[var] else 0
                pheromone[i][v] += delta

        if iter_best_assignment is not None:
            delta_elite = e * Q * iter_best_score / n_clauses
            for i in range(n_vars):
                var = i + 1
                v_best = 1 if iter_best_assignment[var] else 0
                pheromone[i][v_best] += delta_elite

        if best_score == n_clauses:
            break

    exec_time = time.time() - start_time
    stagnation_lengths.append(stagnation_counter)

    stats = {
        "qualite_solution":             best_score,
        "temps_exec":                   exec_time,
        "noeuds_explores":              visited_nodes,
        "noeuds_elagues":               0,
        "couverture_noeuds":            100.0 * visited_nodes / total_possible,
        "nbr_mouvements_ameliorants":   improving,
        "nbr_mouvements_deteriorants":  deteriorating,
        "nbr_mouvements_neutres":       neutral,
        "stagnation_moyenne":           np.mean(stagnation_lengths) if stagnation_lengths else 0,
    }
    return best_assignment, stats


def mmas_sat(formula, n_vars,
             n_ants=20, n_iter=100,
             alpha=1.0, beta=2.0,
             rho=0.1, Q=1.0,
             tau_min=0.01, tau_max=1.0,
             stagnation_limit=20,
             use_global_best=True):
    """
    MAX-MIN Ant System (Stützle & Hoos 2000) adapté à MaxSAT.
    """
    import time, random, numpy as np

    start_time = time.time()
    n_clauses  = len(formula)

    pheromone = [[tau_max, tau_max] for _ in range(n_vars)]

    # Precompute eta once
    eta = [[0.0, 0.0] for _ in range(n_vars)]
    for i in range(n_vars):
        var = i + 1
        for clause in formula:
            for lit in clause:
                if abs(lit) == var:
                    for v in range(2):
                        val = (v == 1)
                        if (lit > 0 and val) or (lit < 0 and not val):
                            eta[i][v] += 1.0
                    break

    best_score      = -1
    best_assignment = None
    visited_nodes   = 0
    improving = 0; deteriorating = 0; neutral = 0
    stagnation_lengths = []; stagnation_counter = 0
    total_possible = n_ants * n_iter * n_vars

    for iteration in range(n_iter):
        solutions = []
        iter_best_score      = -1
        iter_best_assignment = None

        for ant in range(n_ants):
            visited_nodes += n_vars
            assignment = {}

            for i in range(n_vars):
                var = i + 1
                scores = [
                    (pheromone[i][v] ** alpha) * ((eta[i][v] + 1e-6) ** beta)
                    for v in range(2)
                ]
                total = sum(scores)
                prob_true = scores[1] / total if total > 0 else 0.5
                assignment[var] = random.random() < prob_true

            score = satisfied_clauses(formula, assignment)
            solutions.append((assignment, score))

            if score > iter_best_score:
                iter_best_score      = score
                iter_best_assignment = assignment.copy()

            if score > best_score:
                best_score      = score
                best_assignment = assignment.copy()
                improving += 1
                stagnation_lengths.append(stagnation_counter)
                stagnation_counter = 0
            elif score == best_score:
                neutral += 1
                stagnation_counter += 1
            else:
                deteriorating += 1
                stagnation_counter += 1

        # fallback: always use iter_best if global_best not yet set
        if use_global_best and best_assignment is not None:
            best_a, best_s = best_assignment, best_score
        else:
            best_a, best_s = iter_best_assignment, iter_best_score

        for i in range(n_vars):
            pheromone[i][0] *= (1 - rho)
            pheromone[i][1] *= (1 - rho)

        delta_base = Q * best_s / n_clauses
        for i in range(n_vars):
            var = i + 1
            v = 1 if best_a[var] else 0
            pheromone[i][v] += delta_base * (tau_max - pheromone[i][v])

        for i in range(n_vars):
            pheromone[i][0] = min(tau_max, max(tau_min, pheromone[i][0]))
            pheromone[i][1] = min(tau_max, max(tau_min, pheromone[i][1]))

        if stagnation_counter >= stagnation_limit:
            pheromone = [[tau_max, tau_max] for _ in range(n_vars)]
            stagnation_counter = 0

        if best_score == n_clauses:
            break

    exec_time = time.time() - start_time
    stagnation_lengths.append(stagnation_counter)

    stats = {
        "qualite_solution":             best_score,
        "temps_exec":                   exec_time,
        "noeuds_explores":              visited_nodes,
        "noeuds_elagues":               0,
        "couverture_noeuds":            100.0 * visited_nodes / total_possible,
        "nbr_mouvements_ameliorants":   improving,
        "nbr_mouvements_deteriorants":  deteriorating,
        "nbr_mouvements_neutres":       neutral,
        "stagnation_moyenne":           np.mean(stagnation_lengths) if stagnation_lengths else 0,
    }
    return best_assignment, stats


def acs_sat(formula, n_vars,
            n_ants=20, n_iter=100,
            alpha=1.0, beta=2.0,
            rho=0.1, Q=1.0,
            q0=0.9, xi=0.1, tau0=0.5,
            cl=5):
    """
    Ant Colony System (Dorigo & Gambardella 1997) adapté à MaxSAT.
    """
    import time, random, numpy as np

    start_time = time.time()
    n_clauses  = len(formula)

    pheromone = [[tau0, tau0] for _ in range(n_vars)]

    # Precompute eta once
    eta = [[0.0, 0.0] for _ in range(n_vars)]
    for i in range(n_vars):
        var = i + 1
        for clause in formula:
            for lit in clause:
                if abs(lit) == var:
                    for v in range(2):
                        val = (v == 1)
                        if (lit > 0 and val) or (lit < 0 and not val):
                            eta[i][v] += 1.0
                    break

    best_score      = -1
    best_assignment = None
    visited_nodes   = 0
    improving = 0; deteriorating = 0; neutral = 0
    stagnation_lengths = []; stagnation_counter = 0
    total_possible = n_ants * n_iter * n_vars

    for iteration in range(n_iter):
        solutions = []
        iter_best_score      = -1
        iter_best_assignment = None

        for ant in range(n_ants):
            visited_nodes += n_vars
            assignment = {}
            unassigned = list(range(n_vars))
            random.shuffle(unassigned)

            for idx in range(n_vars):
                i = unassigned[idx]
                var = i + 1

                q = random.random()
                if q <= q0:
                    scores = [
                        (pheromone[i][v] ** alpha) * ((eta[i][v] + 1e-6) ** beta)
                        for v in range(2)
                    ]
                    chosen = int(scores[1] > scores[0])
                else:
                    scores = [
                        (pheromone[i][v] ** alpha) * ((eta[i][v] + 1e-6) ** beta)
                        for v in range(2)
                    ]
                    total = sum(scores)
                    prob_true = scores[1] / total if total > 0 else 0.5
                    chosen = 1 if random.random() < prob_true else 0

                assignment[var] = bool(chosen)

                # local update
                pheromone[i][chosen] = (1 - xi) * pheromone[i][chosen] + xi * tau0

            score = satisfied_clauses(formula, assignment)
            solutions.append((assignment, score))

            if score > iter_best_score:
                iter_best_score      = score
                iter_best_assignment = assignment.copy()

            if score > best_score:
                best_score      = score
                best_assignment = assignment.copy()
                improving += 1
                stagnation_lengths.append(stagnation_counter)
                stagnation_counter = 0
            elif score == best_score:
                neutral += 1
                stagnation_counter += 1
            else:
                deteriorating += 1
                stagnation_counter += 1

        # global update: iter best only
        if iter_best_assignment is not None:
            for i in range(n_vars):
                pheromone[i][0] *= (1 - rho)
                pheromone[i][1] *= (1 - rho)

            delta = Q * iter_best_score / n_clauses
            for i in range(n_vars):
                var = i + 1
                v = 1 if iter_best_assignment[var] else 0
                pheromone[i][v] += rho * delta

        if best_score == n_clauses:
            break

    exec_time = time.time() - start_time
    stagnation_lengths.append(stagnation_counter)

    stats = {
        "qualite_solution":             best_score,
        "temps_exec":                   exec_time,
        "noeuds_explores":              visited_nodes,
        "noeuds_elagues":               0,
        "couverture_noeuds":            100.0 * visited_nodes / total_possible,
        "nbr_mouvements_ameliorants":   improving,
        "nbr_mouvements_deteriorants":  deteriorating,
        "nbr_mouvements_neutres":       neutral,
        "stagnation_moyenne":           np.mean(stagnation_lengths) if stagnation_lengths else 0,
    }
    return best_assignment, stats
