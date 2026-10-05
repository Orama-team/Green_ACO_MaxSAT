"""
comparison/aco.py
=================
Ant-colony variants used in the method comparison.

Ported from the earlier ACO codebase, so every method in the comparison table is
produced by this repository and measured under identical conditions.

All six share the signature ``fn(formula, n_vars) -> (assignment, stats)`` and
differ in how the pheromone is exploited and updated:

  aco_sat           Ant System: every ant deposits on its own solution
  aco_sat_elitist   plus extra reinforcement of the iteration's best solution
  mmas_sat          MAX-MIN Ant System: only the best ant updates, pheromone
                    clipped to [tau_min, tau_max], reset on stagnation
  acs_sat           Ant Colony System: pseudo-random proportional rule with
                    q0, online local update and a single global update
  nl_aco_sat        Negative Learning: paired positive/negative pheromone
  faco_sat          Focused construction over a restricted variable set

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

# Present in the source codebase but excluded from the comparison: they do not
# terminate within the time budget on any of the comparison instances, so they
# contribute no usable measurement.
EXCLUDED = ("NL-ACO", "FACO")




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


def nl_aco_sat(formula, n_vars,
               n_ants=20, n_iter=100,
               rho=0.1, rho_neg=0.3,
               drate=0.7,
               tau_min=0.001, tau_max=0.999):
    """
    Negative Learning ACO pour MaxSAT.
    """
    import time, random, numpy as np
    from collections import defaultdict

    start_time = time.time()
    n_clauses  = len(formula)

    # --- initialisation des deux modèles ---
    # T : phéromones positives initialisées à 0.5
    T     = [[0.5, 0.5] for _ in range(n_vars)]
    # T_neg : phéromones négatives initialisées à tau_min
    T_neg = [[tau_min, tau_min] for _ in range(n_vars)]

    # eta_i : nombre d'occurrences de chaque variable (pour sélection de variable)
    eta_var = [0.0] * n_vars
    for clause in formula:
        for lit in clause:
            eta_var[abs(lit) - 1] += 1.0

    # precompute variable -> clause index mapping (optimisation : évite scan complet)
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(formula):
        for lit in clause:
            var_to_clauses[abs(lit)].append(ci)

    # ordre déterministe et probs roulette précalculés une fois (statiques)
    sorted_vars    = sorted(range(n_vars), key=lambda i: eta_var[i], reverse=True)
    total_eta      = sum(eta_var) or 1.0
    roulette_probs = [eta_var[i] / total_eta for i in range(n_vars)]

    best_score      = -1
    best_assignment = None   # S_bsf
    visited_nodes   = 0
    improving = 0; deteriorating = 0; neutral = 0
    stagnation_lengths = []; stagnation_counter = 0
    total_possible = n_ants * n_iter * n_vars

    # --- facteur de convergence ---
    def convergence_factor(T):
        s = 0.0
        for i in range(n_vars):
            s += max(tau_max - T[i][0], T[i][0] - tau_min)
            s += max(tau_max - T[i][1], T[i][1] - tau_min)
        cf = 2.0 * (s / (2 * n_vars * (tau_max - tau_min)) - 0.5)
        return cf

    def clip(val):
        return min(tau_max, max(tau_min, val))

    # --- construction d'une solution ---
    def construct_solution(T, T_neg):
        assignment = {}

        # phase 1 : ordre de sélection des variables (déterministe ou roulette)
        if random.random() <= drate:
            order = sorted_vars  # ordre précalculé
        else:
            remaining_set = set(range(n_vars))
            order = []
            while remaining_set:
                cum = 0.0; rr = random.random()
                remaining_list = list(remaining_set)
                best_i = remaining_list[-1]
                for idx in remaining_list:
                    cum += roulette_probs[idx]
                    if rr <= cum:
                        best_i = idx
                        break
                order.append(best_i)
                remaining_set.remove(best_i)

        # phase 2 : sélection de valeur pour chaque variable dans l'ordre
        for i in order:
            var = i + 1
            scores = []
            for v in range(2):
                # heuristique : 1 / (1 + nouvelles violations si on assigne v)
                # optimisation : on ne scanne que les clauses impliquant var
                new_viol = 0
                tmp = dict(assignment)
                tmp[var] = bool(v)
                for ci in var_to_clauses[var]:
                    clause = formula[ci]
                    if all(abs(lit) in tmp for lit in clause):
                        if not any(
                            (lit > 0 and tmp[abs(lit)]) or
                            (lit < 0 and not tmp[abs(lit)])
                            for lit in clause
                        ):
                            new_viol += 1
                theta = 1.0 / (1.0 + new_viol)
                scores.append(theta * T[i][v] * (1.0 - T_neg[i][v]))

            total_s = sum(scores)
            prob_true = scores[1] / total_s if total_s > 0 else 0.5
            assignment[var] = bool(1 if random.random() < prob_true else 0)

        return assignment

    # --- solveur de sous-instance : flip greedy first-improvement ---
    def solve_subinstance(fixed_vars, assignment_template):
        """
        fixed_vars : dict {var: val} des variables consensuelles (fixées)
        Résout les variables restantes par flip greedy.
        """
        asgn = assignment_template.copy()
        # fixer les variables consensuelles
        for var, val in fixed_vars.items():
            asgn[var] = val
        score = satisfied_clauses(formula, asgn)
        # flip greedy sur les variables NON fixées
        free_vars = [i + 1 for i in range(n_vars) if (i + 1) not in fixed_vars]
        improved = True
        while improved:
            improved = False
            for var in free_vars:
                asgn[var] = not asgn[var]
                new_score = satisfied_clauses(formula, asgn)
                if new_score > score:
                    score = new_score; improved = True
                else:
                    asgn[var] = not asgn[var]
        return asgn, score

    # --- MAJ positive MMAS ---
    def positive_update(T, solution_ib, solution_rb, solution_bsf, cf, bs_update):
        # poids identifiés par l'article selon le facteur de convergence
        if not bs_update:
            if   cf < 0.4: k_ib, k_rb, k_bsf = 1.0, 0.0, 0.0
            elif cf < 0.6: k_ib, k_rb, k_bsf = 2/3, 1/3, 0.0
            elif cf < 0.8: k_ib, k_rb, k_bsf = 1/3, 2/3, 0.0
            else:          k_ib, k_rb, k_bsf = 0.0, 1.0, 0.0
        else:
            k_ib, k_rb, k_bsf = 0.0, 0.0, 1.0

        for i in range(n_vars):
            var = i + 1
            for v in range(2):
                deposit = 0.0
                for sol, k in [(solution_ib, k_ib), (solution_rb, k_rb), (solution_bsf, k_bsf)]:
                    if sol is not None and (1 if sol[var] else 0) == v:
                        deposit += k
                T[i][v] = clip(T[i][v] + rho * (deposit - T[i][v]))

    # --- MAJ négative ---
    def negative_update(T_neg, S_sub, fixed_vars):
        for i in range(n_vars):
            var = i + 1
            if var in fixed_vars:
                continue  # seulement X \ X'
            for v in range(2):
                # pénaliser la valeur OPPOSÉE à celle choisie par S_sub
                sub_val = 1 if S_sub[var] else 0
                xi_neg = 1.0 if sub_val != v else 0.0
                T_neg[i][v] = clip(T_neg[i][v] + rho_neg * (xi_neg - T_neg[i][v]))

    # === boucle principale ===
    bs_update  = False
    S_rb       = None   # restart best
    S_rb_score = -1

    for iteration in range(n_iter):
        S_iter               = []   # (assignment, score)
        iter_best_score      = -1
        iter_best_assignment = None

        for ant in range(n_ants):
            visited_nodes += n_vars
            asgn  = construct_solution(T, T_neg)
            score = satisfied_clauses(formula, asgn)
            S_iter.append((asgn, score))

            if score > iter_best_score:
                iter_best_score      = score
                iter_best_assignment = asgn.copy()

            if score > best_score:
                best_score      = score
                best_assignment = asgn.copy()
                improving += 1
                stagnation_lengths.append(stagnation_counter)
                stagnation_counter = 0
            elif score == best_score:
                neutral += 1
                stagnation_counter += 1
            else:
                deteriorating += 1
                stagnation_counter += 1

        # --- construire la sous-instance X' ---
        # variables où toutes les fourmis sont d'accord
        fixed_vars = {}
        for i in range(n_vars):
            var  = i + 1
            vals = [asgn[var] for asgn, _ in S_iter]
            if all(v == vals[0] for v in vals):
                fixed_vars[var] = vals[0]

        # --- résoudre la sous-instance ---
        # utiliser la meilleure solution de l'itération comme point de départ
        S_sub, S_sub_score = solve_subinstance(fixed_vars, iter_best_assignment)

        # S_sub peut améliorer S_ib
        if S_sub_score > iter_best_score:
            iter_best_score      = S_sub_score
            iter_best_assignment = S_sub.copy()

        if iter_best_score > best_score:
            best_score      = iter_best_score
            best_assignment = iter_best_assignment.copy()

        if S_rb is None or iter_best_score > S_rb_score:
            S_rb       = iter_best_assignment.copy()
            S_rb_score = iter_best_score

        # --- MAJ positive ---
        positive_update(T, iter_best_assignment, S_rb, best_assignment,
                        convergence_factor(T), bs_update)

        # --- MAJ négative ---
        negative_update(T_neg, S_sub, fixed_vars)

        # --- facteur de convergence et restart ---
        cf = convergence_factor(T)
        if cf > 0.999:
            if bs_update:
                S_rb = None; S_rb_score = -1; bs_update = False
                T     = [[0.5,     0.5    ] for _ in range(n_vars)]
                T_neg = [[tau_min, tau_min] for _ in range(n_vars)]
            else:
                bs_update = True

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


def faco_sat(formula, n_vars,
             n_ants=20, n_iter=100,
             alpha=1.0, beta=2.0,
             rho=0.1, Q=1.0,
             tau_min=0.01, tau_max=1.0,
             min_new=8, p_gb=0.05):
    """
    Focused ACO (FACO) pour MaxSAT.
    Référence : Skinderowicz (2022)
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

    best_score           = -1
    best_assignment      = None
    iter_best_assignment = None
    visited_nodes        = 0
    improving = 0; deteriorating = 0; neutral = 0
    stagnation_lengths = []; stagnation_counter = 0
    total_possible = n_ants * n_iter * n_vars

    for iteration in range(n_iter):

        if best_assignment is None:
            source = None
        elif iter_best_assignment is None or random.random() < p_gb:
            source = best_assignment
        else:
            source = iter_best_assignment

        iter_best_score      = -1
        iter_best_assignment = None

        for ant in range(n_ants):
            visited_nodes += n_vars
            assignment = {}
            new_count  = 0

            for i in range(n_vars):
                var = i + 1

                if source is not None and new_count >= min_new:
                    assignment[var] = source[var]
                    continue

                scores = [
                    (pheromone[i][v] ** alpha) * ((eta[i][v] + 1e-6) ** beta)
                    for v in range(2)
                ]
                total = sum(scores)
                prob_true = scores[1] / total if total > 0 else 0.5
                chosen = random.random() < prob_true
                assignment[var] = chosen

                if source is not None and chosen != source[var]:
                    new_count += 1

            score = satisfied_clauses(formula, assignment)

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

        if iter_best_assignment is not None:
            delta = Q * iter_best_score / n_clauses
            for i in range(n_vars):
                var = i + 1
                v   = 1 if iter_best_assignment[var] else 0
                pheromone[i][v] += delta

        for i in range(n_vars):
            pheromone[i][0] = min(tau_max, max(tau_min, pheromone[i][0]))
            pheromone[i][1] = min(tau_max, max(tau_min, pheromone[i][1]))

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
