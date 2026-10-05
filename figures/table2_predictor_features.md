Table 2: Structural features used by the budget predictor. All are computed pre-solve in a single pass over the clause list; none uses solver output, quality, or the chosen budget.

| Feature | Description |
| --- | --- |
| nvars | Number of Boolean variables.
| nclauses | Number of clauses.
| clause/var ratio | Clause density m/n.
| mean clause len | Average literals per clause.
| std clause len | Spread of clause lengths.
| mean var degree | Average number of clauses a variable appears in.
| std var degree | Spread of variable occurrence counts.
| max var degree | Largest variable occurrence count.
| frac. unit clauses | Fraction of clauses with one literal.
| frac. binary clauses | Fraction of clauses with two literals.
| frac. Horn clauses | Fraction of clauses with ≤1 positive literal.
| positive literal frac. | Fraction of all literals that are positive.
