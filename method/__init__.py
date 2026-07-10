"""
Green ACO for MAX-SAT — energy-aware Ant Colony Optimisation with an
Expected-Improvement-per-Joule (EI/J) operator scheduler.

See README.md at the repository root for a full description, and
`TP5_presentation_finale.ipynb`-style analysis of the results this
package produces.
"""

from .config import GreenACOConfig
from .solver import GreenACOSolver, SolveResult

__all__ = ["GreenACOConfig", "GreenACOSolver", "SolveResult"]
