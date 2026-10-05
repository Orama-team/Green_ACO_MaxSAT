"""
greenaco
========

An energy-aware ACO solver for MAX-SAT.

The package is organised around four concerns:

``wcnf``     instance representation and WCNF parsing
``data``     benchmark manifest handling and instance loading
``solver``   the Green ACO search loop
``operators`` / ``scheduler`` / ``energy`` / ``profile`` / ``metrics``
             the operator pool, the EI/J scheduler, energy accounting,
             operator profiling, and the statistical tests
``comparison`` the GA and ACO variants used for the method comparison
"""

from .wcnf import Instance, parse_wcnf

__all__ = ["Instance", "parse_wcnf"]

__version__ = "1.0.0"