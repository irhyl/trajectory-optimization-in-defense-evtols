"""
NSGA-III Optimizer - Top-level module alias.

Re-exports NSGA3Optimizer from planning.optimization.nsga3 so that external
code can do: ``from evtol.planning.nsga3_optimizer import NSGA3Optimizer``
without knowing the internal subpackage layout.
"""
from .optimization.nsga3 import NSGA3Optimizer

__all__ = ["NSGA3Optimizer"]
