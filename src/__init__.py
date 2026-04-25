"""
Taxi Routing Optimization Package
"""
from .model.fixed_point import compute_joint_fixed_point_final
from .model.reward import compute_kappa, compute_R
from .optimization.optimizer import GradientOptimizer

__version__ = '0.1.0'