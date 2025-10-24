"""
UR5e MuJoCo Simulation Package

This package provides tools for simulating the UR5e robot in MuJoCo.

Main components:
- ur5e_simulator: Main simulator class for UR5e robot
- util: Utility functions for transformations and math
- PID: PID controller implementation
- ik_module: Inverse kinematics utilities
"""

from .ur5e_simulator import UR5eSimulator
from .util import *
from .PID import PID_ControllerClass

__version__ = "1.0.0"
__author__ = "UR5e Simulation Team"
