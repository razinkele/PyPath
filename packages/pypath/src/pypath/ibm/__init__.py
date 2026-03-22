"""
IBM (Individual-Based Model) module for PyPath.

Provides individual-based modeling capabilities that can be coupled with
the Ecosim population dynamics engine. IBM-managed functional groups use
super-individuals to represent cohorts of organisms, enabling detailed
bioenergetics, size-structured predation, movement, and reproduction.

Classes
-------
SuperIndividual
    Dataclass representing a cohort of identical organisms.
IBMStepResult
    Dataclass returned by each IBM integration step.
IBMGroup
    Abstract base class for all IBM group implementations.
BioenergParams
    Dataclass holding Wisconsin bioenergetics model parameters.
PredationParams
    Dataclass holding size-structured predation selectivity parameters.
MovementParams
    Dataclass holding spatial movement parameters.
ForagingParams
    Dataclass holding adaptive foraging parameters.
ReproductionParams
    Dataclass holding stochastic reproduction and larval survival parameters.
SmeltParams
    Composite parameter dataclass for Baltic smelt IBM.
SmeltIBM
    Concrete IBMGroup implementation for Baltic smelt.

Example
-------
>>> from pypath.ibm import SuperIndividual, IBMGroup, IBMStepResult
>>>
>>> # SuperIndividual represents a cohort of 1000 fish
>>> si = SuperIndividual(
...     id=0, n_represented=1000.0, weight=0.05, length=12.0,
...     age=2.0, energy_reserve=0.8, patch_idx=0,
...     is_mature=True, sex=1,
... )
>>> si.total_biomass_tonnes()
5e-05
"""

from pypath.ibm.base import IBMGroup, IBMStepResult, LifeStage, SpatialContext, SuperIndividual
from pypath.ibm.behavior import ForagingParams, MovementParams
from pypath.ibm.bioenergetics import BioenergParams
from pypath.ibm.development import (
    EggParams,
    LarvalParams,
    OxygenParams,
    YolkSacParams,
    ZoneParams,
)
from pypath.ibm.predation import PredationParams
from pypath.ibm.reproduction import ReproductionParams
from pypath.ibm.smelt import SmeltIBM, SmeltParams

__all__ = [
    "SuperIndividual",
    "IBMStepResult",
    "IBMGroup",
    "LifeStage",
    "SpatialContext",
    "BioenergParams",
    "ForagingParams",
    "MovementParams",
    "PredationParams",
    "ReproductionParams",
    "EggParams",
    "YolkSacParams",
    "LarvalParams",
    "OxygenParams",
    "ZoneParams",
    "SmeltParams",
    "SmeltIBM",
]

__version__ = "0.4.1"
