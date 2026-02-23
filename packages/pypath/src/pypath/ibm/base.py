"""
Base data structures for the IBM (Individual-Based Model) module.

Provides the foundational dataclasses and abstract base class used by all
IBM group implementations in PyPath. These structures define how individual
organisms (super-individuals) are represented and how IBM groups interface
with the Ecosim population dynamics engine.

Classes
-------
SuperIndividual
    Represents a cohort of biologically identical organisms.
IBMStepResult
    Return type for a single IBM integration step.
IBMGroup
    Abstract base class that all concrete IBM implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SuperIndividual:
    """A super-individual representing a cohort of identical organisms.

    Each super-individual tracks the state of *n_represented* biologically
    identical fish (or other organisms). This is the fundamental unit of
    the IBM module.

    Parameters
    ----------
    id : int
        Unique identifier for this super-individual.
    n_represented : float
        Number of real individuals this super-individual represents.
    weight : float
        Individual body weight (grams).
    length : float
        Individual body length (cm).
    age : float
        Age (years).
    energy_reserve : float
        Dimensionless energy reserve index (0-1 typical range).
    patch_idx : int
        Spatial patch index where this super-individual currently resides.
    is_mature : bool
        Whether this super-individual has reached sexual maturity.
    sex : int
        Sex code (0 = female, 1 = male, or other coding as needed).
    """

    id: int
    n_represented: float
    weight: float
    length: float
    age: float
    energy_reserve: float
    patch_idx: int
    is_mature: bool
    sex: int

    def total_biomass_tonnes(self) -> float:
        """Return total biomass represented by this super-individual in tonnes.

        Computed as ``n_represented * weight / 1e6``, assuming weight is in
        grams and 1 tonne = 1e6 grams.

        Returns
        -------
        float
            Total biomass in metric tonnes.
        """
        return self.n_represented * self.weight / 1e6


@dataclass
class IBMStepResult:
    """Result of a single IBM integration step.

    Returned by :meth:`IBMGroup.compute_step` to communicate the IBM
    state back to the Ecosim integration loop.

    Parameters
    ----------
    biomass : float
        Total group biomass (tonnes) after this step.
    production : float
        Net production during this step (tonnes).
    consumption_by_prey : np.ndarray
        1-D array of shape ``(n_groups,)`` giving the biomass consumed
        from each prey group during this step.
    mortality_count : float
        Number of individuals that died during this step.
    recruitment_count : float
        Number of new individuals recruited during this step.
    """

    biomass: float
    production: float
    consumption_by_prey: np.ndarray
    mortality_count: float
    recruitment_count: float
    patch_biomass: Optional[np.ndarray] = None


@dataclass
class SpatialContext:
    """Spatial data passed to IBM groups during Ecospace simulations.

    When an IBM group is part of a spatial simulation, this context provides
    the patch-level environmental information needed for movement decisions.

    Parameters
    ----------
    adjacency : Any
        Sparse adjacency matrix of shape ``(n_patches, n_patches)``.
        Typically ``scipy.sparse.csr_matrix``.
    habitat_quality : np.ndarray
        Per-patch habitat quality for this group, shape ``(n_patches,)``.
    food_density : np.ndarray
        Per-patch total prey biomass, shape ``(n_patches,)``.
    predator_density : np.ndarray
        Per-patch total predator biomass, shape ``(n_patches,)``.
    n_patches : int
        Number of spatial patches.
    """

    adjacency: Any  # scipy.sparse.csr_matrix (avoid hard import)
    habitat_quality: np.ndarray
    food_density: np.ndarray
    predator_density: np.ndarray
    n_patches: int


class IBMGroup(ABC):
    """Abstract base class for IBM group implementations.

    Every IBM-managed functional group must subclass ``IBMGroup`` and
    implement all four abstract methods. The Ecosim integration loop
    calls these methods to delegate dynamics to the IBM engine while
    keeping the rest of the food-web coupled.

    Parameters
    ----------
    group_index : int
        Zero-based index of this group in the Ecopath/Ecosim model.
    n_groups : int
        Total number of functional groups in the model (used to size
        consumption arrays).

    Attributes
    ----------
    group_index : int
        Index of this group.
    n_groups : int
        Total number of groups in the model.
    individuals : List[SuperIndividual]
        Population of super-individuals managed by this group.
    """

    def __init__(self, group_index: int, n_groups: int) -> None:
        self.group_index: int = group_index
        self.n_groups: int = n_groups
        self.individuals: List[SuperIndividual] = []

    @abstractmethod
    def compute_step(
        self,
        prey_available: np.ndarray,
        predation_pressure: float,
        env_forcing: Dict[str, Any],
        dt: float,
    ) -> IBMStepResult:
        """Advance the IBM population by one time step.

        Parameters
        ----------
        prey_available : np.ndarray
            1-D array of shape ``(n_groups,)`` giving the available biomass
            of each prey group.
        predation_pressure : float
            Total predation mortality pressure on this group from other
            predators in the Ecosim food web.
        env_forcing : Dict[str, Any]
            Dictionary of environmental forcing values (temperature, etc.).
        dt : float
            Time step size (years).

        Returns
        -------
        IBMStepResult
            Aggregated results of this time step.
        """

    @abstractmethod
    def get_aggregate_biomass(self) -> float:
        """Return the total biomass (tonnes) of all super-individuals.

        Returns
        -------
        float
            Sum of biomass across all super-individuals.
        """

    @abstractmethod
    def get_consumption_by_prey(self) -> np.ndarray:
        """Return the consumption vector by prey group.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_groups,)`` with biomass consumed from
            each prey group.
        """

    @abstractmethod
    def initialize_from_ecosim(
        self,
        biomass: float,
        params: Dict[str, Any],
        n_super_individuals: int = 500,
    ) -> None:
        """Initialize the IBM population from Ecosim equilibrium state.

        Parameters
        ----------
        biomass : float
            Initial total biomass (tonnes) from Ecopath.
        params : Dict[str, Any]
            Species-specific biological parameters.
        n_super_individuals : int, optional
            Number of super-individuals to create (default 500).
        """
