"""
SmeltIBM concrete implementation for Baltic smelt (Osmerus eperlanus).

Orchestrates all IBM behavior modules -- bioenergetics, predation, foraging,
reproduction, and growth -- into a single cohesive IBM group that can be
injected into the Ecosim derivative loop.

The SmeltIBM is initialized from Ecopath equilibrium biomass and creates an
age-structured population of super-individuals using Von Bertalanffy growth
curves.  Each time step, it runs up to five phases:

1. **Forage + Grow**: adaptive foraging followed by Wisconsin bioenergetics.
2. **Reproduce**: mature females spawn; surviving larvae become recruits.
3. **Predation mortality**: size-structured mortality from Ecosim predators.
4. **Bookkeeping**: add recruits, age individuals, remove senescent fish.
5. **Spatial movement** (optional): move individuals between patches.

Classes
-------
SmeltParams
    Composite parameter dataclass combining all IBM sub-module parameters.
SmeltIBM
    Concrete IBMGroup implementation for Baltic smelt.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext, SuperIndividual
from pypath.ibm.behavior import (
    ForagingParams,
    MovementParams,
    adaptive_forage,
    calculate_movement_probabilities,
    should_migrate,
)
from pypath.ibm.bioenergetics import (
    BioenergParams,
    growth_step_batch,
    growth_step_batch_ontogenetic,
    oxygen_scalar,
    thornton_lessem,
)
from pypath.ibm.development import (
    EggParams,
    LarvalParams,
    OxygenParams,
    YolkSacParams,
    ZoneParams,
    accumulate_degree_days,
    apply_egg_mortality,
    check_first_feeding,
    check_hatching,
    compute_yolk_depletion,
)
from pypath.ibm.predation import PredationParams, apply_predation_mortality
from pypath.ibm.reproduction import (
    ReproductionParams,
    create_recruits,
    spawn,
)

logger = logging.getLogger(__name__)

# Default dissolved-oxygen concentration (mg/L) used when env_forcing has no
# 'dissolved_oxygen' or 'o2' key.  99.0 represents a normoxic "no limitation"
# sentinel so oxygen never artificially caps consumption or survival.
_DEFAULT_O2 = 99.0


@dataclass
class SmeltParams:
    """Composite parameters for the SmeltIBM.

    Combines all sub-module parameter sets plus species-specific Von
    Bertalanffy growth parameters for Baltic smelt.

    Parameters
    ----------
    bioenerg : BioenergParams
        Wisconsin bioenergetics model parameters.
    predation : PredationParams
        Size-structured predation selectivity parameters.
    foraging : ForagingParams
        Adaptive prey-selection parameters.
    movement : MovementParams
        Spatial movement parameters.
    reproduction : ReproductionParams
        Spawning and larval survival parameters.
    vbgf_k_mean : float
        Mean Von Bertalanffy growth coefficient K (yr^-1).
    vbgf_k_sd : float
        Standard deviation of K across individuals.
    vbgf_linf_mean : float
        Mean asymptotic body length Linf (cm).
    vbgf_linf_sd : float
        Standard deviation of Linf across individuals.
    max_age : float
        Maximum age (years) before natural senescence removal.
    """

    bioenerg: BioenergParams
    predation: PredationParams
    foraging: ForagingParams
    movement: MovementParams
    reproduction: ReproductionParams
    vbgf_k_mean: float = 0.3
    vbgf_k_sd: float = 0.05
    vbgf_linf_mean: float = 25.0
    vbgf_linf_sd: float = 3.0
    max_age: float = 10.0
    max_super_individuals: int = 2000
    egg: Optional[EggParams] = None
    yolk_sac: Optional[YolkSacParams] = None
    larval: Optional[LarvalParams] = None
    oxygen: Optional[OxygenParams] = None
    zones: Optional[ZoneParams] = None

    @classmethod
    def baltic_defaults(cls) -> "SmeltParams":
        """Return default parameters for Baltic smelt (Osmerus eperlanus).

        These are literature-based defaults suitable for the Baltic Sea
        ecosystem.  All sub-parameter sets are populated with species-
        specific values.

        Returns
        -------
        SmeltParams
            Fully populated parameter object.
        """
        # Number of functional groups for foraging arrays (sized generously;
        # actual model may be smaller, but arrays are indexed by group index).
        n_groups_default = 20

        bioenerg = BioenergParams(
            ra=0.0033,  # g O2/g/day at reference temp
            rb=-0.227,  # metabolic weight exponent
            q10=2.1,  # Q10 temperature coefficient
            t_ref=10.0,  # reference temperature (C)
            sda_fraction=0.172,  # specific dynamic action
            unassimilated_fraction=0.27,  # unassimilated fraction
            a_length=0.55,  # allometric length coefficient
            b_length=0.333,  # allometric length exponent (cube root)
            energy_density=5.0,  # kJ/g tissue
            reproduction_fraction=0.3,
        )

        predation = PredationParams(
            optimal_prey_length=10.0,  # cm
            selectivity_sd=0.5,
        )

        foraging = ForagingParams(
            energy_content=np.full(n_groups_default, 4.0),  # kJ/g
            handling_time=np.full(n_groups_default, 1.0),  # time/g
        )

        movement = MovementParams(
            base_speed=0.3,
            habitat_weight=0.4,
            food_weight=0.4,
            predator_weight=0.2,
            migration_temp_threshold=4.0,  # C
            migration_months=(3, 4, 5),
        )

        reproduction = ReproductionParams(
            fecundity_coefficient=200.0,  # eggs per g^exponent
            fecundity_exponent=1.2,
            larval_base_survival=0.01,
            zooplankton_match_window=15.0,  # days
            maturity_energy_threshold=0.5,
            spawning_temp_threshold=4.0,  # C
            larval_duration_days=30,
            recruit_weight=0.5,  # g
            recruit_length=3.0,  # cm
        )

        return cls(
            bioenerg=bioenerg,
            predation=predation,
            foraging=foraging,
            movement=movement,
            reproduction=reproduction,
            vbgf_k_mean=0.3,
            vbgf_k_sd=0.05,
            vbgf_linf_mean=25.0,
            vbgf_linf_sd=3.0,
            max_age=10.0,
        )

    @classmethod
    def baltic_defaults_els(cls) -> "SmeltParams":
        """Return Baltic smelt defaults with early life stages enabled.

        Creates a standard ``baltic_defaults()`` parameter set and adds
        default EggParams, YolkSacParams, LarvalParams, and OxygenParams
        for mechanistic egg-to-larva modeling.

        Returns
        -------
        SmeltParams
            Fully populated parameter object with ELS enabled.
        """
        base = cls.baltic_defaults()
        base.egg = EggParams()
        base.yolk_sac = YolkSacParams()
        base.larval = LarvalParams()
        base.oxygen = OxygenParams()
        return base

    @classmethod
    def baltic_defaults_zonal(cls) -> "SmeltParams":
        """Return Baltic smelt defaults with ELS and zonal model enabled.

        Creates a ``baltic_defaults_els()`` parameter set and adds default
        ``ZoneParams`` for the 3-zone Curonian Lagoon spatial model.

        Returns
        -------
        SmeltParams
            Fully populated parameter object with ELS + zones enabled.
        """
        base = cls.baltic_defaults_els()
        base.zones = ZoneParams()
        return base


class SmeltIBM(IBMGroup):
    """Concrete IBM group implementation for Baltic smelt.

    Orchestrates bioenergetics, predation, foraging, and reproduction
    modules for an age-structured population of super-individuals
    representing Osmerus eperlanus.

    Parameters
    ----------
    group_index : int
        One-based index of this group in the Ecopath/Ecosim model
        (0 is reserved for the "Outside" placeholder).
    n_groups : int
        Total number of functional groups in the model.
    params : SmeltParams
        Species-specific parameters for all IBM sub-modules.
    """

    def __init__(
        self,
        group_index: int,
        n_groups: int,
        params: SmeltParams,
    ) -> None:
        super().__init__(group_index, n_groups)
        self.params = params
        self._last_consumption: np.ndarray = np.zeros(n_groups)
        self._next_id: int = 0
        self._rng: np.random.Generator = np.random.default_rng(42)
        self._o2_warning_issued: bool = False
        self._temp_warning_issued: bool = False

    def initialize_from_ecosim(
        self,
        biomass: float,
        params: Dict[str, Any],
        n_super_individuals: int = 500,
    ) -> None:
        """Initialize an age-structured population from Ecosim equilibrium.

        Creates *n_super_individuals* super-individuals with ages
        distributed from 0.5 to max_age.  Lengths are computed from the
        Von Bertalanffy growth function, weights from inverse allometry,
        and ``n_represented`` from an exponential survival curve scaled
        so that total biomass matches the input.

        Parameters
        ----------
        biomass : float
            Initial total biomass (tonnes) from Ecopath.
        params : Dict[str, Any]
            Additional species-specific parameters (currently unused;
            all parameters come from ``self.params``).
        n_super_individuals : int, optional
            Number of super-individuals to create (default 500).
        """
        sp = self.params

        # Distribute ages uniformly from 0.5 to max_age
        ages = np.linspace(0.5, sp.max_age, n_super_individuals)

        # Draw individual VBGF parameters with slight variation
        k_vals = self._rng.normal(sp.vbgf_k_mean, sp.vbgf_k_sd, n_super_individuals)
        k_vals = np.clip(k_vals, 0.05, None)  # K must be positive

        linf_vals = self._rng.normal(
            sp.vbgf_linf_mean, sp.vbgf_linf_sd, n_super_individuals
        )
        linf_vals = np.clip(linf_vals, 5.0, None)  # Linf must be positive

        # Compute lengths from Von Bertalanffy: L = Linf * (1 - exp(-K * age))
        lengths = linf_vals * (1.0 - np.exp(-k_vals * ages))
        lengths = np.clip(lengths, 0.1, None)

        # Compute weights from inverse allometry: weight = (length / a)^(1/b)
        a = sp.bioenerg.a_length
        b = sp.bioenerg.b_length
        weights = (lengths / a) ** (1.0 / b)
        weights = np.clip(weights, 0.1, None)

        # Exponential survival curve: relative abundance decreases with age
        # Use a natural mortality rate of ~0.5/yr as a reasonable default
        natural_mortality = 0.5
        survival_weights = np.exp(-natural_mortality * ages)

        # Compute n_represented to match target biomass.
        # Each individual contributes: n_represented_i * weight_i / 1e6 tonnes.
        # We want sum_i(n_represented_i * weight_i / 1e6) = biomass.
        # Set n_represented_i proportional to survival_weights and solve for
        # the scaling constant.
        # total_biomass = scale * sum(survival_weights * weights) / 1e6
        raw_biomass_contribution = survival_weights * weights
        total_raw = np.sum(raw_biomass_contribution)

        if total_raw <= 0.0:
            logger.warning(
                "Cannot initialize SmeltIBM: total raw biomass contribution is zero."
            )
            return

        # scale factor so that sum(scale * survival_weights * weight / 1e6) = biomass
        scale = biomass * 1e6 / total_raw
        n_represented = survival_weights * scale

        # Maturity: assume fish mature at age >= 2 years
        maturity_age = 2.0

        sexes = self._rng.integers(0, 2, size=n_super_individuals)

        individuals: List[SuperIndividual] = []
        for i in range(n_super_individuals):
            ind = SuperIndividual(
                id=i,
                n_represented=float(n_represented[i]),
                weight=float(weights[i]),
                length=float(lengths[i]),
                age=float(ages[i]),
                energy_reserve=float(weights[i]) * 0.1,  # initial reserve
                patch_idx=0,
                is_mature=bool(ages[i] >= maturity_age),
                sex=int(sexes[i]),
            )
            individuals.append(ind)

        self.individuals = individuals
        self._next_id = n_super_individuals
        self._last_consumption = np.zeros(self.n_groups)

    def get_aggregate_biomass(self) -> float:
        """Return total biomass (tonnes) across all super-individuals.

        Returns
        -------
        float
            Sum of ``total_biomass_tonnes()`` for each individual.
        """
        return sum(ind.total_biomass_tonnes() for ind in self.individuals)

    def get_consumption_by_prey(self) -> np.ndarray:
        """Return the consumption vector from the last time step.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_groups,)`` with biomass consumed
            from each prey group.
        """
        return self._last_consumption.copy()

    def _aggregate_by_patch(self, n_patches: int) -> np.ndarray:
        """Aggregate individual biomass by spatial patch.

        Parameters
        ----------
        n_patches : int
            Number of spatial patches.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_patches,)`` with total biomass per patch.
        """
        if not self.individuals:
            return np.zeros(n_patches)
        patches = np.array([ind.patch_idx for ind in self.individuals])
        biomasses = np.array([ind.total_biomass_tonnes() for ind in self.individuals])
        valid = (patches >= 0) & (patches < n_patches)
        result = np.zeros(n_patches)
        np.add.at(result, patches[valid], biomasses[valid])
        return result

    def _consolidate_population(self) -> None:
        """Merge similar super-individuals when population exceeds the cap.

        Groups individuals by ``(life_stage, patch_idx)`` and merges the
        two most similar (by weight) within each group until the total count
        is at or below ``self.params.max_super_individuals``.  Biomass is
        conserved exactly: the merged individual inherits the weighted-mean
        weight and recomputes length from allometry.
        """
        cap = self.params.max_super_individuals
        if len(self.individuals) <= cap:
            return

        sp = self.params

        # Group by (life_stage, patch_idx)
        groups: Dict[tuple, List[SuperIndividual]] = defaultdict(list)
        for ind in self.individuals:
            groups[(ind.life_stage, ind.patch_idx, ind.sex)].append(ind)

        # Within each group, sort by weight and merge closest pairs
        while sum(len(v) for v in groups.values()) > cap:
            # Find the largest group to merge within
            largest_key = max(groups, key=lambda k: len(groups[k]))
            grp = groups[largest_key]
            if len(grp) < 2:
                break
            grp.sort(key=lambda x: x.weight)

            # Merge the two closest by weight (adjacent after sort)
            best_idx = 0
            best_diff = float("inf")
            for j in range(len(grp) - 1):
                diff = abs(grp[j + 1].weight - grp[j].weight)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = j

            a = grp[best_idx]
            b = grp[best_idx + 1]

            # Save allometric params BEFORE mutation
            a_n = a.n_represented
            b_n = b.n_represented
            total_n = a_n + b_n

            if total_n > 0:
                # Weighted-mean weight (conserves biomass exactly)
                merged_weight = (a_n * a.weight + b_n * b.weight) / total_n
                merged_age = (a_n * a.age + b_n * b.age) / total_n
                merged_energy = (
                    a_n * a.energy_reserve + b_n * b.energy_reserve
                ) / total_n
                merged_dd = (a_n * a.degree_days + b_n * b.degree_days) / total_n
                merged_yolk = (
                    a_n * a.yolk_energy_kj + b_n * b.yolk_energy_kj
                ) / total_n

                a.n_represented = total_n
                a.weight = merged_weight
                a.age = merged_age
                a.energy_reserve = merged_energy
                a.degree_days = merged_dd
                a.yolk_energy_kj = merged_yolk
                a.starvation_days = (
                    a_n * a.starvation_days + b_n * b.starvation_days
                ) / total_n
                a.is_mature = a.is_mature or b.is_mature

                # Recompute length from allometry
                if a.life_stage < 3 and sp.larval is not None:
                    a.length = (
                        sp.larval.a_length_larval
                        * max(a.weight, 0.0) ** sp.larval.b_length_larval
                    )
                else:
                    a.length = (
                        sp.bioenerg.a_length
                        * max(a.weight, 0.0) ** sp.bioenerg.b_length
                    )

            # Remove b from the group
            grp.pop(best_idx + 1)

        # Rebuild self.individuals from groups
        self.individuals = []
        for grp in groups.values():
            self.individuals.extend(grp)

    def _resolve_forcing(self, env_forcing: dict, patch_idx: int) -> dict:
        """Resolve environment for a specific zone/patch.

        When ``env_forcing`` contains a ``'zone_forcing'`` dict keyed by
        zone index, the zone-specific values override the global defaults.
        If no zone override exists for the given *patch_idx*, the original
        ``env_forcing`` is returned unchanged.

        Parameters
        ----------
        env_forcing : dict
            Global environmental forcing dict, optionally containing a
            ``'zone_forcing'`` key mapping zone indices to override dicts.
        patch_idx : int
            Zone or patch index to resolve for.

        Returns
        -------
        dict
            Resolved forcing dict with zone-specific overrides applied.
        """
        zone_forcing = env_forcing.get("zone_forcing")
        if zone_forcing and patch_idx in zone_forcing:
            resolved = dict(env_forcing)  # shallow copy
            resolved.update(zone_forcing[patch_idx])
            return resolved
        return env_forcing

    def compute_step(
        self,
        prey_available: np.ndarray,
        predation_pressure: float,
        env_forcing: Dict[str, Any],
        dt: float,
        spatial_context: Optional[SpatialContext] = None,
    ) -> IBMStepResult:
        """Advance the SmeltIBM population by one time step.

        Executes up to five phases:

        1. **Forage + Grow**: For each individual, compute adaptive foraging
           allocation, then update weight and energy via bioenergetics.
        2. **Reproduce**: Mature females spawn; surviving larvae create recruits.
        3. **Predation mortality**: Apply size-structured predation.
        4. **Bookkeeping**: Age individuals, add recruits, remove senescent fish.
        5. **Spatial movement** (optional): Move individuals between patches
           when a spatial context is provided.

        Parameters
        ----------
        prey_available : np.ndarray
            1-D array of shape ``(n_groups,)`` giving available biomass
            per prey group.
        predation_pressure : float
            Total predation mortality rate on this group (yr^-1).
        env_forcing : Dict[str, Any]
            Environmental forcing with keys like ``"temperature"``,
            ``"month"``, ``"zoo_peak_day"``.
        dt : float
            Time step size (fraction of a year).
        spatial_context : SpatialContext, optional
            Spatial patch data for Ecospace simulations. When ``None``
            (default), no spatial movement is performed.

        Returns
        -------
        IBMStepResult
            Aggregated results of this time step.
        """
        sp = self.params
        temperature = env_forcing.get("temperature", 10.0)
        if "temperature" not in env_forcing and not self._temp_warning_issued:
            logger.debug(
                "SmeltIBM: 'temperature' missing from env_forcing, defaulting to 10.0"
            )
            self._temp_warning_issued = True
        month = env_forcing.get("month", 6)
        zoo_peak_day = env_forcing.get("zoo_peak_day", 120.0)

        biomass_before = self.get_aggregate_biomass()

        # Pre-compute zone forcing cache to avoid redundant _resolve_forcing calls
        _zone_cache: Dict[int, dict] = {}

        def _get_zone_forcing(zone_idx: int) -> dict:
            if zone_idx not in _zone_cache:
                _zone_cache[zone_idx] = self._resolve_forcing(env_forcing, zone_idx)
            return _zone_cache[zone_idx]

        # Convert prey_available ndarray to dict for adaptive_forage
        prey_dict: Dict[int, float] = {}
        for idx in range(len(prey_available)):
            if prey_available[idx] > 0.0:
                prey_dict[idx] = float(prey_available[idx])

        # Accumulate consumption across all individuals
        total_consumption = np.zeros(self.n_groups)

        # ================================================================
        # Phase 1a: Egg development (degree-day accumulation + hatching)
        # ================================================================
        if sp.egg is not None:
            dt_days = dt * 365.0
            # Support both 'dissolved_oxygen' and legacy 'o2' keys
            o2 = env_forcing.get("dissolved_oxygen", env_forcing.get("o2", _DEFAULT_O2))
            if (
                "dissolved_oxygen" not in env_forcing
                and "o2" not in env_forcing
                and not self._o2_warning_issued
            ):
                logger.debug(
                    "SmeltIBM: 'dissolved_oxygen'/'o2' missing from env_forcing, "
                    "defaulting to %.1f (no limitation)",
                    _DEFAULT_O2,
                )
                self._o2_warning_issued = True
            active_individuals: List[SuperIndividual] = []

            for ind in self.individuals:
                if ind.life_stage == 0:
                    # Resolve zone-specific forcing for this individual
                    ind_env = _get_zone_forcing(ind.patch_idx)
                    ind_temp = ind_env.get("temperature", temperature)
                    ind_o2 = ind_env.get(
                        "dissolved_oxygen", ind_env.get("o2", _DEFAULT_O2)
                    )
                    # Accumulate degree-days
                    ind.degree_days = accumulate_degree_days(
                        ind.degree_days, ind_temp, sp.egg.t_zero, dt_days
                    )
                    # Apply egg mortality
                    hypoxia_rate = (
                        sp.oxygen.hypoxia_mortality_rate
                        if sp.oxygen is not None
                        else 0.5
                    )
                    ind.n_represented = apply_egg_mortality(
                        n_represented=ind.n_represented,
                        background_rate=sp.egg.background_mortality_rate,
                        dt_days=dt_days,
                        o2=ind_o2,
                        o2_lethal=sp.egg.o2_lethal,
                        degree_days=ind.degree_days,
                        dd_mortality=sp.egg.dd_mortality,
                        hypoxia_mortality_rate=hypoxia_rate,
                    )
                    if ind.n_represented < 1.0:
                        continue  # egg cohort dead
                    # Check hatching
                    if check_hatching(ind.degree_days, sp.egg.dd_hatch):
                        ind.life_stage = 1
                        if sp.yolk_sac is not None:
                            ind.yolk_energy_kj = sp.yolk_sac.initial_yolk_kj
                    active_individuals.append(ind)
                elif ind.life_stage == 1 and sp.yolk_sac is not None:
                    # Phase 1b: Yolk-sac depletion + first feeding
                    # Resolve zone-specific forcing for this individual
                    ind_env = _get_zone_forcing(ind.patch_idx)
                    ind_temp = ind_env.get("temperature", temperature)
                    ind_o2 = ind_env.get(
                        "dissolved_oxygen", ind_env.get("o2", _DEFAULT_O2)
                    )
                    yolk_rate = compute_yolk_depletion(
                        weight=ind.weight,
                        temperature=ind_temp,
                        rs_a_larval=sp.larval.rs_a_larval if sp.larval else 0.12,
                        rs_b=sp.bioenerg.rb,
                        q10=sp.bioenerg.q10,
                        t_ref=sp.bioenerg.t_ref,
                        oxycal=sp.yolk_sac.oxycal_kj_per_g_o2,
                        dt_days=dt_days,
                    )
                    # Oxygen stress: accelerate yolk depletion under hypoxia
                    if sp.oxygen is not None and ind_o2 < sp.oxygen.pcrit_yolk_sac:
                        stress_factor = 1.0 + 0.5 * (
                            1.0 - ind_o2 / sp.oxygen.pcrit_yolk_sac
                        )
                        yolk_rate *= stress_factor
                    ind.yolk_energy_kj = max(0.0, ind.yolk_energy_kj - yolk_rate)

                    # Oxygen lethal mortality for yolk-sac larvae
                    if (
                        sp.oxygen is not None
                        and sp.oxygen.o2_lethal_yolk_sac > 0.0
                        and ind_o2 < sp.oxygen.o2_lethal_yolk_sac
                    ):
                        o2_mort_rate = (
                            sp.yolk_sac.background_mortality_rate
                            + sp.oxygen.hypoxia_mortality_rate
                            * (1.0 - ind_o2 / sp.oxygen.o2_lethal_yolk_sac)
                        )
                        ind.n_represented *= np.exp(-o2_mort_rate * dt_days)
                        if ind.n_represented < 1.0:
                            ind.n_represented = 0.0
                            continue  # cohort dead

                    # Determine zoo_density from zone-resolved env or prey_available
                    zoo_density = ind_env.get("zoo_density", None)
                    if zoo_density is None and sp.larval is not None:
                        pidx = sp.larval.zooplankton_prey_idx
                        if pidx < len(prey_available):
                            zoo_density = (
                                prey_available[pidx] * sp.larval.zoo_conversion_factor
                            )
                        else:
                            zoo_density = 0.0
                    elif zoo_density is None:
                        zoo_density = 0.0

                    status = check_first_feeding(
                        yolk_energy_kj=ind.yolk_energy_kj,
                        threshold_kj=sp.yolk_sac.first_feeding_threshold_kj,
                        zoo_density=zoo_density,
                        minimum_prey=sp.yolk_sac.minimum_prey_density,
                        starvation_days=ind.starvation_days,
                        pnr=sp.yolk_sac.point_of_no_return,
                    )

                    if status == "feed":
                        ind.life_stage = 2
                        ind.energy_reserve = ind.weight * 0.1
                        ind.starvation_days = 0.0
                        active_individuals.append(ind)
                    elif status == "dead":
                        ind.n_represented = 0.0
                        # do not add to active — effectively removed
                    elif status == "starving":
                        ind.starvation_days += dt_days
                        active_individuals.append(ind)
                    else:
                        # "yolk_sac" — still on yolk
                        active_individuals.append(ind)
                else:
                    active_individuals.append(ind)

            self.individuals = active_individuals

        # Filter: feeding individuals (life_stage >= 2) go through Phase 1;
        # eggs (0) and yolk-sac larvae (1) are excluded.
        if sp.egg is not None:
            feeding = [ind for ind in self.individuals if ind.life_stage >= 2]
            early = [ind for ind in self.individuals if ind.life_stage < 2]
        else:
            feeding = self.individuals
            early = []

        # ================================================================
        # Phase 1: Forage + Grow (with ontogenetic blending for larvae)
        # ================================================================
        n_ind = len(feeding)
        ind_consumptions = np.empty(n_ind)

        dt_days = dt * 365.0

        if n_ind > 0:
            weights = np.array([ind.weight for ind in feeding])

            # Sigmoid helper for blending
            def _sigmoid(x):
                return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

            # Foraging loop with consumption blending
            for i, ind in enumerate(feeding):
                w = ind.weight

                # Resolve zone-specific forcing for this individual
                ind_env = _get_zone_forcing(ind.patch_idx)
                ind_temp = ind_env.get("temperature", temperature)

                # Determine zoo_density for larval consumption (per-zone)
                zoo_density = ind_env.get("zoo_density", None)
                if zoo_density is None and sp.larval is not None:
                    pidx = sp.larval.zooplankton_prey_idx
                    if pidx < len(prey_available):
                        zoo_density = (
                            prey_available[pidx] * sp.larval.zoo_conversion_factor
                        )
                    else:
                        zoo_density = 0.0
                elif zoo_density is None:
                    zoo_density = 0.0

                # Oxygen limitation on consumption (per-zone)
                o2_feeding = ind_env.get(
                    "dissolved_oxygen", ind_env.get("o2", _DEFAULT_O2)
                )

                # Compute per-individual oxygen scalar based on life stage
                o2_scale = 1.0
                if sp.oxygen is not None:
                    if ind.life_stage == 2:
                        pcrit = sp.oxygen.pcrit_larva
                    elif ind.life_stage == 3:
                        pcrit = sp.oxygen.pcrit_juvenile
                    else:
                        pcrit = sp.oxygen.pcrit_adult
                    o2_scale = oxygen_scalar(o2_feeding, pcrit)

                if sp.larval is not None:
                    alpha = float(
                        _sigmoid(
                            (w - sp.larval.w_forage_mid) / sp.larval.w_forage_scale
                        )
                    )
                else:
                    alpha = 1.0  # pure adaptive foraging

                # --- Larval consumption (concentration-dependent) ---
                c_larval_vec = np.zeros(self.n_groups)
                if alpha < 0.99 and sp.larval is not None:
                    lp = sp.larval
                    f_temp = thornton_lessem(
                        ind_temp,
                        CQ=lp.cmax_CQ,
                        CTO=lp.cmax_CTO,
                        CTM=lp.cmax_CTM,
                        CTL=lp.cmax_CTL,
                        CK1=lp.cmax_CK1,
                        CK4=lp.cmax_CK4,
                    )
                    cmax = lp.cmax_c_a * (w**lp.cmax_c_b) * f_temp * dt_days
                    cmax *= o2_scale  # oxygen limitation
                    denom = zoo_density + lp.k_half_zoo
                    c_larval_scalar = cmax * (zoo_density / denom) if denom > 0 else 0.0
                    pidx = lp.zooplankton_prey_idx
                    if pidx < self.n_groups:
                        c_larval_vec[pidx] = c_larval_scalar

                # --- Adaptive foraging consumption ---
                c_adaptive_vec = np.zeros(self.n_groups)
                if alpha > 0.01:
                    # Pre-existing heuristic for the adaptive foraging path;
                    # differs from Thornton-Lessem Cmax intentionally to give
                    # a simpler, weight-only upper bound on adult consumption.
                    max_cons = 0.1 * (w**0.7) * dt_days
                    max_cons *= o2_scale  # oxygen limitation
                    allocation = adaptive_forage(
                        prey_available=prey_dict,
                        max_consumption=max_cons,
                        individual_length=ind.length,
                        params=sp.foraging,
                    )
                    for prey_idx, amount in allocation.items():
                        if prey_idx < self.n_groups:
                            c_adaptive_vec[prey_idx] = amount

                # --- Blend ---
                c_total_vec = (1.0 - alpha) * c_larval_vec + alpha * c_adaptive_vec

                ind_consumptions[i] = c_total_vec.sum()
                for prey_idx in range(self.n_groups):
                    if c_total_vec[prey_idx] > 0:
                        total_consumption[prey_idx] += (
                            c_total_vec[prey_idx] * ind.n_represented / 1e6
                        )

            # Batch growth step (vectorized bioenergetics)
            energy_reserves = np.array([ind.energy_reserve for ind in feeding])
            is_mature = np.array([ind.is_mature for ind in feeding])

            # Per-individual temperature from zone-specific forcing
            temps = np.array(
                [
                    _get_zone_forcing(ind.patch_idx).get("temperature", temperature)
                    for ind in feeding
                ]
            )

            if sp.larval is not None:
                new_weights, new_energies = growth_step_batch_ontogenetic(
                    weights=weights,
                    energy_reserves=energy_reserves,
                    consumptions=ind_consumptions,
                    temperature=temps,
                    is_mature=is_mature,
                    dt=dt,
                    bioenerg_params=sp.bioenerg,
                    larval_params=sp.larval,
                )
            else:
                new_weights, new_energies = growth_step_batch(
                    weights=weights,
                    energy_reserves=energy_reserves,
                    consumptions=ind_consumptions,
                    temperature=temps,
                    is_mature=is_mature,
                    dt=dt,
                    params=sp.bioenerg,
                )

            # Batch allometric length (use larval allometry based on life_stage)
            if sp.larval is not None:
                life_stages = np.array([ind.life_stage for ind in feeding])
                new_lengths = np.where(
                    life_stages < 3,
                    sp.larval.a_length_larval
                    * np.maximum(new_weights, 1e-6) ** sp.larval.b_length_larval,
                    sp.bioenerg.a_length
                    * np.maximum(new_weights, 1e-6) ** sp.bioenerg.b_length,
                )
            else:
                new_lengths = (
                    sp.bioenerg.a_length
                    * np.maximum(new_weights, 0.0) ** sp.bioenerg.b_length
                )

            # Write back to individuals
            for i, ind in enumerate(feeding):
                ind.weight = float(new_weights[i])
                ind.energy_reserve = float(new_energies[i])
                ind.length = float(new_lengths[i])

        # Apply oxygen-dependent lethal mortality to feeding larvae
        if sp.oxygen is not None and n_ind > 0:
            dt_days_mort = dt * 365.0
            for ind in feeding:
                if ind.life_stage == 2:
                    ind_env_o2 = _get_zone_forcing(ind.patch_idx)
                    o2_val = ind_env_o2.get(
                        "dissolved_oxygen", ind_env_o2.get("o2", _DEFAULT_O2)
                    )
                    if (
                        sp.oxygen.o2_lethal_larva > 0.0
                        and o2_val < sp.oxygen.o2_lethal_larva
                    ):
                        lp_bg = (
                            sp.larval.background_mortality_rate if sp.larval else 0.01
                        )
                        o2_mort = lp_bg + sp.oxygen.hypoxia_mortality_rate * (
                            1.0 - o2_val / sp.oxygen.o2_lethal_larva
                        )
                        ind.n_represented *= np.exp(-o2_mort * dt_days_mort)

            # Remove dead individuals
            feeding = [ind for ind in feeding if ind.n_represented >= 1.0]

        # Recombine early life stages with feeding individuals
        self.individuals = early + feeding

        # ================================================================
        # Phase 2: Reproduce
        # ================================================================
        recruits: List[SuperIndividual] = []
        # Approximate spawn day from month
        spawn_day = month * 30.0

        if sp.egg is not None:
            # ELS mode: collect eggs by zone and create egg cohorts
            eggs_by_zone: Dict[int, float] = defaultdict(float)
            for ind in self.individuals:
                eggs_from = spawn(ind, temperature, sp.reproduction)
                if eggs_from > 0:
                    eggs_by_zone[ind.patch_idx] += eggs_from
            for zone_idx, zone_eggs in eggs_by_zone.items():
                if zone_eggs > 0:
                    n_cohorts = min(
                        sp.egg.max_egg_cohorts,
                        max(1, int(zone_eggs / sp.egg.eggs_per_cohort)),
                    )
                    per_cohort = zone_eggs / n_cohorts
                    for _i in range(n_cohorts):
                        egg_si = SuperIndividual(
                            id=self._next_id,
                            n_represented=per_cohort,
                            weight=sp.egg.egg_weight,
                            length=sp.egg.egg_length_cm,
                            age=0.0,
                            energy_reserve=0.0,
                            patch_idx=zone_idx,
                            is_mature=False,
                            sex=0,
                            life_stage=0,
                            degree_days=0.0,
                        )
                        recruits.append(egg_si)
                        self._next_id += 1
        else:
            # Legacy mode: create recruits directly via create_recruits()
            for ind in self.individuals:
                total_eggs = spawn(ind, temperature, sp.reproduction)
                if total_eggs > 0.0:
                    new_recruits = create_recruits(
                        total_eggs=total_eggs,
                        spawn_day=spawn_day,
                        zoo_peak_day=zoo_peak_day,
                        patch_idx=ind.patch_idx,
                        next_id=self._next_id,
                        params=sp.reproduction,
                        n_super_individuals=1,
                    )
                    self._next_id += len(new_recruits)
                    recruits.extend(new_recruits)

        # ================================================================
        # Phase 3: Apply predation mortality
        # ================================================================
        n_before_predation = sum(ind.n_represented for ind in self.individuals)

        survivors = apply_predation_mortality(
            individuals=self.individuals,
            total_mortality_rate=predation_pressure,
            dt=dt,
            params=sp.predation,
        )

        n_after_predation = sum(ind.n_represented for ind in survivors)
        mortality_count = max(0.0, n_before_predation - n_after_predation)

        # ================================================================
        # Phase 4: Add recruits, age individuals, remove senescent fish
        # ================================================================
        # Age all survivors
        for ind in survivors:
            ind.age += dt

        # Juvenile transition: advance life_stage 2→3 when length threshold met
        if sp.larval is not None:
            juv_len = sp.larval.juvenile_length_cm
            for ind in survivors:
                if ind.life_stage == 2 and ind.length >= juv_len:
                    ind.life_stage = 3

        # Remove fish that exceeded max_age
        survivors = [ind for ind in survivors if ind.age <= sp.max_age]

        # Add recruits
        survivors.extend(recruits)

        # Update population
        self.individuals = survivors

        # Consolidate if over cap
        if len(self.individuals) > sp.max_super_individuals:
            self._consolidate_population()

        # Record consumption
        self._last_consumption = total_consumption

        # ================================================================
        # Phase 5: Spatial movement
        # ================================================================
        # Ontogenetic habitat constraints: allowed zones per life stage
        _ALLOWED_ZONES = {
            0: {0},  # Eggs: river only (sessile)
            1: {0, 1},  # Yolk-sac: river, lagoon
            2: {0, 1},  # Larvae: river, lagoon
            3: {1, 2},  # Juvenile: lagoon, coastal
            4: {0, 1, 2},  # Adult: all zones
        }

        patch_biomass = None
        if spatial_context is not None:
            for ind in self.individuals:
                probs = calculate_movement_probabilities(
                    current_patch=ind.patch_idx,
                    adjacency=spatial_context.adjacency,
                    habitat_quality=spatial_context.habitat_quality,
                    food_density=spatial_context.food_density,
                    predator_density=spatial_context.predator_density,
                    params=sp.movement,
                )
                ind.patch_idx = int(self._rng.choice(len(probs), p=probs))

            patch_biomass = self._aggregate_by_patch(spatial_context.n_patches)

        elif sp.zones is not None:
            # Standalone zonal mode (3-zone Curonian Lagoon model)
            conn = sp.zones.connectivity
            n_zones = len(conn)

            for ind in self.individuals:
                # Eggs: sessile — no movement
                if ind.life_stage == 0:
                    continue

                # Spawning migration: mature adults migrate to river
                if ind.life_stage == 4 and ind.is_mature:
                    ind_env = _get_zone_forcing(ind.patch_idx)
                    ind_temp = ind_env.get("temperature", temperature)
                    if should_migrate(ind_temp, month, sp.movement):
                        ind.patch_idx = 0  # migrate to spawning zone
                        continue

                # Passive drift for yolk-sac and larvae
                if ind.life_stage in (1, 2):
                    drift_rate = sp.zones.base_drift_rate
                    # Larvae have reduced drift as they grow
                    if ind.life_stage == 2:
                        drift_rate *= max(0.1, 1.0 - ind.length / 2.0)
                    if self._rng.random() < drift_rate:
                        probs = conn[ind.patch_idx]
                        new_zone = int(self._rng.choice(n_zones, p=probs))
                        allowed = _ALLOWED_ZONES.get(ind.life_stage, {0, 1, 2})
                        if new_zone in allowed:
                            ind.patch_idx = new_zone
                    continue

                # Active movement for juveniles and non-migrating adults
                if ind.life_stage >= 3:
                    probs = conn[ind.patch_idx].copy().astype(float)
                    # O2 behavioral avoidance: weight connectivity by oxygen quality
                    if sp.oxygen is not None:
                        pcrit = (
                            sp.oxygen.pcrit_juvenile
                            if ind.life_stage == 3
                            else sp.oxygen.pcrit_adult
                        )
                        for z in range(n_zones):
                            z_env = _get_zone_forcing(z)
                            z_o2 = z_env.get(
                                "dissolved_oxygen", z_env.get("o2", _DEFAULT_O2)
                            )
                            probs[z] *= oxygen_scalar(
                                z_o2, pcrit
                            ) * sp.oxygen.oxygen_avoidance_weight + (
                                1.0 - sp.oxygen.oxygen_avoidance_weight
                            )
                        # Renormalize
                        p_sum = probs.sum()
                        if p_sum > 0:
                            probs /= p_sum
                        else:
                            probs = conn[ind.patch_idx].copy().astype(float)
                    new_zone = int(self._rng.choice(n_zones, p=probs))
                    allowed = _ALLOWED_ZONES.get(ind.life_stage, {0, 1, 2})
                    if new_zone in allowed:
                        ind.patch_idx = new_zone

            patch_biomass = self._aggregate_by_patch(n_zones)

        # Compute results
        biomass_after = self.get_aggregate_biomass()
        production = biomass_after - biomass_before
        recruitment_count = sum(r.n_represented for r in recruits)

        return IBMStepResult(
            biomass=biomass_after,
            production=production,
            consumption_by_prey=total_consumption,
            mortality_count=mortality_count,
            recruitment_count=recruitment_count,
            patch_biomass=patch_biomass,
        )
