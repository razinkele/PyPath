"""Recruitment support for Ecosim scenarios.

Monkey-patches ``RsimScenario`` with recruitment model registration,
trajectory helpers, and optional larval connectivity application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from pypath.core.ecosim import RsimScenario

if TYPE_CHECKING:
    from pypath.core.recruitment import LarvalConnectivity, RecruitmentModel


def add_recruitment_support() -> None:
    """Monkey-patch RsimScenario with recruitment methods."""
    if hasattr(RsimScenario, "set_recruitment_model"):
        return

    def _init_recruitment(self: RsimScenario) -> None:
        if not hasattr(self, "_recruitment_models"):
            self._recruitment_models: Dict[str, RecruitmentModel] = {}
        if not hasattr(self, "_larval_connectivity"):
            self._larval_connectivity: Optional[LarvalConnectivity] = None

    def set_recruitment_model(
        self: RsimScenario, group_name: str, model: RecruitmentModel
    ) -> None:
        from pypath.core.recruitment import RecruitmentModel

        _init_recruitment(self)
        if group_name not in self.params.spname:
            raise ValueError(f"Unknown group: {group_name}")
        if not isinstance(model, RecruitmentModel):
            raise TypeError(
                "model must be an instance of RecruitmentModel, "
                f"got {type(model).__name__}"
            )
        self._recruitment_models[group_name] = model

    def get_recruitment_model(
        self: RsimScenario, group_name: str
    ) -> Optional[RecruitmentModel]:
        _init_recruitment(self)
        return self._recruitment_models.get(group_name)

    def list_recruitment_models(self: RsimScenario) -> List[str]:
        _init_recruitment(self)
        return sorted(self._recruitment_models.keys())

    def set_larval_connectivity(
        self: RsimScenario, connectivity: Optional[LarvalConnectivity]
    ) -> None:
        from pypath.core.recruitment import LarvalConnectivity

        _init_recruitment(self)
        if connectivity is not None and not isinstance(connectivity, LarvalConnectivity):
            raise TypeError(
                "connectivity must be LarvalConnectivity or None, "
                f"got {type(connectivity).__name__}"
            )
        self._larval_connectivity = connectivity

    def apply_recruitment(
        self: RsimScenario,
        time: float,
        ssb_by_group: Dict[str, float],
        env_factors: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        _init_recruitment(self)

        time_val = float(time)
        outputs: Dict[str, float] = {}
        for group in self.params.spname:
            if group not in self._recruitment_models:
                outputs[group] = 1.0
                continue
            model = self._recruitment_models[group]
            ssb = float(ssb_by_group.get(group, 0.0))
            env = 1.0 if env_factors is None else float(env_factors.get(group, 1.0))
            outputs[group] = float(model.get_recruitment(ssb=ssb, time=time_val, env_factor=env))

        if self._larval_connectivity is not None:
            dispersed = self._larval_connectivity.disperse(outputs)
            outputs.update(dispersed)

        return outputs


    def build_recruitment_vector(
        self: RsimScenario,
        time: float,
        ssb_by_group: Dict[str, float],
        env_factors: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Build (NUM_GROUPS+1)-length recruitment multiplier vector."""
        outputs = apply_recruitment(self, time, ssb_by_group, env_factors)
        vector = np.ones(self.params.NUM_GROUPS + 1, dtype=float)
        for idx, group in enumerate(self.params.spname):
            if idx >= vector.size:
                break
            vector[idx] = float(outputs.get(group, 1.0))
        return vector

    def apply_recruitment_to_forcing(
        self: RsimScenario,
        month_index: int,
        time: float,
        ssb_by_group: Dict[str, float],
        env_factors: Optional[Dict[str, float]] = None,
    ) -> None:
        """Write recruitment multipliers into scenario.forcing.ForcedRecs."""
        _init_recruitment(self)
        if not hasattr(self, "forcing") or getattr(self.forcing, "ForcedRecs", None) is None:
            raise ValueError("Scenario forcing.ForcedRecs is not available")
        if month_index < 0 or month_index >= self.forcing.ForcedRecs.shape[0]:
            raise ValueError(f"month_index out of bounds: {month_index}")

        self.forcing.ForcedRecs[month_index, :] = build_recruitment_vector(
            self,
            time=time,
            ssb_by_group=ssb_by_group,
            env_factors=env_factors,
        )

    def get_recruitment_trajectory(
        self: RsimScenario,
        group_name: str,
        ssb_timeline: Dict[float, float],
        times: np.ndarray,
    ) -> np.ndarray:
        _init_recruitment(self)
        model = self._recruitment_models.get(group_name)
        if model is None:
            raise ValueError(f"No recruitment model registered for group: {group_name}")
        return model.get_recruitment_trajectory(ssb_timeline=ssb_timeline, times=np.asarray(times))

    def batch_recruitment_evaluation(
        self: RsimScenario,
        groups: List[str],
        ssb_matrix: np.ndarray,
        time_array: np.ndarray,
        env_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        _init_recruitment(self)

        ssb = np.asarray(ssb_matrix, dtype=float)
        times = np.asarray(time_array, dtype=float).reshape(-1)
        if ssb.ndim != 2:
            raise ValueError("ssb_matrix must be a 2D array")
        if ssb.shape[0] != len(groups):
            raise ValueError("ssb_matrix row count must match number of groups")
        if ssb.shape[1] != len(times):
            raise ValueError("ssb_matrix column count must match time_array length")

        if env_matrix is not None:
            env = np.asarray(env_matrix, dtype=float)
            if env.shape != ssb.shape:
                raise ValueError("env_matrix must have same shape as ssb_matrix")
        else:
            env = None

        result: Dict[str, np.ndarray] = {}
        for i, group in enumerate(groups):
            model = self._recruitment_models.get(group)
            if model is None:
                result[group] = np.ones(len(times), dtype=float)
                continue

            group_env = None if env is None else env[i, :]
            result[group] = model.get_recruitment_batch(
                ssb_array=ssb[i, :],
                time_array=times,
                env_factors=group_env,
            )
        return result

    RsimScenario._init_recruitment = _init_recruitment
    RsimScenario.set_recruitment_model = set_recruitment_model
    RsimScenario.get_recruitment_model = get_recruitment_model
    RsimScenario.list_recruitment_models = list_recruitment_models
    RsimScenario.set_larval_connectivity = set_larval_connectivity
    RsimScenario.apply_recruitment = apply_recruitment
    RsimScenario.build_recruitment_vector = build_recruitment_vector
    RsimScenario.apply_recruitment_to_forcing = apply_recruitment_to_forcing
    RsimScenario.get_recruitment_trajectory = get_recruitment_trajectory
    RsimScenario.batch_recruitment_evaluation = batch_recruitment_evaluation


add_recruitment_support()
