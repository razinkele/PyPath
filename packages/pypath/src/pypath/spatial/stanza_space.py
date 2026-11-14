"""EwE's NewMultiStanza distribution of stanza groups over Ecospace patches.

EwE 6.7.144's Ecospace (ModelType 2, its default for a new scenario) does not
integrate a multi-stanza group cell by cell. Each life stage keeps a
population index W per cell (``Bcell(nvar2 + isc)``), solved with the other
groups each time step (``cSpaceSolver.vb`` 665-720):

    dW/dt = Rflow - Z W + movement

- Z is the cell's loss rate of the stage group, loss / B (``FlowoutRate``),
  or 100 where the cell holds none;
- Rflow is RelR (1 - exp(-Z Tstanza)) for the first stage, with RelR the
  cell's adult density over the adults' start biomass (``RelRepStanza``), and
  the previous stage's density for the others;
- W moves between cells like the stage group itself. PyPath's index
  carries the capacity-weighted diffusion only, so a species with a stage
  moved by an external flux or habitat-directed advection keeps the
  patch-share rule (``rsim_run_spatial`` leaves it out).

EwE leaves the stage groups out of the integration (``ByPassIntegrate``).
Every step it updates the map-wide age structure with the cells' rates
averaged by W (``SpaceSplitUpdate``, ``cEcoSpace.vb`` 7451-7470) and spreads
each stage's new total over the cells in proportion to W
(``UpdateMultiStanza``, 7406-7445). The W it spreads by is the one read
before the step's solve. The index starts at RecSplit / Z x the cell's
habitat capacity over the map's mean (``NstanzaBase``, 3187-3210 and 2569).

PyPath keeps biomass as amounts per patch (density x area share), so W, a
density, weighs each patch by its area. One step is backward Euler on the
local terms, then the stage's implicit movement step on the amounts W x area.
EwE solves both together, with its second-order time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["StanzaSpaceIndex"]

#: EwE's loss rate of a cell without the group (cSpaceSolver.vb 1029)
EMPTY_CELL_LOSS_RATE = 100.0
_EMPTY = 1e-20


@dataclass
class StanzaSpaceIndex:
    """Per-patch population index of each life stage (EwE's NewMultiStanza).

    Parameters
    ----------
    species : list of (rows, tstanza)
        For each multi-stanza species, its stage groups' state rows (Rsim
        indexing, youngest first) and each stage's duration in years,
        (Age2 - Age1 + 1) / 12.
    patch_share : np.ndarray
        Each patch's share of the map's area.
    """

    species: List[Tuple[List[int], List[float]]]
    patch_share: np.ndarray
    index: Dict[int, np.ndarray] = field(default_factory=dict)
    adult_start: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_stanzas(cls, stanzas, patch_share, leave_out=()) -> "StanzaSpaceIndex":
        """Read the stages from an ``RsimStanzas``-like object.

        A species with a stage row in ``leave_out`` gets no index.
        """
        leave_out = {int(r) for r in leave_out}
        code = np.asarray(stanzas.ecopath_code, dtype=float)
        age1 = np.asarray(stanzas.age1, dtype=float)
        age2 = np.asarray(stanzas.age2, dtype=float)
        n_stanzas = np.asarray(stanzas.n_stanzas)
        species = []
        for isp in range(1, int(stanzas.n_split) + 1):
            stages = range(1, int(n_stanzas[isp]) + 1)
            if leave_out & {int(code[isp, ist]) for ist in stages}:
                continue
            species.append(
                (
                    [int(code[isp, ist]) for ist in stages],
                    [(age2[isp, ist] - age1[isp, ist] + 1) / 12.0 for ist in stages],
                )
            )
        return cls(species, np.asarray(patch_share, dtype=float))

    @property
    def started(self) -> bool:
        return bool(self.index)

    def _loss_rate(self, biomass, loss_prop_to_b, row):
        b = biomass[row]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                b > _EMPTY,
                loss_prop_to_b[row] / np.where(b > _EMPTY, b, 1.0),
                EMPTY_CELL_LOSS_RATE,
            )

    def start(self, biomass: np.ndarray, loss_prop_to_b: np.ndarray) -> None:
        """Set EwE's start index, NstanzaBase x relative capacity.

        ``biomass`` and ``loss_prop_to_b`` hold each patch's amounts at the
        start; the map-wide loss rate stands in for EwE's FlowoutRate at the
        start biomass, and a patch's density over the map's mean for
        HabCap x nWater / TotHabCap.
        """
        for rows, tstanza in self.species:
            survivors = 1.0
            for k, row in enumerate(rows):
                total = float(biomass[row].sum())
                z = float(loss_prop_to_b[row].sum()) / total if total > 0 else 0.0
                remaining = survivors * np.exp(-tstanza[k] * z)
                rec = survivors - remaining if k < len(rows) - 1 else survivors
                survivors = remaining
                relative = (
                    biomass[row] / self.patch_share / total
                    if total > 0
                    else np.ones_like(self.patch_share)
                )
                self.index[row] = rec / z * relative if z > 0 else relative.copy()
            self.adult_start[rows[-1]] = float(biomass[rows[-1]].sum())

    def weighted_sums(self, biomass, food_gain, loss_prop_to_b):
        """Map-wide biomass, consumption and losses for the age structure.

        Groups without stanzas get the plain sums over the patches. A stage
        gets EwE's averages over the cells weighted by W (TotBiom, TotEatenBy,
        TotLoss). EwE's cells all count alike (ThabArea counts them); here a
        patch counts by its area, so each is the sum of the patch amounts x W
        over the sum of share x W, with share the patch's part of the stage's
        biomass: the plain sums when W is uniform, and EwE's sums of cell
        densities x W when the patches have equal areas. Divided by the age
        structure's map-wide pred, the consumption gives EwE's
        TotEatenBy / TotPred, as each cell's pred is its share of the map's.
        """
        sums = [
            np.asarray(a, dtype=float).sum(axis=1)
            for a in (biomass, food_gain, loss_prop_to_b)
        ]
        for row, w in self.index.items():
            total = biomass[row].sum()
            share = biomass[row] / total if total > 0 else self.patch_share
            norm = float(np.sum(share * w))
            if norm > 0:
                for out, a in zip(sums, (biomass, food_gain, loss_prop_to_b)):
                    out[row] = float(np.sum(a[row] * w)) / norm
        return tuple(sums)

    def distribute(self, total: np.ndarray, biomass: np.ndarray) -> np.ndarray:
        """Spread each stage's map-wide total over the patches by W.

        Returns a copy of ``biomass`` with the stage rows replaced; a stage
        whose index is zero everywhere keeps its current shares.
        """
        out = np.array(biomass, dtype=float, copy=True)
        for row, w in self.index.items():
            amounts = w * self.patch_share
            s = amounts.sum()
            if s > 0:
                out[row] = total[row] * amounts / s
            else:
                current = out[row].sum()
                share = out[row] / current if current > 0 else self.patch_share
                out[row] = total[row] * share
        return out

    def advance(
        self,
        biomass: np.ndarray,
        loss_prop_to_b: np.ndarray,
        dt: float,
        move: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    ) -> None:
        """One time step of the index from the step's start state.

        ``move(row, amounts)``, if given, moves the amounts W x area of the
        stage group ``row`` over the step and returns them.
        """
        density = biomass / self.patch_share[None, :]
        for rows, tstanza in self.species:
            adult = rows[-1]
            relr = (
                density[adult] / self.adult_start[adult]
                if self.adult_start[adult] > 0
                else 0.0
            )
            for k, row in enumerate(rows):
                z = self._loss_rate(biomass, loss_prop_to_b, row)
                if k == 0:
                    rflow = relr * (1.0 - np.exp(-z * tstanza[0]))
                else:
                    rflow = density[rows[k - 1]]
                w = (self.index[row] + dt * rflow) / (1.0 + dt * z)
                if move is not None:
                    w = move(row, w * self.patch_share) / self.patch_share
                self.index[row] = np.maximum(w, 0.0)
