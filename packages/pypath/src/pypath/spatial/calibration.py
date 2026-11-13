"""EwE's Ecospace base calibration, mapped onto PyPath's consumption kernel.

EwE 6.7.144 sets Ecospace's search and vulnerability rates in
``AdjustSpaceParsNew`` (cEcoSpace.vb 6036-6180) so that, with every group
spread in proportion to its habitat capacity, map-wide consumption follows
Ecopath's:

- ``BRatio = ThabArea / TotHabCap``: a group's density in its habitat over
  its map-wide mean, and its base biomass ``B0 x BRatio``;
- ``Vspace = X x Qarena / (B0_prey x BRatio_prey)`` with
  ``Qarena = Q0 x BRatio_pred`` (one predator per arena), floored at
  Ecosim's vulnerability rate ``X x Q0 / B0_prey``;
- search rates from five passes of
  ``A = Q0 / (Vtemp x MeanBP)``, ``Vtemp = Vspace / (2 Vspace + A x Pbar)``,
  ``MeanBP`` the map-wide mean of ``HabCap_prey x HabCap_pred`` times the
  base densities, ``Pbar = B0_pred x BRatio_pred``; the passes stop early
  when the change is small, and on donor-controlled links (X near 1) they do
  not converge, so EwE's map-wide base flow there is below Ecopath's;
- producers' shading ``PbSpace = pbbiomass / BRatio``.

In a cell EwE eats ``A V B_prey P / (2V + A P / EatEff_pred)`` (handling
aside), with ``EatEff = HabCap``. PyPath's kernel eats
``QQ x PDY x PYY x X / (X - 1 + PDY / EatEff)`` on biomass relative to a base.
The two are the same with the base ``B0 x BRatio`` for every group and, per
link, ``X' = 1 + 2 Vspace / (A x Pb)`` and ``QQ' = Vspace x Bb_prey / X'``
(Pb, Bb the predator's and prey's bases). With a uniform habitat this gives
back Ecosim's X and Q0 exactly. A producer's gain is its link from group 0;
``PbSpace = pbbiomass / BRatio`` on the base ``B0 x BRatio`` is that link
with ``QQ' = Q0 x BRatio`` and its X unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["SpaceCalibration", "ewe_space_calibration", "habitat_ratio"]


@dataclass
class SpaceCalibration:
    """Ecospace's base parameters for PyPath's kernel.

    Attributes
    ----------
    bbase : np.ndarray
        Every group's base density, B0 x BRatio [n_groups + 1].
    qq : np.ndarray
        Base flows QQ' [prey, pred]; the producers' links from group 0 carry
        Q0 x BRatio.
    vv : np.ndarray
        Vulnerabilities X' [prey, pred].
    bratio : np.ndarray
        BRatio = map-wide mean / mean over the habitat [n_groups + 1].
    vspace : np.ndarray
        EwE's Vspace by link [prey, pred] (0 where there is no link).
    search : np.ndarray
        EwE's search rate before the handling correction (Atemp) [prey, pred].
    """

    bbase: np.ndarray
    qq: np.ndarray
    vv: np.ndarray
    bratio: np.ndarray
    vspace: np.ndarray
    search: np.ndarray


def habitat_ratio(
    capacity: np.ndarray, patch_share: Optional[np.ndarray] = None
) -> np.ndarray:
    """EwE's BRatio = ThabArea / TotHabCap of each group row of ``capacity``.

    ``capacity`` holds each group's habitat capacity per patch (rows by
    group). The ratio is 1 / the area-weighted mean capacity.
    """
    cap = np.asarray(capacity, dtype=float)
    n = cap.shape[1]
    w = (
        np.full(n, 1.0 / n)
        if patch_share is None
        else np.asarray(patch_share, dtype=float)
    )
    mean = (cap * w[np.newaxis, :]).sum(axis=1)
    return np.where(mean > 0, 1.0 / np.where(mean > 0, mean, 1.0), 1.0)


def ewe_space_calibration(
    b0: np.ndarray,
    qq: np.ndarray,
    vv: np.ndarray,
    capacity: np.ndarray,
    patch_share: Optional[np.ndarray] = None,
) -> SpaceCalibration:
    """EwE's AdjustSpaceParsNew, as PyPath kernel parameters.

    Parameters
    ----------
    b0 : np.ndarray
        Base biomass by group, index 0 the outside [n_groups + 1].
    qq : np.ndarray
        Ecopath base flows [prey, pred], row 0 the producers' links.
    vv : np.ndarray
        Ecosim vulnerabilities X [prey, pred].
    capacity : np.ndarray
        Habitat capacity per group and patch [n_groups + 1, n_patches]; row 0
        and groups without habitat are 1.
    patch_share : np.ndarray, optional
        Patch area fractions (equal patches when omitted).

    Returns
    -------
    SpaceCalibration
    """
    b0 = np.asarray(b0, dtype=float)
    q0 = np.asarray(qq, dtype=float)
    x0 = np.asarray(vv, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    n = cap.shape[1]
    w = (
        np.full(n, 1.0 / n)
        if patch_share is None
        else np.asarray(patch_share, dtype=float)
    )
    br = habitat_ratio(cap, w)
    br[0] = 1.0

    qq_new, vv_new = q0.copy(), x0.copy()
    vspace = np.zeros_like(q0)
    search = np.zeros_like(q0)
    prey, pred = np.nonzero(q0 > 0)
    keep = (prey >= 1) & (pred >= 1)
    prey, pred = prey[keep], pred[keep]
    if len(prey):
        q = q0[prey, pred]
        x = x0[prey, pred] + 1e-10  # EwE adds 1e-10 to VulMult
        qarena = q * br[pred]
        v = x * qarena / (b0[prey] * br[prey])
        v = np.where(v == 0, 1.0, v)
        # Ecosim's VulArena (cEcoSimModel.vb 5384) is Vspace's floor (vb 6090)
        v = np.maximum(v, x * q / b0[prey])
        vul_biom = (x - 1.0) * qarena / (2.0 * v)
        vul_biom = np.where(vul_biom == 0, 1.0, vul_biom)
        # the predator index: X' and QQ' do not depend on its scale
        a = np.where(vul_biom > 0, q / (vul_biom * b0[pred]), 0.0)
        pbar = b0 * br
        overlap = (cap * w[np.newaxis, :]) @ cap.T
        mean_bp = overlap[prey, pred] * b0[prey] * b0[pred] * br[prey] * br[pred]
        for _ in range(5):  # vb 6147-6174
            vtemp = v / (2.0 * v + a * pbar[pred])
            new = np.where(
                mean_bp > 0, q / (vtemp * np.where(mean_bp > 0, mean_bp, 1.0)), a
            )
            change = np.abs(new - a).sum()
            a = new
            if change < 0.01 * len(q):
                break
        pb = b0[pred] * br[pred]
        x_new = 1.0 + 2.0 * v / (a * pb)
        qq_new[prey, pred] = v * b0[prey] * br[prey] / x_new
        vv_new[prey, pred] = x_new
        vspace[prey, pred] = v
        search[prey, pred] = a
    # producers' links from group 0: PbSpace = pbbiomass / BRatio
    sun = np.nonzero(q0[0] > 0)[0]
    qq_new[0, sun] = q0[0, sun] * br[sun]
    return SpaceCalibration(
        bbase=b0 * br, qq=qq_new, vv=vv_new, bratio=br, vspace=vspace, search=search
    )
