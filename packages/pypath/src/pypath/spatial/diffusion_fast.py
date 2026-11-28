"""Implicit diffusion of all moving groups in one numba call.

``rsim_run_spatial`` diffuses each moving group by backward Euler,
``implicit_diffusion_step``: a solve with the group's SuperLU factorisation
(``diffusion_solver``), one SciPy call per group and step. This module
solves the same systems with the same factors, ``Pr A Pc = L U`` (SciPy's
``SuperLU.L``, ``.U``, ``.perm_r``, ``.perm_c``), by forward and backward
substitution in compiled code, every group in one call, the groups over
threads.

The factors are SuperLU's, but its solve adds the terms of each row in its
own order (blocked by supernodes), so the results agree with
``SuperLU.solve`` to round-off (a few 1e-16 relative per step), not bit for
bit. ``PYPATH_FAST_ECOSPACE_MONTH=0`` keeps SciPy's solve.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from pypath.core.ecosim_deriv import HAS_NUMBA

if HAS_NUMBA:
    import numba

    _njit_par = numba.njit(cache=True, nogil=True, error_model="numpy", parallel=True)
    _prange = numba.prange
else:  # pragma: no cover - the module is not used without numba

    def _njit_par(f):
        return f

    _prange = range


@_njit_par
def _solve_rows(
    B, rows, perm_r, perm_c, Lp, Li, Lx, Up, Ui, Ux, Udiag, loff, uoff, n_chunks
):
    """``B[rows[g]] <- A_g^-1 B[rows[g]]`` for each group g, in place, with
    ``Pr A Pc = L U``: ``x = Pc U^-1 L^-1 Pr b``. ``L`` (unit lower) and
    ``U`` (upper) are CSC, the groups' entries concatenated (offsets
    ``loff``, ``uoff``); ``Udiag`` the position of each column's diagonal in
    ``Ui``. The groups are split into ``n_chunks`` chunks, one per thread."""
    n_rows = len(rows)
    n = B.shape[1]
    if n_chunks < 1:
        n_chunks = 1
    if n_chunks > n_rows:
        n_chunks = n_rows
    for c in _prange(n_chunks):
        z = np.empty(n)
        ci = np.int64(c)
        for g in range(ci * n_rows // n_chunks, (ci + 1) * n_rows // n_chunks):
            r = rows[g]
            pr = perm_r[g]
            pc = perm_c[g]
            for i in range(n):
                z[pr[i]] = B[r, i]
            lp = Lp[g]
            o = loff[g]
            for j in range(n):  # forward substitution, L unit diagonal
                zj = z[j]
                if zj != 0.0:
                    for k in range(lp[j], lp[j + 1]):
                        i = Li[o + k]
                        if i > j:
                            z[i] -= Lx[o + k] * zj
            up = Up[g]
            o = uoff[g]
            ud = Udiag[g]
            for j in range(n - 1, -1, -1):  # backward substitution
                zj = z[j] / Ux[o + ud[j]]
                z[j] = zj
                if zj != 0.0:
                    for k in range(up[j], up[j + 1]):
                        i = Ui[o + k]
                        if i < j:
                            z[i] -= Ux[o + k] * zj
            for i in range(n):
                B[r, i] = z[pc[i]]


class DiffusionBatch:
    """The moving groups' implicit diffusion step, compiled.

    Built from ``[(row, solver), ...]`` (``diffusion_solver``'s SuperLU or
    None); :meth:`apply` does what ``implicit_diffusion_step`` does for each
    row, ``biomass[row] = solver.solve(biomass[row])``, rows without a solver
    unchanged.
    """

    def __init__(self, solvers, n_patches: int):
        live = [(int(row), lu) for row, lu in solvers if lu is not None]
        self.n_patches = int(n_patches)
        self.n_groups = len(live)
        if not live:
            return
        rows, prs, pcs, lps, ups, udiag = [], [], [], [], [], []
        lis, lxs, uis, uxs, loff, uoff = [], [], [], [], [], []
        ol = ou = 0
        for row, lu in live:
            L = scipy.sparse.csc_matrix(lu.L)
            U = scipy.sparse.csc_matrix(lu.U)
            L.sort_indices()
            U.sort_indices()
            n = L.shape[0]
            if n != self.n_patches:
                raise ValueError("a solver does not match the number of patches")
            # sorted CSC: an upper triangular column ends with its diagonal
            diag = np.asarray(U.indptr[1:], dtype=np.int64) - 1
            if not (
                np.all(U.indptr[1:] > U.indptr[:-1])
                and np.array_equal(U.indices[diag], np.arange(n))
            ):
                raise ValueError("U has no diagonal entry in a column")
            rows.append(row)
            prs.append(lu.perm_r)
            pcs.append(lu.perm_c)
            lps.append(L.indptr)
            ups.append(U.indptr)
            udiag.append(diag)
            lis.append(L.indices)
            lxs.append(L.data)
            uis.append(U.indices)
            uxs.append(U.data)
            loff.append(ol)
            uoff.append(ou)
            ol += L.nnz
            ou += U.nnz
        i64 = np.int64
        self._args = (
            np.array(rows, i64),
            np.array(prs, i64),
            np.array(pcs, i64),
            np.array(lps, i64),
            np.ascontiguousarray(np.concatenate(lis), i64),
            np.ascontiguousarray(np.concatenate(lxs), np.float64),
            np.array(ups, i64),
            np.ascontiguousarray(np.concatenate(uis), i64),
            np.ascontiguousarray(np.concatenate(uxs), np.float64),
            np.array(udiag, i64),
            np.array(loff, i64),
            np.array(uoff, i64),
        )

    def apply(self, biomass: np.ndarray, n_threads: int = 1) -> None:
        """Diffuse ``biomass`` (groups x patches, C-contiguous float64) in
        place, the groups over at most ``n_threads`` threads."""
        if self.n_groups == 0:
            return
        if not (
            biomass.dtype == np.float64
            and biomass.flags.c_contiguous
            and biomass.ndim == 2
            and biomass.shape[1] == self.n_patches
        ):
            raise ValueError("biomass must be C-contiguous float64 (groups x patches)")
        k = max(1, min(int(n_threads), numba.config.NUMBA_NUM_THREADS, self.n_groups))
        prev = numba.get_num_threads()
        if k != prev:
            numba.set_num_threads(k)
        try:
            _solve_rows(biomass, *self._args, k)
        finally:
            if k != prev:
                numba.set_num_threads(prev)
