"""Compressed link-list for sparse food webs.

Pre-computes arrays of active prey-predator link indices so that the
consumption kernel can iterate only over active links instead of the full
(NUM_GROUPS+1) x (NUM_GROUPS+1) matrix.  For a 50-group model with 70%
sparsity this eliminates 70% of iterations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ActiveLinkArray:
    """Pre-computed arrays of active prey-predator link indices.

    Attributes
    ----------
    prey : np.ndarray
        1-D int64 array of prey indices for each active link.
    pred : np.ndarray
        1-D int64 array of predator indices for each active link.
    n_links : int
        Number of active links.
    """

    prey: np.ndarray  # int64 shape (n_links,)
    pred: np.ndarray  # int64 shape (n_links,)
    n_links: int

    @classmethod
    def from_bool_matrix(cls, active: np.ndarray) -> "ActiveLinkArray":
        """Build an ActiveLinkArray from a boolean (or int) ActiveLink matrix.

        Parameters
        ----------
        active : np.ndarray
            Boolean or integer matrix of shape (N, N) where ``active[prey, pred]``
            is truthy when the link is active.

        Returns
        -------
        ActiveLinkArray
            Compressed representation containing only the active links.
        """
        prey, pred = np.nonzero(active)
        return cls(
            prey=prey.astype(np.int64),
            pred=pred.astype(np.int64),
            n_links=len(prey),
        )
