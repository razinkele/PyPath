"""Continuous integration pathway for Ecosim (independent of the EwE pathway).

The pathway's pieces: the handling-time solver (``solve_hden_kernel``, with
``solve_hden`` its full-derivative reference), the pure right-hand side
(``ContinuousRHS``) and the month integrator (``integrate_month``) that
``rsim_run`` calls for the continuous methods. Under EwE's rule the
derivative scales each predator's search rate by ``Hden0 / Hden`` where
``Hden`` satisfies the disc-equation fixed point

    Hden = 1 + Hden * q(Hden) / (CmCo * QB * index)

EwE takes one damped pass of that per derivative call; here it is solved
exactly at each state, on the consumption kernel alone (``HdenKernel``),
with a safeguarded Newton per predator on the kernel's analytic ``dq/dH``
(``iterate_hden``), or undamped passes where pooling couples the predators.

The package's modules, by responsibility: ``options`` (``continuous_options``,
``METHOD_NAMES``, the option modes, ``default_atol``), ``hden`` (handling
time), ``forcing`` (``neutral_forcing``, ``ForcingInterpolant``), ``feeding``
(feeding time), ``rhs`` (``ContinuousRHS``, ``exchange_rate``), ``jacobian``
(``JacobianReuse``), ``quasi_steady`` (``QuasiSteady``) and ``month``
(``integrate_month``, ``MonthResult``, ``month_budget``). Every public name is
re-exported here. A module global (``solve_ivp``, ``deriv_vector``, ``root``,
``solve_hden_kernel``, ``_ATOL_Z_MAX``) is read by the submodule that uses it,
so a test that monkeypatches one patches that submodule: setting this
package's attribute does not reach it.
"""

from pypath.core.ecosim_continuous.feeding import (
    FeedingTime,
    feeding_time,
    ftime_target,
)
from pypath.core.ecosim_continuous.forcing import (
    INTERPOLATED_KEYS,
    SIGNED_KEYS,
    ForcingInterpolant,
    neutral_forcing,
)
from pypath.core.ecosim_continuous.hden import (
    hden_index,
    iterate_hden,
    recalibrated_hden0,
    solve_hden,
    solve_hden_kernel,
)
from pypath.core.ecosim_continuous.jacobian import JacobianReuse
from pypath.core.ecosim_continuous.month import (
    MonthResult,
    integrate_month,
    month_budget,
)
from pypath.core.ecosim_continuous.month import _split_nodes as _split_nodes
from pypath.core.ecosim_continuous.options import (
    FAST_GROUP_MODES,
    FORCING_INTERPOLATIONS,
    JACOBIAN_MODES,
    METHOD_NAMES,
    POSITIVITY_MODES,
    QS_RATE_DEFAULT,
    continuous_options,
    default_atol,
)
from pypath.core.ecosim_continuous.quasi_steady import (
    QS_MAX_ITER,
    QS_PTC_MAX_ITER,
    QS_TOL,
    QuasiSteady,
    quasi_steady_groups,
)
from pypath.core.ecosim_continuous.rhs import (
    KERNEL_CHECK_TOL,
    NEGATIVE_EXPORT_TOL,
    SWITCH_NAMES,
    ContinuousRHS,
    continuous_emig,
    exchange_rate,
)

# in the order the names had in the single module
__all__ = [
    "recalibrated_hden0",
    "hden_index",
    "iterate_hden",
    "solve_hden",
    "solve_hden_kernel",
    "SWITCH_NAMES",
    "continuous_emig",
    "neutral_forcing",
    "FORCING_INTERPOLATIONS",
    "INTERPOLATED_KEYS",
    "SIGNED_KEYS",
    "ForcingInterpolant",
    "FeedingTime",
    "feeding_time",
    "ftime_target",
    "exchange_rate",
    "NEGATIVE_EXPORT_TOL",
    "KERNEL_CHECK_TOL",
    "ContinuousRHS",
    "METHOD_NAMES",
    "JACOBIAN_MODES",
    "POSITIVITY_MODES",
    "FAST_GROUP_MODES",
    "QS_RATE_DEFAULT",
    "continuous_options",
    "default_atol",
    "JacobianReuse",
    "QS_TOL",
    "QS_MAX_ITER",
    "QS_PTC_MAX_ITER",
    "quasi_steady_groups",
    "QuasiSteady",
    "MonthResult",
    "integrate_month",
    "month_budget",
]
