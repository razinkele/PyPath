"""The numba consumption kernel and its dispatch.

``_compute_consumption_sparse_numba`` must give exactly what the Python link
loop (``_compute_consumption_sparse_python``, the reference) gives, on every
optional branch; ``_compute_consumption_sparse`` is the numba wrapper when
numba is installed and ``PYPATH_NUMBA_CONSUMPTION`` is not ``0`` at import
time, and the Python loop otherwise.
"""

import inspect
import os
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim_deriv as deriv_mod
from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_deriv import _compute_consumption_sparse_python
from tests.continuous_support import needs_reco, reco_stanzas, scenario

SENTINEL = -7.0
N_TRIALS = 300
COUPLED_VALUES = (1, 0, 2, 0.5)
POOLING_MODES = ("none", "full", "full", "partial")
SWITCH_MODES = ("none", "ones", "random", "exact", "mixed")
EAT_MODES = ("none", "none", "none", "positive", "positive", "zeros", "view")


def _needs_numba():
    pytest.importorskip("numba")


def _sprinkle(rng, a, value, frac):
    """``a`` with about ``frac`` of its entries set to ``value``."""
    a = np.array(a, dtype=float, copy=True)
    a[rng.random(a.shape) < frac] = value
    return a


def _trial(trial):
    """The arguments of one random kernel call, and the branches it takes.

    Each trial has its own seed; the optional branches follow the trial
    number with coprime periods, so that every combination occurs.
    """
    rng = np.random.default_rng(1000 + trial)
    n = int(rng.integers(3, 30))
    act = rng.random((n, n)) < rng.uniform(0.1, 0.6)
    prey, pred = np.nonzero(act)
    if rng.random() < 0.5:  # links in another order than row-major
        order = rng.permutation(len(prey))
        prey, pred = prey[order], pred[order]
    prey = prey.astype(np.int32 if trial % 11 == 0 else np.int64)
    pred = pred.astype(np.int32 if trial % 11 == 0 else np.int64)
    n_links = len(prey)
    if trial % 13 == 0 and n_links > 2:  # only the first links
        n_links -= int(rng.integers(1, 3))

    BB = _sprinkle(rng, _sprinkle(rng, rng.random(n) * 2, 0.0, 0.1), -0.3, 0.05)
    VV = 1.0 + rng.random((n, n)) * 5
    VV = _sprinkle(rng, _sprinkle(rng, VV, 1.0, 0.1), 0.5, 0.05)
    VV = _sprinkle(rng, VV, 1e6, 0.03)
    DD = 1.0 + rng.random((n, n)) * 50
    DD = _sprinkle(rng, _sprinkle(rng, DD, 1.0, 0.1), 0.2, 0.05)
    QQbase = _sprinkle(rng, _sprinkle(rng, rng.random((n, n)), 0.0, 0.1), -0.1, 0.05)
    preyYY = _sprinkle(rng, _sprinkle(rng, rng.random(n) * 2, 0.0, 0.08), 1e-12, 0.08)
    preyYY = _sprinkle(rng, preyYY, 1e3, 0.03)
    predYY = _sprinkle(rng, _sprinkle(rng, rng.random(n) * 2, 0.0, 0.08), 1e-12, 0.08)
    predYY = _sprinkle(rng, predYY, 1e3, 0.03)
    kwargs = {}
    tags = {}

    pooling = POOLING_MODES[trial % len(POOLING_MODES)]
    if pooling != "none":
        scramble = _sprinkle(rng, _sprinkle(rng, rng.random((n, n)), 0.0, 0.1), 1, 0.1)
        handle = _sprinkle(rng, _sprinkle(rng, rng.random((n, n)), 0.0, 0.1), 1, 0.1)
        kwargs.update(ScrambleSelf=scramble, HandleSelf=handle)
        if pooling == "full":
            kwargs.update(
                PredPredWeight=rng.random((n, n)), PreyPreyWeight=rng.random((n, n))
            )
        # partial: no weights, so the loop does not pool
    tags["pooling"] = pooling == "full"

    switch = SWITCH_MODES[trial % len(SWITCH_MODES)]
    if switch == "ones":
        kwargs["HandleSwitch"] = np.ones((n, n))
    elif switch == "random":  # zeros, negatives and exponents other than 1
        hs = rng.random((n, n)) * 2.5
        kwargs["HandleSwitch"] = _sprinkle(rng, _sprinkle(rng, hs, 0.0, 0.2), -1, 0.05)
    elif switch == "exact":
        kwargs["HandleSwitch"] = rng.choice([0.5, 1.0, 2.0], size=(n, n))
    elif switch == "mixed":
        kwargs["HandleSwitch"] = rng.choice([0.0, 1.0], size=(n, n))
    tags["switch"] = switch != "none"
    kwargs["COUPLED"] = COUPLED_VALUES[(trial // 7) % len(COUPLED_VALUES)]

    tags["med"] = trial % 3 == 0
    if tags["med"]:
        med = 0.5 + rng.random((n, n))
        kwargs["med_multipliers"] = _sprinkle(
            rng, _sprinkle(rng, med, 0.0, 0.1), -1, 0.05
        )

    eat = EAT_MODES[trial % len(EAT_MODES)]
    if eat in ("positive", "zeros"):
        eat_eff = 0.1 + rng.random(n)
        if eat == "zeros":  # x / 0 and 0 / 0 in the predator term
            eat_eff = _sprinkle(rng, eat_eff, 0.0, 0.3)
            if n_links:
                eat_eff[pred[0]] = 0.0
        kwargs["eat_eff"] = eat_eff
    elif eat == "view":  # a patch column, as Ecospace passes it
        kwargs["eat_eff"] = (0.1 + rng.random((n, 3)))[:, 1]
    tags["eat"] = eat != "none"
    tags["eat_zero"] = eat == "zeros"

    args = (BB, VV, DD, QQbase, preyYY, predYY, prey, pred, n_links)
    return n, args, kwargs, tags


def _qq(kernel, n, args, kwargs):
    qq = np.full((n, n), SENTINEL)
    with np.errstate(all="ignore"):  # the loop's x / 0 warns; numba's does not
        result = kernel(qq, *args, **kwargs)
    assert result is None
    return qq


def test_the_wrapper_has_the_loops_signature():
    assert inspect.signature(
        deriv_mod._compute_consumption_sparse_numba
    ) == inspect.signature(_compute_consumption_sparse_python)


def test_random_inputs_give_identical_consumption_on_every_branch():
    _needs_numba()
    combos = Counter()
    seen = Counter()
    bad = []
    written = 0
    for trial in range(N_TRIALS):
        n, args, kwargs, tags = _trial(trial)
        ref = _qq(_compute_consumption_sparse_python, n, args, kwargs)
        got = _qq(deriv_mod._compute_consumption_sparse_numba, n, args, kwargs)
        if not np.array_equal(got, ref):
            with np.errstate(all="ignore"):
                diff = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
            bad.append((trial, tags, kwargs.get("COUPLED"), float(np.max(diff))))
        written += int(np.count_nonzero(ref != SENTINEL))
        combos[(tags["pooling"], tags["switch"], tags["med"], tags["eat"])] += 1
        seen.update(k for k, v in tags.items() if v)
        if tags["switch"]:
            seen[("COUPLED", kwargs["COUPLED"])] += 1
    assert not bad, f"{len(bad)} of {N_TRIALS} trials differ: {bad[:5]}"
    # the trials reach every branch, and every combination of the four
    assert len(combos) == 16, combos
    assert seen["eat_zero"] >= 20
    assert all(seen[("COUPLED", c)] >= 40 for c in COUPLED_VALUES), seen
    assert written > 10 * N_TRIALS


@pytest.mark.parametrize("pooling", [False, True])
def test_a_zero_feeding_efficiency_gives_the_loops_result(pooling):
    """EatEff 0 (an MPA capacity bonus of 0): the loop's x / 0 is inf and
    0 / 0 is nan, with a warning; the kernel must not raise instead."""
    _needs_numba()
    n = 5
    rng = np.random.default_rng(3)
    prey, pred = np.nonzero(np.ones((n, n), dtype=bool))
    pred_yy = np.array([1.0, 1.2, 0.0, 0.8, 1.1])  # group 2: 0 / 0
    eat_eff = np.array([1.0, 0.0, 0.0, 0.9, 0.5])  # group 1: 1.2 / 0
    kwargs = {"eat_eff": eat_eff}
    if pooling:
        kwargs.update(
            ScrambleSelf=np.full((n, n), 0.5),
            HandleSelf=np.full((n, n), 0.5),
            PredPredWeight=np.ones((n, n)),
            PreyPreyWeight=np.ones((n, n)),
        )
    args = (
        np.ones(n),
        2.0 + rng.random((n, n)),
        2.0 + rng.random((n, n)),
        rng.random((n, n)),
        np.ones(n),
        pred_yy,
        prey.astype(np.int64),
        pred.astype(np.int64),
        len(prey),
    )
    ref = _qq(_compute_consumption_sparse_python, n, args, kwargs)
    got = _qq(deriv_mod._compute_consumption_sparse_numba, n, args, kwargs)
    np.testing.assert_array_equal(got, ref)
    assert np.all(ref[:, 1] == 0.0) and np.all(ref[:, 2] == 0.0)


@pytest.mark.parametrize("short", ["prey", "pred"])
def test_more_links_than_the_link_arrays_raise_before_the_kernel(short):
    """numba does no bounds checking: an ``n_links`` past either link array
    raises ValueError, and the kernel writes nothing."""
    _needs_numba()
    n, args, kwargs, _ = _trial(1)  # pooling and HandleSwitch, no eat_eff
    args = list(args)
    prey, pred = args[6], args[7]
    assert len(prey) > 3
    if short == "prey":
        args[6] = prey[:-1]
    else:
        args[7] = pred[:-1]
    args[8] = len(prey)
    qq = np.full((n, n), SENTINEL)
    with pytest.raises(ValueError, match="n_links"):
        deriv_mod._compute_consumption_sparse_numba(qq, *args, **kwargs)
    assert np.all(qq == SENTINEL)


# --- dispatch -------------------------------------------------------------


def _alias_in_subprocess(env_value, block_numba=False):
    """``(alias name, HAS_NUMBA)`` of a fresh import of ``ecosim_deriv``."""
    env = dict(os.environ)
    env.pop(deriv_mod.NUMBA_CONSUMPTION_ENV, None)
    if env_value is not None:
        env[deriv_mod.NUMBA_CONSUMPTION_ENV] = env_value
    src = str(Path(deriv_mod.__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    code = (
        "import sys\n"
        + ("sys.modules['numba'] = None\n" if block_numba else "")
        + "import pypath.core.ecosim_deriv as ed\n"
        "print(ed._compute_consumption_sparse.__name__, ed.HAS_NUMBA)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    name, has_numba = out.stdout.split()[-2:]
    return name, has_numba == "True"


def test_the_alias_is_the_numba_wrapper_by_default():
    _needs_numba()
    assert _alias_in_subprocess(None) == ("_compute_consumption_sparse_numba", True)


def test_the_switch_at_0_keeps_the_python_loop():
    assert _alias_in_subprocess("0")[0] == "_compute_consumption_sparse_python"


def test_without_numba_the_alias_is_the_python_loop():
    assert _alias_in_subprocess(None, block_numba=True) == (
        "_compute_consumption_sparse_python",
        False,
    )


@pytest.mark.parametrize(
    "value, enabled",
    [
        (None, True),
        ("", True),
        ("1", True),
        ("0", False),
        (" 0 ", False),
        ("false", False),
        ("OFF", False),
        ("no", False),
    ],
)
def test_the_switch_spellings(value, enabled):
    environ = {} if value is None else {deriv_mod.NUMBA_CONSUMPTION_ENV: value}
    assert deriv_mod._numba_consumption_requested(environ) is enabled


def test_this_process_follows_the_switch():
    off = not deriv_mod._numba_consumption_requested(os.environ)
    if deriv_mod.HAS_NUMBA and not off:
        expected = deriv_mod._compute_consumption_sparse_numba
    else:
        expected = _compute_consumption_sparse_python
    assert deriv_mod._compute_consumption_sparse is expected


# --- end to end -----------------------------------------------------------


@needs_reco
@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_a_reco_run_is_identical_with_the_numba_kernel(rule, monkeypatch):
    _needs_numba()
    years = range(1, 3)
    outs = {}
    kernels = {
        "python": _compute_consumption_sparse_python,
        "numba": deriv_mod._compute_consumption_sparse_numba,
    }
    for name, kernel in kernels.items():
        calls = {"n": 0}

        def counted(*args, _kernel=kernel, _calls=calls, **kwargs):
            _calls["n"] += 1
            return _kernel(*args, **kwargs)

        monkeypatch.setattr(deriv_mod, "_compute_consumption_sparse", counted)
        scen = scenario(reco_stanzas(), rule=rule, years=years)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outs[name] = rsim_run(scen, method="RK4", years=years)
        assert calls["n"] > 100, name  # the run goes through the kernel
    assert deriv_mod._compute_consumption_sparse_numba_kernel.signatures
    py, nb = outs["python"], outs["numba"]
    assert np.array_equal(nb.out_Biomass, py.out_Biomass)
    assert np.array_equal(nb.out_Catch, py.out_Catch)
    assert np.count_nonzero(py.out_Catch) > 0
