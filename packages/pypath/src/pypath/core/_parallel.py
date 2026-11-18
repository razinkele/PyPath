"""Worker processes for ensemble analyses (Monte Carlo, sensitivity).

Both analyses draw their whole design up front in the parent, then evaluate
it point by point. These helpers run the evaluations in a process pool and
hand the outputs back in order, so a parallel run gives the serial run's
results, bit for bit, as long as each evaluation runs under
:data:`NUMPY_DEFAULT_ERRSTATE` with warnings ignored.
"""

from __future__ import annotations

import collections
import contextlib
import itertools
import multiprocessing
import numbers
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

# numpy's default floating-point error handling. Each evaluation runs under
# it with warnings ignored, so its outcome does not depend on the process it
# runs in: a worker does not inherit the caller's warning filters or
# np.seterr.
NUMPY_DEFAULT_ERRSTATE = dict(
    divide="warn", over="warn", under="ignore", invalid="warn"
)


def resolve_n_jobs(n_jobs, n_tasks: int) -> int:
    """Number of worker processes for ``n_jobs``: 1 runs serially.

    ``n_jobs`` >= 1 asks for that many workers and -1 for one per CPU this
    process may use. The count never exceeds the number of tasks, nor 61 on
    Windows (ProcessPoolExecutor's limit there), and is at least 1.
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral):
        raise ValueError(f"n_jobs must be an integer >= 1 or -1, not {n_jobs!r}")
    if n_jobs == -1:
        wanted = usable_cpus()
    elif n_jobs >= 1:
        wanted = int(n_jobs)
    else:
        raise ValueError(f"n_jobs must be an integer >= 1 or -1, not {n_jobs!r}")
    if sys.platform == "win32":
        wanted = min(wanted, 61)
    return max(1, min(wanted, n_tasks))


def usable_cpus() -> int:
    """CPUs this process may run on (affinity, cgroup masks), at least 1."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def mp_context(name: str | None):
    """The workers' start method: forkserver on Linux, spawn on macOS and
    Windows (CPython's own defaults from 3.14), or the one named."""
    if name is None:
        methods = multiprocessing.get_all_start_methods()
        if sys.platform in ("darwin", "win32") or "forkserver" not in methods:
            name = "spawn"
        else:
            name = "forkserver"
    return multiprocessing.get_context(name)


@contextlib.contextmanager
def outputs_in_order(
    evaluate: Callable,
    items: Iterable,
    *,
    n_workers: int,
    context,
    serial: Callable,
    initializer: Callable,
    initargs: tuple,
    broken_message: str,
) -> Iterator[Iterator]:
    """Yield an iterator over each item's output, in item order.

    With one worker the items are evaluated lazily in this process by
    ``serial``. Otherwise a pool of ``n_workers`` processes started by
    ``context`` evaluates them with ``evaluate``, a module-level function,
    after ``initializer(*initargs)`` gives each worker the shared state
    once. At most ``4 * n_workers`` items are in the pool at a time, so a
    long design costs this process little memory. Leaving the block early
    (an error, or a raising progress callback) cancels the items not yet
    handed to a worker and returns without waiting for the few that were;
    those finish in the background.
    """
    if n_workers == 1:
        yield map(serial, items)
        return
    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    )
    finished = False
    try:
        yield _in_flight(pool, evaluate, items, window=4 * n_workers)
        finished = True
    except BrokenProcessPool as exc:
        raise BrokenProcessPool(broken_message) from exc
    finally:
        pool.shutdown(wait=finished, cancel_futures=True)


def _in_flight(pool, evaluate, items, window: int) -> Iterator:
    """Each item's output, in order, with at most ``window`` items submitted
    and not yet consumed.

    One item per task: an evaluation dwarfs the IPC, and a run that stops
    early leaves little work behind. The window keeps every worker busy
    while the outputs are consumed in order.
    """
    items = iter(items)
    pending = collections.deque(
        pool.submit(evaluate, item) for item in itertools.islice(items, window)
    )
    while pending:
        future = pending.popleft()
        for item in itertools.islice(items, 1):
            pending.append(pool.submit(evaluate, item))
        yield future.result()
