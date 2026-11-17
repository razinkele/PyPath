"""Re-balancing keeps prey-switching overrides; any other model change clears them."""

import ast
import inspect
from pathlib import Path

from pypath_shiny.pages import ecopath as ecopath_page
from pypath_shiny.pages import ecosim as ecosim_page
from tests.test_ecosim_continuous_ui import _ecosim_tree, _find_function

APP_PATH = Path(__file__).parents[1] / "src" / "pypath_shiny" / "app.py"


def _is_call(node, dotted):
    return isinstance(node, ast.Call) and ast.unparse(node.func) == dotted


def _is_mark_true(stmt):
    if isinstance(stmt, ast.If):  # `if rebalance_mark is not None:` guard
        return len(stmt.body) == 1 and _is_mark_true(stmt.body[0])
    return (
        isinstance(stmt, ast.Expr)
        and _is_call(stmt.value, "rebalance_mark.set")
        and ast.unparse(stmt.value.args[0]) == "True"
    )


def _is_model_data_set(stmt):
    return isinstance(stmt, ast.Expr) and _is_call(stmt.value, "model_data.set")


def _blocks(tree):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def _ecopath_tree():
    return ast.parse(Path(ecopath_page.__file__).read_text(encoding="utf-8"))


def test_servers_take_an_optional_rebalance_mark():
    for server in (ecopath_page.ecopath_server, ecosim_page.ecosim_server):
        param = inspect.signature(server).parameters["rebalance_mark"]
        assert param.default is None


def test_only_the_rebalance_sets_the_mark_before_model_data():
    tree = _ecopath_tree()
    marked, all_sets = [], 0
    for block in _blocks(tree):
        for i, stmt in enumerate(block):
            if _is_model_data_set(stmt):
                all_sets += 1
                if i and _is_mark_true(block[i - 1]):
                    marked.append(block)
    assert all_sets >= 2
    assert len(marked) == 1
    block = marked[0]
    assert any(
        isinstance(s, ast.Assign) and "rpath(" in ast.unparse(s.value) for s in block
    )
    mark_sets = [n for n in ast.walk(tree) if _is_call(n, "rebalance_mark.set")]
    assert len(mark_sets) == 1


def test_retire_handler_consumes_the_mark_and_applies_the_store_rule():
    fn = _find_function(_ecosim_tree(), "_retire_run_on_model_change")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    resets = [
        c
        for c in calls
        if _is_call(c, "rebalance_mark.set")
        and isinstance(c.args[0], ast.Constant)
        and c.args[0].value is False
    ]
    assert resets
    assert any(_is_call(c, "on_model_change") for c in calls)
    notices = [
        c
        for c in calls
        if _is_call(c, "notify_info") and ast.unparse(c.args[0]) == "switching_note"
    ]
    assert notices
    # Independent of the `if note:` block.
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "note":
            assert not any(
                _is_call(n, "notify_info")
                and ast.unparse(n.args[0]) == "switching_note"
                for n in ast.walk(node)
            )
    last = fn.body[-1]
    assert (
        ast.unparse(last) == "switching_generation.set(switching_generation.get() + 1)"
    )


def test_app_passes_the_mark_to_both_servers():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "ecopath.ecopath_server",
            "ecosim.ecosim_server",
        ):
            seen[ast.unparse(node.func)] = {
                k.arg: ast.unparse(k.value) for k in node.keywords
            }
    assert len(seen) == 2
    for kw in seen.values():
        assert kw.get("rebalance_mark") == "rebalance_mark"
    assert "rebalance_mark = reactive.Value(False)" in APP_PATH.read_text()
