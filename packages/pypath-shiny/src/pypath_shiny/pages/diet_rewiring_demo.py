"""
Prey Switching Demonstration Page

Illustrates EwE's prey switching (the per-predator ``SwitchPower``) on one
predator with three prey. The numbers come from the core's
``apply_prey_switching``, the function ``rsim_run`` uses, so the page shows
the formula a model run applies. (The module keeps its old name because the
navigation id and the app wiring refer to it.)
"""

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from pypath.core.ecosim_deriv import apply_prey_switching
from shiny import Inputs, Outputs, Session, render, ui

PREY_NAMES = ("Prey 1", "Prey 2", "Prey 3")
BASE_DIET = (0.5, 0.3, 0.2)
VULNERABILITY = 2.0
SWITCH_RANGE = (0.0, 2.0)  # EwE's SwitchPower range


def switching_diet(
    b_rel: Sequence[float],
    power: float,
    base_diet: Sequence[float] = BASE_DIET,
    vulnerability: float = VULNERABILITY,
) -> np.ndarray:
    """Diet proportions of one predator over prey with biomass ``b_rel`` (B / B0).

    Builds a system of an Outside row, the prey (rows 1..n, base biomass 1) and
    the predator (last column, base biomass 1) whose base flows are
    ``base_diet``. ``apply_prey_switching`` gives the effective vulnerability
    and base flow; consumption is then the closed-form kernel at PDY = 1,
    ``Q = Q0' * b_rel``, normalised to proportions.
    """
    q_prey = np.asarray(base_diet, dtype=float)
    b_rel = np.asarray(b_rel, dtype=float)
    n_prey = len(q_prey)
    n = n_prey + 2  # Outside, prey, predator
    pred = n - 1
    q0 = np.zeros((n, n))
    q0[1 : 1 + n_prey, pred] = q_prey
    vv = np.full((n, n), float(vulnerability))
    biomass = np.zeros(n)
    biomass[1 : 1 + n_prey] = b_rel
    b_base = np.ones(n)
    b_base[0] = 0.0
    _, q_eff = apply_prey_switching(biomass, vv, q0, float(power), b_base)
    q = q_eff[1 : 1 + n_prey, pred] * b_rel
    return q / q.sum()


def build_prey_switching_code(power: float) -> str:
    """Build the example script shown on the prey switching demo page.

    A plain function (no reactive reads) so the code renderer and the
    download handler can share it.
    """
    return f"""# Prey switching (EwE's SwitchPower) example
# Generated from PyPath Demo

from pypath.core.ecosim import rsim_run
from pypath.examples import example_ecosim_scenario

scenario = example_ecosim_scenario("Central_Baltic")

# Set the switching power of one predator (0 = off; EwE's range is 0 to 2).
predator = "Ad. Cod"  # a consumer group of this model
names = [str(n).strip() for n in scenario.params.spname]
scenario.params.SwitchPower[names.index(predator)] = {power}

result = rsim_run(scenario, years=range(1, 2))
print("Final biomass of", predator, result.out_Biomass[-1, names.index(predator)])
"""


def diet_rewiring_demo_ui():
    """UI for the prey switching demonstration page."""
    return ui.div(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Prey Switching (SwitchPower)"),
                ui.input_slider(
                    "demo_switching_power",
                    "Switching power",
                    min=SWITCH_RANGE[0],
                    max=SWITCH_RANGE[1],
                    value=1.0,
                    step=0.1,
                ),
                ui.input_radio_buttons(
                    "demo_varied_prey",
                    "Prey whose abundance varies",
                    choices=dict(enumerate(PREY_NAMES)),
                    selected=0,
                ),
                width=300,
            ),
            ui.div(
                ui.h3("Prey Switching"),
                ui.p(
                    "Illustrates EwE's switching formula on one predator with "
                    "three prey; in a model, set a predator's switching power "
                    "on the Ecosim page."
                ),
                ui.output_ui("demo_switching_plot"),
                ui.hr(),
                ui.h5("Example code"),
                ui.output_code("diet_code_example"),
                ui.download_button("diet_download_code", "Download example"),
            ),
        ),
    )


def diet_rewiring_demo_server(input: Inputs, output: Outputs, session: Session):
    """Server logic for the prey switching demonstration."""

    @output
    @render.ui
    def demo_switching_plot():
        power = float(input.demo_switching_power())
        varied = int(input.demo_varied_prey())
        abundance = np.logspace(-1, 1, 81)
        fig = go.Figure()
        for p, name, dash in (
            (power, f"Power {power:.1f}", "solid"),
            (0.0, "Power 0", "dash"),
        ):
            share = []
            for a in abundance:
                b = np.ones(len(BASE_DIET))
                b[varied] = a
                share.append(switching_diet(b, p)[varied])
            fig.add_trace(
                go.Scatter(
                    x=abundance,
                    y=np.array(share) * 100,
                    mode="lines",
                    name=name,
                    line=dict(dash=dash),
                )
            )
        fig.update_xaxes(
            type="log", title_text=f"{PREY_NAMES[varied]} relative abundance (B / B0)"
        )
        fig.update_yaxes(title_text=f"{PREY_NAMES[varied]} share of diet (%)")
        fig.update_layout(height=450, template="plotly_white", hovermode="x unified")
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.code
    def diet_code_example():
        return build_prey_switching_code(input.demo_switching_power())

    @render.download(filename="prey_switching_example.py")
    def diet_download_code():
        """Download code example (yield: a returned str is treated as a path)."""
        yield build_prey_switching_code(input.demo_switching_power())
