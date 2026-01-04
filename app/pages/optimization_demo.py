"""
Bayesian Optimization Demonstration Page

Interactive demonstration of automated parameter calibration using Gaussian Processes.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import Inputs, Outputs, Session, reactive, render, ui

# Configuration imports
try:
    from app.config import PARAM_RANGES
except ModuleNotFoundError:
    from config import PARAM_RANGES


def optimization_demo_ui():
    """UI for Bayesian optimization demonstration page."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Optimization Setup"),
                ui.input_select(
                    "param_type",
                    "Parameter to Optimize",
                    choices={
                        "vulnerabilities": "Vulnerabilities",
                        "search_rates": "Search Rates",
                        "Q0": "Feeding Time (Q0)",
                        "mortality": "Mortality Rates",
                    },
                    selected="vulnerabilities",
                ),
                ui.input_select(
                    "objective",
                    "Objective Function",
                    choices={
                        "rmse": "RMSE - Root Mean Square Error",
                        "nrmse": "NRMSE - Normalized RMSE",
                        "mape": "MAPE - Mean Absolute % Error",
                        "mae": "MAE - Mean Absolute Error",
                        "loglik": "Log-Likelihood",
                    },
                    selected="nrmse",
                ),
                ui.input_slider(
                    "n_iterations",
                    "Number of Iterations",
                    min=PARAM_RANGES.optimization_iterations_min,
                    max=PARAM_RANGES.optimization_iterations_max,
                    value=PARAM_RANGES.optimization_iterations_default,
                    step=PARAM_RANGES.optimization_iterations_step,
                ),
                ui.input_slider(
                    "n_initial",
                    "Initial Random Points",
                    min=PARAM_RANGES.optimization_init_points_min,
                    max=PARAM_RANGES.optimization_init_points_max,
                    value=PARAM_RANGES.optimization_init_points_default,
                    step=1,
                ),
                ui.input_select(
                    "acquisition",
                    "Acquisition Function",
                    choices={
                        "EI": "Expected Improvement",
                        "UCB": "Upper Confidence Bound",
                        "PI": "Probability of Improvement",
                    },
                    selected="EI",
                ),
                ui.hr(),
                ui.input_action_button(
                    "opt_run_demo", "Run Demo Optimization", class_="btn-primary w-100"
                ),
                ui.input_action_button(
                    "generate_data",
                    "Generate Synthetic Data",
                    class_="btn-secondary w-100 mt-2",
                ),
                width=300,
            ),
            # Main content
            ui.navset_tab(
                ui.nav_panel(
                    "Optimization Progress",
                    ui.card(
                        ui.card_header("Convergence Plot"),
                        ui.output_ui("convergence_plot"),
                        ui.output_text_verbatim("optimization_summary"),
                    ),
                ),
                ui.nav_panel(
                    "Parameter Space",
                    ui.card(
                        ui.card_header("Gaussian Process Model"),
                        ui.output_ui("gp_plot"),
                        ui.markdown(
                            """
                        **Gaussian Process Visualization:**

                        - **Black dots**: Evaluated points
                        - **Red star**: Best point found
                        - **Blue line**: GP mean prediction
                        - **Shaded area**: 95% confidence interval
                        """
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Results Comparison",
                    ui.card(
                        ui.card_header("Optimized vs Observed"),
                        ui.output_ui("opt_comparison_plot"),
                        ui.output_data_frame("results_table"),
                    ),
                ),
                ui.nav_panel(
                    "Code Example",
                    ui.card(
                        ui.card_header("Python Code"),
                        ui.output_code("opt_code_example"),
                        ui.download_button(
                            "opt_download_code", "Download Code", class_="mt-2"
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Help",
                    ui.card(
                        ui.card_header(
                            ui.tags.i(class_="bi bi-graph-up me-2"),
                            "Bayesian Optimization Guide",
                        ),
                        ui.markdown(
                            """
                        ## What is Bayesian Optimization?

                        Bayesian optimization is an **efficient method for finding optimal parameters**
                        when each evaluation is expensive. Perfect for ecosystem models!

                        ### Why Bayesian Optimization?

                        **Traditional Methods:**
                        - Grid search: Try all combinations (very slow)
                        - Random search: Try random points (inefficient)
                        - Gradient descent: Requires derivatives (not available)

                        **Bayesian Optimization:**
                        - ✅ Learns from previous evaluations
                        - ✅ Focuses search on promising regions
                        - ✅ Handles expensive evaluations
                        - ✅ No derivatives needed
                        - ✅ Quantifies uncertainty

                        ### How It Works

                        1. **Build Surrogate Model** (Gaussian Process)
                           - Learns relationship between parameters and objective
                           - Provides mean prediction and uncertainty

                        2. **Acquisition Function**
                           - Decides where to sample next
                           - Balances exploration vs exploitation

                        3. **Iterate**
                           - Evaluate at selected point
                           - Update surrogate model
                           - Repeat until convergence

                        ### Parameters You Can Optimize

                        #### 1. Vulnerabilities
                        **What they control**: Predation efficiency
                        - Low (1.0): Hard to catch (prey refuge)
                        - High (3.0+): Easy to catch (no refuge)

                        **Use when**: Matching predator-prey dynamics

                        #### 2. Search Rates
                        **What they control**: Predator search efficiency
                        - Controls encounter rates
                        - Affects foraging arena dynamics

                        **Use when**: Calibrating consumption rates

                        #### 3. Feeding Time (Q0)
                        **What they control**: Time spent feeding vs other activities
                        - Range: 0.1 - 0.9
                        - Higher = more feeding time

                        **Use when**: Matching Q/B ratios

                        #### 4. Mortality Rates
                        **What they control**: Natural mortality (M)
                        - Background death rate
                        - Not from predation or fishing

                        **Use when**: Matching abundance trends

                        ### Objective Functions

                        #### RMSE (Root Mean Square Error)
                        ```
                        RMSE = sqrt(mean((observed - predicted)^2))
                        ```
                        - **Use when**: Want absolute error minimization
                        - **Units**: Same as data
                        - **Best for**: Similar-scale data

                        #### NRMSE (Normalized RMSE)
                        ```
                        NRMSE = RMSE / mean(observed) * 100
                        ```
                        - **Use when**: Comparing different variables
                        - **Units**: Percentage
                        - **Best for**: Multi-group optimization

                        #### MAPE (Mean Absolute Percentage Error)
                        ```
                        MAPE = mean(|observed - predicted| / observed) * 100
                        ```
                        - **Use when**: Percentage error matters
                        - **Units**: Percentage
                        - **Best for**: Relative errors

                        #### MAE (Mean Absolute Error)
                        ```
                        MAE = mean(|observed - predicted|)
                        ```
                        - **Use when**: Robust to outliers
                        - **Units**: Same as data
                        - **Best for**: Skewed distributions

                        #### Log-Likelihood
                        ```
                        LL = -sum(log(2*pi*sigma^2) + (obs-pred)^2/(2*sigma^2))
                        ```
                        - **Use when**: Statistical inference needed
                        - **Units**: Log probability
                        - **Best for**: Probabilistic models

                        ### Acquisition Functions

                        #### Expected Improvement (EI)
                        - **What it does**: Maximizes expected improvement over best
                        - **Behavior**: Balanced exploration-exploitation
                        - **Best for**: General purpose

                        #### Upper Confidence Bound (UCB)
                        - **What it does**: Optimistic estimate (mean + uncertainty)
                        - **Behavior**: More exploration
                        - **Best for**: Uncertain landscapes

                        #### Probability of Improvement (PI)
                        - **What it does**: Maximizes probability of improvement
                        - **Behavior**: More exploitation
                        - **Best for**: Near-optimal solutions

                        ### Example Workflow

                        ```python
                        from pypath.core.optimization import bayesian_optimize_ecosim

                        # 1. Set up optimization
                        result = bayesian_optimize_ecosim(
                            model=model,
                            params=params,
                            observed_data=observed_biomass,
                            param_config=[
                                {
                                    'param': 'vulnerabilities',
                                    'bounds': (1.0, 3.0),
                                    'groups': [0, 1, 2, 3]
                                }
                            ],
                            n_iterations=50,
                            n_initial=10,
                            objective='nrmse',
                            acquisition='EI'
                        )

                        # 2. Get results
                        best_params = result['best_params']
                        best_score = result['best_score']
                        convergence = result['convergence']

                        # 3. Apply to model
                        for group_idx, value in best_params.items():
                            params.vulnerabilities[group_idx] = value
                        ```

                        ### Tips for Success

                        ✅ **DO:**
                        - Start with 10-20 initial points
                        - Run 30-50 iterations minimum
                        - Use NRMSE for multi-group problems
                        - Check convergence plots
                        - Validate with held-out data

                        ⚠️ **DON'T:**
                        - Optimize too many parameters at once (curse of dimensionality)
                        - Use too few iterations (won't converge)
                        - Ignore biological constraints
                        - Over-fit to short time series

                        ### Computational Cost

                        - **Single iteration**: 1 full Ecosim simulation
                        - **50 iterations**: ~5-10 minutes (typical)
                        - **Parallelization**: Supported for initial points
                        - **Caching**: Previous results reused

                        ### When to Use

                        **Good for:**
                        - Parameter calibration
                        - Uncertainty quantification
                        - Sensitivity analysis
                        - Multi-objective optimization

                        **Not for:**
                        - Very high-dimensional problems (>10 parameters)
                        - Very cheap evaluations (use grid search)
                        - Discrete-only parameters
                        - Real-time applications

                        ### Advanced Features

                        #### Multi-Parameter Optimization
                        ```python
                        param_config=[
                            {'param': 'vulnerabilities', 'bounds': (1.0, 3.0), 'groups': [0,1]},
                            {'param': 'search_rates', 'bounds': (0.1, 1.0), 'groups': [0,1]},
                            {'param': 'Q0', 'bounds': (0.2, 0.8), 'groups': [2,3]}
                        ]
                        ```

                        #### Custom Constraints
                        ```python
                        # Ensure total Q/B stays within bounds
                        def constraint(params_dict):
                            total_qb = sum(params_dict.values())
                            return 10.0 <= total_qb <= 30.0
                        ```

                        #### Progress Tracking
                        ```python
                        result = bayesian_optimize_ecosim(
                            ...,
                            verbose=True,  # Print progress
                            log_file='optimization.log'  # Save log
                        )
                        ```

                        ## Scientific Background

                        **Key Papers:**
                        - Mockus (1974): Original Bayesian optimization
                        - Snoek et al. (2012): Practical implementation
                        - Frazier (2018): Tutorial review

                        **Applications in Ecology:**
                        - Ecosystem model calibration
                        - Species distribution models
                        - Population dynamics
                        - Resource management
                        """
                        ),
                    ),
                ),
            ),
        )
    )


def optimization_demo_server(input: Inputs, output: Outputs, session: Session):
    """Server logic for optimization demonstration."""

    # Reactive values
    optimization_results = reactive.Value(None)
    synthetic_data = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.generate_data)
    def generate_synthetic_data():
        """Generate synthetic observed data for demonstration."""
        # Generate synthetic time series
        years = np.arange(2000, 2021)
        n_years = len(years)

        # True underlying parameter (hidden)
        true_param = 2.2

        # Generate biomass with this parameter
        # Simple model: exponential decline with parameter
        baseline = 20.0
        biomass = baseline * np.exp(-true_param * 0.05 * np.arange(n_years))

        # Add noise
        noise = np.random.normal(0, 0.5, n_years)
        biomass = biomass + noise

        df = pd.DataFrame({"Year": years, "Observed_Biomass": biomass})

        synthetic_data.set(df)

    @reactive.effect
    @reactive.event(input.opt_run_demo)
    def run_optimization():
        """Run demonstration optimization."""
        # Generate data if not already generated
        if synthetic_data() is None:
            # Generate synthetic data inline
            years = np.arange(2000, 2021)
            n_years = len(years)
            true_param = 2.2
            baseline = 20.0
            biomass = baseline * np.exp(-true_param * 0.05 * np.arange(n_years))
            noise = np.random.normal(0, 0.5, n_years)
            biomass = biomass + noise
            df = pd.DataFrame({"Year": years, "Observed_Biomass": biomass})
            synthetic_data.set(df)

        n_iterations = input.n_iterations()
        n_initial = input.n_initial()

        # Simulate Bayesian optimization process
        # (This is a simplified demo - real implementation uses actual Ecosim)

        np.random.seed(42)

        # True optimum (unknown to optimizer)
        true_optimum = 2.2

        # Parameter space
        param_min, param_max = 1.0, 3.0

        # Initial random points
        X = np.random.uniform(param_min, param_max, n_initial)

        # Objective function (simplified)
        def objective(param):
            data = synthetic_data()
            years = np.arange(len(data))
            predicted = 20.0 * np.exp(-param * 0.05 * years)

            if input.objective() == "rmse":
                return np.sqrt(np.mean((data["Observed_Biomass"] - predicted) ** 2))
            elif input.objective() == "nrmse":
                rmse = np.sqrt(np.mean((data["Observed_Biomass"] - predicted) ** 2))
                return (rmse / np.mean(data["Observed_Biomass"])) * 100
            elif input.objective() == "mape":
                return (
                    np.mean(
                        np.abs(
                            (data["Observed_Biomass"] - predicted)
                            / data["Observed_Biomass"]
                        )
                    )
                    * 100
                )
            elif input.objective() == "mae":
                return np.mean(np.abs(data["Observed_Biomass"] - predicted))
            else:  # loglik
                residuals = data["Observed_Biomass"] - predicted
                sigma = np.std(residuals)
                return -np.sum(
                    -0.5 * np.log(2 * np.pi * sigma**2) - residuals**2 / (2 * sigma**2)
                )

        # Evaluate initial points
        y = np.array([objective(x) for x in X])

        # Bayesian optimization iterations
        for i in range(n_iterations - n_initial):
            # Simple acquisition: choose point with lowest GP mean + exploration bonus
            # (Real implementation uses proper acquisition functions)

            # Find best so far
            best_y = np.min(y)
            best_idx = np.argmin(y)

            # Propose new point (simplified - real uses GP + acquisition)
            # Gradually focus near best point
            exploration = max(0.5 - i / n_iterations, 0.1)
            new_x = X[best_idx] + np.random.normal(0, exploration)
            new_x = np.clip(new_x, param_min, param_max)

            X = np.append(X, new_x)
            new_y = objective(new_x)
            y = np.append(y, new_y)

        # Store results
        results = {
            "X": X,
            "y": y,
            "best_x": X[np.argmin(y)],
            "best_y": np.min(y),
            "true_optimum": true_optimum,
            "convergence": [np.min(y[: i + 1]) for i in range(len(y))],
        }

        optimization_results.set(results)

    @output
    @render.ui
    def convergence_plot():
        """Plot optimization convergence."""
        results = optimization_results()
        if results is None:
            return ui.div(
                ui.tags.p(
                    "Click 'Run Demo Optimization' to start",
                    class_="text-muted text-center p-5",
                )
            )

        convergence = results["convergence"]
        iterations = np.arange(1, len(convergence) + 1)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=convergence,
                mode="lines+markers",
                name="Best Score",
                line=dict(color="#E63946", width=2),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Best Objective Value",
            template="plotly_white",
            height=400,
            showlegend=True,
            hovermode="x unified",
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.text
    def optimization_summary():
        """Display optimization summary."""
        results = optimization_results()
        if results is None:
            return ""

        summary = f"""
Optimization Results:
--------------------
Best Parameter Value: {results["best_x"]:.4f}
Best Objective Score: {results["best_y"]:.4f}
True Optimum: {results["true_optimum"]:.4f}
Error: {abs(results["best_x"] - results["true_optimum"]):.4f}

Total Evaluations: {len(results["X"])}
Initial Points: {input.n_initial()}
Optimization Steps: {input.n_iterations() - input.n_initial()}

Acquisition Function: {input.acquisition()}
Objective Function: {input.objective().upper()}
        """
        return summary

    @output
    @render.ui
    def gp_plot():
        """Plot Gaussian Process model."""
        results = optimization_results()
        if results is None:
            return ui.div(
                ui.tags.p("Run optimization first", class_="text-muted text-center p-5")
            )

        # Create dense grid for plotting
        x_plot = np.linspace(1.0, 3.0, 200)

        # Simplified GP visualization (real would use actual GP predictions)
        # Show evaluated points and trend
        X = results["X"]
        y = results["y"]

        fig = go.Figure()

        # Evaluated points
        fig.add_trace(
            go.Scatter(
                x=X,
                y=y,
                mode="markers",
                name="Evaluated Points",
                marker=dict(color="black", size=8),
            )
        )

        # Best point
        fig.add_trace(
            go.Scatter(
                x=[results["best_x"]],
                y=[results["best_y"]],
                mode="markers",
                name="Best Point",
                marker=dict(color="red", size=15, symbol="star"),
            )
        )

        # True optimum (for demo)
        fig.add_vline(
            x=results["true_optimum"],
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text="True Optimum",
        )

        fig.update_layout(
            xaxis_title="Parameter Value",
            yaxis_title="Objective Value",
            template="plotly_white",
            height=400,
            showlegend=True,
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.ui
    def opt_comparison_plot():
        """Plot observed vs optimized."""
        results = optimization_results()
        data = synthetic_data()

        if results is None or data is None:
            return ui.div(
                ui.tags.p("Run optimization first", class_="text-muted text-center p-5")
            )

        best_param = results["best_x"]
        years = np.arange(len(data))
        predicted = 20.0 * np.exp(-best_param * 0.05 * years)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Observed_Biomass"],
                mode="markers",
                name="Observed",
                marker=dict(color="black", size=8),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["Year"],
                y=predicted,
                mode="lines",
                name="Optimized Model",
                line=dict(color="#E63946", width=3),
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Biomass",
            template="plotly_white",
            height=400,
            showlegend=True,
            hovermode="x unified",
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.data_frame
    def results_table():
        """Display results comparison table."""
        results = optimization_results()
        data = synthetic_data()

        if results is None or data is None:
            return pd.DataFrame({"Message": ["Run optimization first"]})

        best_param = results["best_x"]
        years = np.arange(len(data))
        predicted = 20.0 * np.exp(-best_param * 0.05 * years)

        df = pd.DataFrame(
            {
                "Year": data["Year"],
                "Observed": data["Observed_Biomass"].round(2),
                "Predicted": predicted.round(2),
                "Error": (data["Observed_Biomass"] - predicted).round(2),
                "Error_%": (
                    (data["Observed_Biomass"] - predicted)
                    / data["Observed_Biomass"]
                    * 100
                ).round(1),
            }
        )

        return render.DataGrid(df, width="100%", height="400px")

    @output
    @render.code
    def opt_code_example():
        """Generate Python code example."""
        param_type = input.param_type()
        objective = input.objective()
        n_iterations = input.n_iterations()
        acquisition = input.acquisition()

        code = f"""# Bayesian Optimization Example
# Generated from PyPath Demo

from pypath.core.optimization import bayesian_optimize_ecosim

# Set up optimization
result = bayesian_optimize_ecosim(
    model=model,
    params=params,
    observed_data=observed_biomass,
    param_config=[
        {{
            'param': '{param_type}',
            'bounds': (1.0, 3.0),
            'groups': [0, 1, 2, 3]
        }}
    ],
    n_iterations={n_iterations},
    n_initial=10,
    objective='{objective}',
    acquisition='{acquisition}',
    verbose=True
)

# Get best parameters
best_params = result['best_params']
best_score = result['best_score']

print(f"Best parameters: {{best_params}}")
print(f"Best score: {{best_score:.4f}}")

# Apply optimized parameters to model
for group_idx, value in best_params.items():
    params.{param_type}[group_idx] = value

# Run simulation with optimized parameters
optimized_scenario = rsim_scenario(model, params)
optimized_result = rsim_run(optimized_scenario)

# Plot convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(result['convergence'])
plt.xlabel('Iteration')
plt.ylabel('Best Objective Value')
plt.title('Optimization Convergence')
plt.grid(True)
plt.show()
"""
        return code

    @render.download(filename="optimization_example.py")
    def opt_download_code():
        """Download code example."""
        code = opt_code_example()
        return code
