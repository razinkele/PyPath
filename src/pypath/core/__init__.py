"""
Core module for PyPath.

Contains the main Ecopath and Ecosim implementations.
"""

from pypath.core.adjustments import (
    adjust_fishing,
    adjust_forcing,
    adjust_group_parameter,
    adjust_scenario,
    create_fishing_ramp,
    create_pulse_forcing,
    create_seasonal_forcing,
    set_handling_time,
    set_vulnerability,
)
from pypath.core.analysis import (
    NetworkIndices,
    calculate_network_indices,
    check_ecopath_balance,
    check_ecosim_stability,
    compare_scenarios,
    export_ecopath_to_dataframe,
    export_ecosim_to_dataframe,
    keystoneness_index,
    mixed_trophic_impacts,
    summarize_ecosim_output,
)
from pypath.core.autofix import (
    AutofixResult,
    autofix_parameters,
    diagnose_crash_causes,
    validate_and_fix_scenario,
)
from pypath.core.ecopath import Rpath, rpath
from pypath.core.ecosim import (
    RsimFishing,
    RsimForcing,
    RsimOutput,
    RsimParams,
    RsimScenario,
    RsimState,
    rsim_fishing,
    rsim_forcing,
    rsim_params,
    rsim_run,
    rsim_scenario,
    rsim_state,
)
from pypath.core.ecosim_deriv import (
    deriv_vector,
    integrate_ab,
    integrate_rk4,
    mediation_function,
    prey_switching,
    primary_production_forcing,
    run_ecosim,
)
from pypath.core.params import (
    RpathParams,
    check_rpath_params,
    create_rpath_params,
    read_rpath_params,
    write_rpath_params,
)
from pypath.core.stanzas import (
    RsimStanzas,
    StanzaGroup,
    StanzaIndividual,
    StanzaParams,
    calculate_survival,
    create_stanza_params,
    rpath_stanzas,
    rsim_stanzas,
    split_set_pred,
    split_update,
    von_bertalanffy_consumption,
    von_bertalanffy_weight,
)

# Optimization (optional - requires scikit-optimize)
try:
    from pypath.core.optimization import (
        HAS_SKOPT,
        EcosimOptimizer,
        OptimizationResult,
        log_likelihood,
        mean_absolute_percentage_error,
        mean_squared_error,
        normalized_root_mean_squared_error,
        plot_fit,
        plot_optimization_results,
    )

    # Reflect actual availability of scikit-optimize
    HAS_OPTIMIZATION = bool(HAS_SKOPT)
except ImportError:
    HAS_OPTIMIZATION = False
from pypath.core.plotting import (
    HAS_NETWORKX,
    HAS_PLOTLY,
    plot_biomass,
    plot_biomass_grid,
    plot_catch,
    plot_ecosim_summary,
    plot_foodweb,
    plot_mti_heatmap,
    plot_trophic_spectrum,
    save_plots,
)

__all__ = [
    # Ecopath
    "RpathParams",
    "create_rpath_params",
    "read_rpath_params",
    "write_rpath_params",
    "check_rpath_params",
    "Rpath",
    "rpath",
    # Ecosim
    "RsimParams",
    "RsimState",
    "RsimForcing",
    "RsimFishing",
    "RsimScenario",
    "RsimOutput",
    "rsim_params",
    "rsim_state",
    "rsim_forcing",
    "rsim_fishing",
    "rsim_scenario",
    "rsim_run",
    # Stanzas
    "StanzaGroup",
    "StanzaIndividual",
    "StanzaParams",
    "RsimStanzas",
    "von_bertalanffy_weight",
    "von_bertalanffy_consumption",
    "calculate_survival",
    "rpath_stanzas",
    "rsim_stanzas",
    "split_update",
    "split_set_pred",
    "create_stanza_params",
    # Adjustments
    "adjust_fishing",
    "adjust_forcing",
    "adjust_scenario",
    "set_vulnerability",
    "set_handling_time",
    "adjust_group_parameter",
    "create_fishing_ramp",
    "create_pulse_forcing",
    "create_seasonal_forcing",
    # Derivatives
    "deriv_vector",
    "integrate_rk4",
    "integrate_ab",
    "run_ecosim",
    "prey_switching",
    "mediation_function",
    "primary_production_forcing",
    # Analysis
    "mixed_trophic_impacts",
    "keystoneness_index",
    "calculate_network_indices",
    "NetworkIndices",
    "summarize_ecosim_output",
    "compare_scenarios",
    "check_ecopath_balance",
    "check_ecosim_stability",
    "export_ecopath_to_dataframe",
    "export_ecosim_to_dataframe",
    # Autofix
    "AutofixResult",
    "diagnose_crash_causes",
    "autofix_parameters",
    "validate_and_fix_scenario",
    # Optimization
    "HAS_OPTIMIZATION",
    "EcosimOptimizer",
    "OptimizationResult",
    "mean_squared_error",
    "mean_absolute_percentage_error",
    "normalized_root_mean_squared_error",
    "log_likelihood",
    "plot_optimization_results",
    "plot_fit",
    # Plotting
    "plot_foodweb",
    "plot_biomass",
    "plot_catch",
    "plot_biomass_grid",
    "plot_trophic_spectrum",
    "plot_mti_heatmap",
    "plot_ecosim_summary",
    "save_plots",
    "HAS_NETWORKX",
    "HAS_PLOTLY",
]
