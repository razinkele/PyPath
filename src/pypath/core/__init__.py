"""
Core module for PyPath.

Contains the main Ecopath and Ecosim implementations.
"""

from pypath.core.params import (
    RpathParams,
    create_rpath_params,
    read_rpath_params,
    write_rpath_params,
    check_rpath_params,
)
from pypath.core.ecopath import Rpath, rpath
from pypath.core.ecosim import (
    RsimParams,
    RsimState,
    RsimForcing,
    RsimFishing,
    RsimScenario,
    RsimOutput,
    rsim_params,
    rsim_state,
    rsim_forcing,
    rsim_fishing,
    rsim_scenario,
    rsim_run,
)
from pypath.core.stanzas import (
    StanzaGroup,
    StanzaIndividual,
    StanzaParams,
    RsimStanzas,
    von_bertalanffy_weight,
    von_bertalanffy_consumption,
    calculate_survival,
    rpath_stanzas,
    rsim_stanzas,
    split_update,
    split_set_pred,
    create_stanza_params,
)
from pypath.core.adjustments import (
    adjust_fishing,
    adjust_forcing,
    adjust_scenario,
    set_vulnerability,
    set_handling_time,
    adjust_group_parameter,
    create_fishing_ramp,
    create_pulse_forcing,
    create_seasonal_forcing,
)
from pypath.core.ecosim_deriv import (
    deriv_vector,
    integrate_rk4,
    integrate_ab,
    run_ecosim,
    prey_switching,
    mediation_function,
    primary_production_forcing,
)
from pypath.core.analysis import (
    mixed_trophic_impacts,
    keystoneness_index,
    calculate_network_indices,
    NetworkIndices,
    summarize_ecosim_output,
    compare_scenarios,
    check_ecopath_balance,
    check_ecosim_stability,
    export_ecopath_to_dataframe,
    export_ecosim_to_dataframe,
)
from pypath.core.autofix import (
    AutofixResult,
    diagnose_crash_causes,
    autofix_parameters,
    validate_and_fix_scenario,
)

# Optimization (optional - requires scikit-optimize)
try:
    from pypath.core.optimization import (
        EcosimOptimizer,
        OptimizationResult,
        mean_squared_error,
        mean_absolute_percentage_error,
        normalized_root_mean_squared_error,
        log_likelihood,
        plot_optimization_results,
        plot_fit,
    )

    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False
from pypath.core.plotting import (
    plot_foodweb,
    plot_biomass,
    plot_catch,
    plot_biomass_grid,
    plot_trophic_spectrum,
    plot_mti_heatmap,
    plot_ecosim_summary,
    save_plots,
    HAS_NETWORKX,
    HAS_PLOTLY,
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
