"""PyPath Application Configuration.

Centralized configuration constants to eliminate magic values scattered throughout the codebase.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class DisplayConfig:
    """Display and formatting configuration."""

    no_data_value: int = 9999
    decimal_places: int = 3
    table_max_rows: int = 100
    date_format: str = '%Y-%m-%d'

    # Group type labels
    type_labels: Dict[int, str] = None

    def __post_init__(self):
        """Initialize type labels dictionary."""
        if self.type_labels is None:
            self.type_labels = {
                0: 'Consumer',
                1: 'Producer',
                2: 'Detritus',
                3: 'Fleet'
            }


@dataclass
class PlotConfig:
    """Matplotlib plot configuration."""

    default_width: int = 8
    default_height: int = 5
    dpi: int = 100
    style: str = 'seaborn-v0_8-darkgrid'

    # Fallback styles if preferred not available
    fallback_styles: list = None

    def __post_init__(self):
        """Initialize fallback styles."""
        if self.fallback_styles is None:
            self.fallback_styles = [
                'seaborn-v0_8-darkgrid',
                'seaborn-darkgrid',
                'default'
            ]


@dataclass
class ColorScheme:
    """Color scheme for visualizations."""

    # Group type colors
    producer: str = '#2ecc71'      # Green
    consumer: str = '#3498db'      # Blue
    top_predator: str = '#e74c3c'  # Red
    detritus: str = '#95a5a6'      # Gray
    fleet: str = '#f39c12'         # Orange

    # Spatial visualization colors
    boundary: str = '#ff0000'      # Red
    grid: str = 'steelblue'
    grid_fill: str = 'lightblue'

    # Plot series colors (for time series, etc.)
    series_primary: str = '#1D3557'    # Dark blue
    series_secondary: str = '#E63946'  # Red
    series_tertiary: str = '#2A9D8F'   # Teal

    # Status colors
    success: str = '#28a745'
    warning: str = '#ffc107'
    error: str = '#dc3545'
    info: str = '#17a2b8'


@dataclass
class ModelDefaults:
    """Default parameter values for ecosystem models."""

    # Ecopath defaults
    unassim_consumers: float = 0.2
    unassim_producers: float = 0.0
    ba_consumers: float = 0.0      # Biomass accumulation
    ba_producers: float = 0.0
    gs_consumers: float = 2.0       # Growth scalar for multi-stanza

    # Ecosim defaults
    default_months: int = 120       # 10 years
    default_years: int = 50         # For UI sliders
    timestep: float = 1.0           # Monthly timestep
    default_vulnerability: float = 2.0  # Mixed functional response

    # Diet rewiring defaults
    min_dc: float = 0.1             # Minimum diet coefficient
    max_dc: float = 5.0             # Maximum diet coefficient
    switching_power: float = 2.0     # Switching power exponent (also used in forcing_demo)
    diet_update_interval: int = 12  # Months between diet updates
    min_diet_proportion: float = 0.001  # Minimum proportion in diet


@dataclass
class SpatialConfig:
    """Configuration for spatial (ECOSPACE) features."""

    # Grid parameters
    default_rows: int = 10
    default_cols: int = 10
    max_patches_warning: int = 1000  # Warn if grid exceeds this
    max_patches_performance: int = 500  # Switch to optimized rendering above this

    # Hexagon parameters
    min_hexagon_size_km: float = 0.25
    max_hexagon_size_km: float = 3.0
    default_hexagon_size_km: float = 1.0

    # Map visualization
    default_zoom: int = 8
    default_tile_layer: str = 'OpenStreetMap'

    # Performance thresholds
    large_grid_threshold: int = 500  # Patches - use simplified rendering
    huge_grid_threshold: int = 1000  # Patches - show warning


@dataclass
class ValidationConfig:
    """Validation rules and constraints."""

    # Valid group types
    valid_group_types: set = None

    # Parameter ranges
    min_biomass: float = 0.0
    max_biomass: float = 1e6

    min_pb: float = 0.0
    max_pb: float = 100.0  # Default for consumers
    max_pb_producer: float = 250.0  # Higher limit for phytoplankton/producers

    min_qb: float = 0.0
    max_qb: float = 1000.0

    min_ee: float = 0.0
    max_ee: float = 1.0

    min_ge: float = 0.0
    max_ge: float = 1.0

    def __post_init__(self):
        """Initialize validation sets."""
        if self.valid_group_types is None:
            self.valid_group_types = frozenset({0, 1, 2, 3})


@dataclass
class UIConfig:
    """User interface layout and styling constants."""

    # Sidebar dimensions
    sidebar_width: int = 300  # Shiny expects integer pixels
    sidebar_min_width: int = 250

    # Plot heights
    plot_height_small_px: str = "400px"
    plot_height_medium_px: str = "500px"
    plot_height_large_px: str = "600px"

    # DataGrid dimensions
    datagrid_height_default_px: str = "300px"
    datagrid_height_tall_px: str = "500px"

    # Input controls
    textarea_rows_default: int = 8
    textarea_rows_large: int = 12

    # Column widths (proportional)
    col_width_narrow: int = 4
    col_width_medium: int = 6
    col_width_wide: int = 8

    # CSS values
    font_size_small_px: str = "8px"
    font_size_normal_px: str = "10px"
    font_size_large_px: str = "12px"
    border_radius_default_px: str = "5px"
    padding_default_px: str = "10px"

    # Icon sizes
    icon_height_px: str = "32px"

    # Table column constraints
    table_col_min_width_px: str = "180px"
    table_col_max_width_px: str = "250px"


@dataclass
class ThresholdsConfig:
    """Numerical thresholds for simulations and balancing."""

    # Ecosim stability thresholds
    vv_cap: float = 5.0  # Vulnerability cap for autofix
    qq_cap: float = 3.0  # Consumption cap for autofix
    min_biomass: float = 0.001  # Minimum viable biomass
    crash_threshold: float = 0.0001  # Below this = crash
    recovery_threshold: float = 0.01  # Above this = recovered

    # Diet proportions
    min_diet_proportion_range_min: float = 0.0001
    min_diet_proportion_range_default: float = 0.001
    min_diet_proportion_range_max: float = 0.1

    # Normalization ranges
    normalization_min: float = 0.0
    normalization_max: float = 3.0
    normalization_default: float = 1.0
    normalization_step: float = 0.1

    # Minimum multipliers
    minimum_effort_multiplier: float = 0.01

    # Analysis offsets
    log_offset_small: float = 0.001  # Prevent log(0)

    # Ecopath type threshold
    type_threshold_consumer_toppred: float = 2.5  # TL < 2.5 = consumer

    # Sentinel values
    negative_no_data_value: int = -9999


@dataclass
class ParameterRangesConfig:
    """Parameter ranges for UI sliders and inputs."""

    # Simulation time ranges
    years_min: int = 1
    years_max: int = 500
    years_default: int = 50

    # Vulnerability ranges
    vulnerability_min: int = 1
    vulnerability_max: int = 100
    vulnerability_default: int = 2

    # Switching power ranges
    switching_power_min: float = 1.0
    switching_power_max: float = 5.0
    switching_power_default: float = 2.5

    # Rewiring interval ranges
    rewiring_interval_min: int = 1
    rewiring_interval_max: int = 24
    rewiring_interval_default: int = 12

    # Multi-stanza ranges
    stanzas_min: int = 1
    stanzas_max: int = 10
    vbgf_k_min: float = 0.01
    vbgf_k_max: float = 2.0
    vbgf_k_default: float = 0.5
    asymptotic_length_min: int = 1
    asymptotic_length_max: int = 500
    asymptotic_length_default: int = 100
    t0_min: float = -5.0
    t0_max: float = 5.0
    length_weight_a_min: float = 0.0001
    length_weight_a_max: float = 1.0
    length_weight_b_min: float = 1.0
    length_weight_b_max: float = 5.0

    # Effort change rates
    effort_change_min: int = 0
    effort_change_max: int = 50
    effort_change_default: int = 5

    # Optimization parameters
    optimization_iterations_min: int = 10
    optimization_iterations_max: int = 100
    optimization_iterations_default: int = 30
    optimization_iterations_step: int = 5
    optimization_init_points_min: int = 5
    optimization_init_points_max: int = 20
    optimization_init_points_default: int = 10

    # Biodata input ranges
    biomass_input_min: float = 0.001
    biomass_input_step: float = 0.5

    # Ecospace parameters
    dispersal_rate_max: float = 5.0
    default_center_lat: float = 55.0
    default_center_lon: float = 20.0

    # Demo forcing ranges
    seasonal_amplitude_max: float = 2.0
    seasonal_baseline_default: float = 15.0
    pulse_strength_min: float = 0.5
    pulse_strength_max: float = 5.0
    pulse_strength_default: float = 2.5


# Singleton instances - import these in other modules
DISPLAY = DisplayConfig()
PLOTS = PlotConfig()
COLORS = ColorScheme()
DEFAULTS = ModelDefaults()
SPATIAL = SpatialConfig()
VALIDATION = ValidationConfig()
UI = UIConfig()
THRESHOLDS = ThresholdsConfig()
PARAM_RANGES = ParameterRangesConfig()


# Convenience exports
TYPE_LABELS = DISPLAY.type_labels
NO_DATA_VALUE = DISPLAY.no_data_value
VALID_GROUP_TYPES = VALIDATION.valid_group_types
