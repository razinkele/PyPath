"""Physical and biological constants for ecosystem modeling.

This module centralizes magic numbers and constants used throughout PyPath,
improving maintainability and reducing errors from scattered hard-coded values.
"""

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Geospatial constants
KM_PER_DEGREE_LAT = 111.0  # Kilometers per degree of latitude (approximate)
KM_PER_DEGREE_LON_EQUATOR = 111.32  # Kilometers per degree longitude at equator

# ============================================================================
# BIOLOGICAL/ECOLOGICAL CONSTANTS
# ============================================================================

# Von Bertalanffy Growth Function (VBGF) constants
VBGF_D_EXPONENT = 0.66667  # Mass exponent in VBGF (2/3 power law)

# Prey switching and foraging
DEFAULT_PREY_SWITCHING_POWER = 2.0  # Default switching power exponent
MIN_PREY_SWITCHING_POWER = 0.5  # Minimum prey switching power
MAX_PREY_SWITCHING_POWER = 5.0  # Maximum prey switching power

# Vulnerability defaults
DEFAULT_VULNERABILITY = 2.0  # Mixed functional response
MIN_VULNERABILITY = 1.0  # Type II (Holling disk)
MAX_VULNERABILITY_SAFE = 10.0  # Maximum safe vulnerability before instability

# Density dependence
MAX_QQ_SAFE = 5.0  # Maximum safe QQ (density-dependent catchability)

# ============================================================================
# NUMERICAL THRESHOLDS
# ============================================================================

# Biomass thresholds
MIN_BIOMASS_VIABLE = 0.001  # Minimum viable biomass
MIN_BIOMASS_CRASH_THRESHOLD = 0.0001  # Below this is considered crashed
MIN_BIOMASS_RECOVERY_THRESHOLD = 0.01  # Above this is considered recovered

# Ecotrophic Efficiency
MIN_EE = 0.0  # Minimum ecotrophic efficiency
MAX_EE = 1.0  # Maximum ecotrophic efficiency (100% consumption)
MAX_EE_WARNING = 0.95  # Warn if EE exceeds this (overfishing risk)

# Gross Efficiency (GE = P/Q)
MIN_GE_CONSUMER = 0.05  # Minimum realistic GE for consumers
MAX_GE_CONSUMER = 0.50  # Maximum realistic GE for consumers
MIN_QB_PB_RATIO = 2.0  # Minimum QB/PB ratio (GE_max = PB/QB)
MAX_QB_PB_RATIO = 20.0  # Maximum QB/PB ratio (GE_min = PB/QB)

# Diet composition
MIN_DIET_PROPORTION = 0.001  # Minimum diet proportion to include
DIET_SUM_THRESHOLD = 0.9  # Diet should sum to at least this value

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Time steps
MONTHS_PER_YEAR = 12
DEFAULT_TIMESTEP_MONTHS = 1.0  # Monthly timestep
STEPS_PER_YEAR_MONTHLY = 12

# Simulation durations (defaults)
DEFAULT_SIMULATION_MONTHS = 120  # 10 years
DEFAULT_SIMULATION_YEARS = 50  # For longer runs

# ============================================================================
# CONVERGENCE AND TOLERANCE
# ============================================================================

# Numerical tolerance for comparisons
EPSILON = 1e-10  # Small value for floating point comparisons
BALANCE_TOLERANCE = 1e-6  # Tolerance for mass balance convergence

# Integration tolerances
INTEGRATION_RTOL = 1e-5  # Relative tolerance
INTEGRATION_ATOL = 1e-8  # Absolute tolerance

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Bayesian optimization defaults
DEFAULT_OPTIMIZATION_ITERATIONS = 30
MIN_OPTIMIZATION_ITERATIONS = 10
MAX_OPTIMIZATION_ITERATIONS = 100

DEFAULT_OPTIMIZATION_INIT_POINTS = 10
MIN_OPTIMIZATION_INIT_POINTS = 5
MAX_OPTIMIZATION_INIT_POINTS = 20

# ============================================================================
# SPATIAL (ECOSPACE) CONSTANTS
# ============================================================================

# Grid parameters
DEFAULT_GRID_ROWS = 10
DEFAULT_GRID_COLS = 10
MAX_PATCHES_WARNING = 1000  # Warn if grid exceeds this
MAX_PATCHES_PERFORMANCE = 500  # Use optimized rendering above this

# Hexagon parameters (spatial)
MIN_HEXAGON_SIZE_KM = 0.25
MAX_HEXAGON_SIZE_KM = 3.0
DEFAULT_HEXAGON_SIZE_KM = 1.0

# Dispersal defaults
DEFAULT_DISPERSAL_RATE = 0.1  # Proportion dispersing per timestep
MAX_DISPERSAL_RATE = 5.0  # Maximum dispersal rate

# ============================================================================
# DISPLAY/UI CONSTANTS
# ============================================================================

# Sentinel values for missing data
NO_DATA_VALUE = 9999
NO_DATA_VALUE_NEGATIVE = -9999

# Decimal precision for display
DISPLAY_DECIMAL_PLACES = 3

# Trophic level thresholds
TL_PRODUCER = 1.0  # Trophic level for primary producers
TL_CONSUMER_THRESHOLD = 2.5  # Below this: consumer, above: top predator

# ============================================================================
# PARAMETER BOUNDS FOR VALIDATION
# ============================================================================

# P/B (Production/Biomass) bounds
MIN_PB = 0.0
MAX_PB_CONSUMER = 100.0  # Default for consumers
MAX_PB_PRODUCER = 250.0  # Higher limit for phytoplankton/producers

# Q/B (Consumption/Biomass) bounds
MIN_QB = 0.0
MAX_QB = 1000.0

# Biomass bounds
MIN_BIOMASS = 0.0
MAX_BIOMASS = 1e6

# ============================================================================
# DIET REWIRING CONSTANTS
# ============================================================================

# Diet rewiring defaults
DEFAULT_DIET_UPDATE_INTERVAL_MONTHS = 12  # Update diet annually
MIN_DIET_COEFFICIENT = 0.1  # Minimum diet coefficient
MAX_DIET_COEFFICIENT = 5.0  # Maximum diet coefficient
DEFAULT_SWITCHING_POWER_REWIRING = 2.0  # Switching power for rewiring

# ============================================================================
# FORCING AND ENVIRONMENTAL DRIVERS
# ============================================================================

# Seasonal forcing defaults
DEFAULT_SEASONAL_BASELINE = 15.0  # Baseline temperature (°C)
MAX_SEASONAL_AMPLITUDE = 2.0  # Maximum amplitude multiplier

# Pulse forcing defaults
MIN_PULSE_STRENGTH = 0.5
MAX_PULSE_STRENGTH = 5.0
DEFAULT_PULSE_STRENGTH = 2.5

# ============================================================================
# FILE/DATABASE CONSTANTS
# ============================================================================

# Subprocess timeouts
SUBPROCESS_TIMEOUT_SECONDS = 30  # Timeout for external commands

# Database file extensions
VALID_DB_EXTENSIONS = [".ewemdb", ".mdb", ".accdb"]
