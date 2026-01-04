# Extract Rpath REcosystem test data for PyPath validation
# This script extracts the REcosystem model from Rpath R package
# and saves reference outputs for testing PyPath conversion

# Install Rpath if needed
if (!require("Rpath", quietly = TRUE)) {
  install.packages("Rpath")
}

library(Rpath)
library(jsonlite)

# Output directory for reference data
output_dir <- "tests/data/rpath_reference"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Load REcosystem model from Rpath
# =============================================================================

# REcosystem is the standard test model in Rpath
# It's available in the package data
data(REco.params)

# Create output directory structure
dir.create(file.path(output_dir, "ecopath"), showWarnings = FALSE)
dir.create(file.path(output_dir, "ecosim"), showWarnings = FALSE)

# =============================================================================
# Save Ecopath Parameters
# =============================================================================

# Save model parameters
write.csv(REco.params$model,
          file.path(output_dir, "ecopath", "model_params.csv"),
          row.names = FALSE)

# Save diet matrix
write.csv(REco.params$diet,
          file.path(output_dir, "ecopath", "diet_matrix.csv"),
          row.names = FALSE)

# Save stanza parameters if present
if (!is.null(REco.params$stanzas)) {
  # Save stanzas groups
  if (!is.null(REco.params$stanzas$stgroups)) {
    write.csv(REco.params$stanzas$stgroups,
              file.path(output_dir, "ecopath", "stanza_groups.csv"),
              row.names = FALSE)
  }

  # Save stanza individuals
  if (!is.null(REco.params$stanzas$stindiv)) {
    write.csv(REco.params$stanzas$stindiv,
              file.path(output_dir, "ecopath", "stanza_indiv.csv"),
              row.names = FALSE)
  }
}

# Save pedigree if present
if (!is.null(REco.params$pedigree)) {
  write.csv(REco.params$pedigree,
            file.path(output_dir, "ecopath", "pedigree.csv"),
            row.names = FALSE)
}

# =============================================================================
# Run Ecopath Balance
# =============================================================================

cat("Running Ecopath balance...\n")
REco <- rpath(REco.params, eco.name = "REcosystem")

# Extract balanced model outputs
# Convert vectors to lists for JSON export (unname to remove names)
ecopath_output <- list(
  Group = unname(as.character(REco$Group)),
  Type = unname(as.numeric(REco$type)),
  Biomass = unname(as.numeric(REco$Biomass)),
  PB = unname(as.numeric(REco$PB)),
  QB = unname(as.numeric(REco$QB)),
  EE = unname(as.numeric(REco$EE)),
  GE = unname(as.numeric(REco$GE)),
  M0 = unname(as.numeric(REco$M0)),
  TL = unname(as.numeric(REco$TL))
)

# Save as JSON for easy parsing in Python
write_json(ecopath_output,
           file.path(output_dir, "ecopath", "balanced_model.json"),
           pretty = TRUE, digits = 10)

# Also save as CSV
balanced_df <- data.frame(
  Group = unname(as.character(REco$Group)),
  Type = unname(as.numeric(REco$type)),
  Biomass = unname(as.numeric(REco$Biomass)),
  PB = unname(as.numeric(REco$PB)),
  QB = unname(as.numeric(REco$QB)),
  EE = unname(as.numeric(REco$EE)),
  GE = unname(as.numeric(REco$GE)),
  M0 = unname(as.numeric(REco$M0)),
  TL = unname(as.numeric(REco$TL)),
  stringsAsFactors = FALSE
)
write.csv(balanced_df,
          file.path(output_dir, "ecopath", "balanced_output.csv"),
          row.names = FALSE)

# Save DC matrix separately (it's a matrix)
write.csv(REco$DC,
          file.path(output_dir, "ecopath", "dc_matrix.csv"),
          row.names = TRUE)

cat("Ecopath outputs saved.\n")

# =============================================================================
# Create Ecosim Scenario
# =============================================================================

cat("Creating Ecosim scenario...\n")

# Create base Ecosim parameters
REco.sim <- rsim.scenario(REco, REco.params, years = 1:100)

# Extract Ecosim parameters
ecosim_params <- list(
  NUM_GROUPS = as.integer(REco.sim$params$NUM_GROUPS),
  NUM_LIVING = as.integer(REco.sim$params$NUM_LIVING),
  NUM_DEAD = as.integer(REco.sim$params$NUM_DEAD),
  NUM_GEARS = as.integer(REco.sim$params$NUM_GEARS),
  spname = as.character(REco.sim$params$spname),

  # Biomass and rates
  B_BaseRef = as.numeric(REco.sim$params$B_BaseRef),
  PBopt = as.numeric(REco.sim$params$PBopt),
  FtimeQBOpt = as.numeric(REco.sim$params$FtimeQBOpt),
  MzeroMort = as.numeric(REco.sim$params$MzeroMort),
  UnassimRespFrac = as.numeric(REco.sim$params$UnassimRespFrac),

  # Predator-prey links
  PreyFrom = as.integer(REco.sim$params$PreyFrom),
  PreyTo = as.integer(REco.sim$params$PreyTo),
  QQ = as.numeric(REco.sim$params$QQ),
  DD = as.numeric(REco.sim$params$DD),
  VV = as.numeric(REco.sim$params$VV),

  # Initial state
  start_biomass = as.numeric(REco.sim$start_state$Biomass),
  start_ftime = as.numeric(REco.sim$start_state$Ftime)
)

# Save Ecosim parameters
write_json(ecosim_params,
           file.path(output_dir, "ecosim", "ecosim_params.json"),
           pretty = TRUE, digits = 10, auto_unbox = FALSE)

cat("Ecosim parameters saved.\n")

# =============================================================================
# Run Ecosim Simulation (Baseline)
# =============================================================================

cat("Running Ecosim simulation (100 years)...\n")

# Run with RK4 method (more stable)
REco.run.rk4 <- rsim.run(REco.sim, method = 'RK4', years = 1:100)

# Extract biomass trajectory
biomass_trajectory <- as.data.frame(REco.run.rk4$out_Biomass)
colnames(biomass_trajectory) <- REco.sim$params$spname

# Add time column
biomass_trajectory <- cbind(
  Year = 1:nrow(biomass_trajectory),
  biomass_trajectory
)

# Save biomass trajectory
write.csv(biomass_trajectory,
          file.path(output_dir, "ecosim", "biomass_trajectory_rk4.csv"),
          row.names = FALSE)

# Extract catch trajectory if present
if (!is.null(REco.run.rk4$out_Catch)) {
  catch_trajectory <- as.data.frame(REco.run.rk4$out_Catch)
  colnames(catch_trajectory) <- REco.sim$params$spname
  catch_trajectory <- cbind(
    Year = 1:nrow(catch_trajectory),
    catch_trajectory
  )
  write.csv(catch_trajectory,
            file.path(output_dir, "ecosim", "catch_trajectory_rk4.csv"),
            row.names = FALSE)
}

# Run with Adams-Bashforth method for comparison
REco.run.ab <- rsim.run(REco.sim, method = 'AB', years = 1:100)

biomass_trajectory_ab <- as.data.frame(REco.run.ab$out_Biomass)
colnames(biomass_trajectory_ab) <- REco.sim$params$spname
biomass_trajectory_ab <- cbind(
  Year = 1:nrow(biomass_trajectory_ab),
  biomass_trajectory_ab
)

write.csv(biomass_trajectory_ab,
          file.path(output_dir, "ecosim", "biomass_trajectory_ab.csv"),
          row.names = FALSE)

cat("Ecosim simulations saved.\n")

# =============================================================================
# Test Scenarios with Forcing
# =============================================================================

cat("Running test scenarios...\n")

# Scenario 1: Increased fishing effort
REco.sim.fishing <- REco.sim
# Double fishing effort for all gears
REco.sim.fishing$forcing$ForcedEffort <- REco.sim$forcing$ForcedEffort * 2

REco.run.fishing <- rsim.run(REco.sim.fishing, method = 'RK4', years = 1:50)

biomass_fishing <- as.data.frame(REco.run.fishing$out_Biomass)
colnames(biomass_fishing) <- REco.sim$params$spname
biomass_fishing <- cbind(Year = 1:nrow(biomass_fishing), biomass_fishing)

write.csv(biomass_fishing,
          file.path(output_dir, "ecosim", "biomass_doubled_fishing.csv"),
          row.names = FALSE)

# Scenario 2: Zero fishing
REco.sim.nofishing <- REco.sim
# Set all fishing to zero
REco.sim.nofishing$forcing$ForcedEffort <- REco.sim$forcing$ForcedEffort * 0

REco.run.nofishing <- rsim.run(REco.sim.nofishing, method = 'RK4', years = 1:50)

biomass_nofishing <- as.data.frame(REco.run.nofishing$out_Biomass)
colnames(biomass_nofishing) <- REco.sim$params$spname
biomass_nofishing <- cbind(Year = 1:nrow(biomass_nofishing), biomass_nofishing)

write.csv(biomass_nofishing,
          file.path(output_dir, "ecosim", "biomass_zero_fishing.csv"),
          row.names = FALSE)

cat("Test scenarios saved.\n")

# =============================================================================
# Save Summary Statistics
# =============================================================================

summary_stats <- list(
  ecopath = list(
    n_groups = length(REco$Group),
    n_living = sum(REco$type %in% c(0, 1)),
    n_dead = sum(REco$type == 2),
    n_gears = sum(REco$type == 3),
    balanced = TRUE,
    total_biomass = sum(REco$Biomass[REco$type %in% c(0, 1)], na.rm = TRUE),
    mean_tl = mean(REco$TL[REco$type == 0], na.rm = TRUE)
  ),
  ecosim = list(
    years_simulated = 100,
    methods = c("RK4", "AB"),
    final_biomass_rk4 = as.numeric(tail(biomass_trajectory[, -1], 1)),
    final_biomass_ab = as.numeric(tail(biomass_trajectory_ab[, -1], 1)),
    biomass_stable = max(abs(diff(rowSums(biomass_trajectory[, -1], na.rm = TRUE)))) < 1.0
  )
)

write_json(summary_stats,
           file.path(output_dir, "summary_statistics.json"),
           pretty = TRUE, digits = 10)

# =============================================================================
# Create README
# =============================================================================

readme_text <- paste0(
  "# Rpath Reference Data for PyPath Validation\n\n",
  "This directory contains reference data extracted from the Rpath R package.\n\n",
  "## Source\n",
  "- Package: Rpath\n",
  "- Repository: https://github.com/NOAA-EDAB/Rpath\n",
  "- Model: REcosystem (standard test model)\n\n",
  "## Contents\n\n",
  "### ecopath/\n",
  "- `model_params.csv`: Input model parameters\n",
  "- `diet_matrix.csv`: Diet composition matrix\n",
  "- `balanced_model.json`: Complete balanced model output\n",
  "- `balanced_output.csv`: Key balanced parameters (B, PB, QB, EE, GE, M0, TL)\n",
  "- `dc_matrix.csv`: Diet composition matrix (balanced)\n\n",
  "### ecosim/\n",
  "- `ecosim_params.json`: Ecosim simulation parameters\n",
  "- `biomass_trajectory_rk4.csv`: 100-year simulation with RK4\n",
  "- `biomass_trajectory_ab.csv`: 100-year simulation with Adams-Bashforth\n",
  "- `catch_trajectory_rk4.csv`: Catch outputs\n",
  "- `biomass_doubled_fishing.csv`: Scenario with 2x fishing effort\n",
  "- `biomass_zero_fishing.csv`: Scenario with zero fishing\n\n",
  "## Generation\n",
  "Generated by: extract_rpath_data.R\n",
  "Date: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n",
  "R version: ", R.version.string, "\n",
  "Rpath version: ", packageVersion("Rpath"), "\n"
)

writeLines(readme_text, file.path(output_dir, "README.md"))

cat("\n=============================================================================\n")
cat("Reference data extraction complete!\n")
cat("Output directory:", output_dir, "\n")
cat("=============================================================================\n")
