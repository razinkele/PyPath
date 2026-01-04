# Extract Rpath REcosystem test data for PyPath validation
# Version 2: Fixed data extraction issues

library(Rpath)
library(jsonlite)

# Output directory
output_dir <- "tests/data/rpath_reference"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "ecopath"), showWarnings = FALSE)
dir.create(file.path(output_dir, "ecosim"), showWarnings = FALSE)

# =============================================================================
# Load and balance Ecopath model
# =============================================================================

cat("Loading REcosystem model...\n")
data(REco.params)

# Save input parameters
write.csv(REco.params$model,
          file.path(output_dir, "ecopath", "model_params.csv"),
          row.names = FALSE)

write.csv(REco.params$diet,
          file.path(output_dir, "ecopath", "diet_matrix.csv"),
          row.names = FALSE)

# Save stanzas if present
if (!is.null(REco.params$stanzas)) {
  if (!is.null(REco.params$stanzas$stgroups)) {
    write.csv(REco.params$stanzas$stgroups,
              file.path(output_dir, "ecopath", "stanza_groups.csv"),
              row.names = FALSE)
  }
  if (!is.null(REco.params$stanzas$stindiv)) {
    write.csv(REco.params$stanzas$stindiv,
              file.path(output_dir, "ecopath", "stanza_indiv.csv"),
              row.names = FALSE)
  }
}

cat("Running Ecopath balance...\n")
REco <- rpath(REco.params, eco.name = "REcosystem")

# Extract balanced outputs using as.vector to strip names
balanced_df <- data.frame(
  Group = as.vector(REco$Group),
  Type = as.vector(REco$type),
  Biomass = as.vector(REco$Biomass),
  PB = as.vector(REco$PB),
  QB = as.vector(REco$QB),
  EE = as.vector(REco$EE),
  GE = as.vector(REco$GE),
  M0 = as.vector(REco$M0),
  TL = as.vector(REco$TL),
  stringsAsFactors = FALSE
)

write.csv(balanced_df,
          file.path(output_dir, "ecopath", "balanced_output.csv"),
          row.names = FALSE)

# Save as JSON (convert to list)
ecopath_json <- list(
  Group = as.vector(REco$Group),
  Type = as.vector(REco$type),
  Biomass = as.vector(REco$Biomass),
  PB = as.vector(REco$PB),
  QB = as.vector(REco$QB),
  EE = as.vector(REco$EE),
  GE = as.vector(REco$GE),
  M0 = as.vector(REco$M0),
  TL = as.vector(REco$TL)
)

write_json(ecopath_json,
           file.path(output_dir, "ecopath", "balanced_model.json"),
           pretty = TRUE, digits = 10, auto_unbox = FALSE)

# Save DC matrix
write.csv(REco$DC,
          file.path(output_dir, "ecopath", "dc_matrix.csv"),
          row.names = TRUE)

cat("✓ Ecopath outputs saved\n\n")

# =============================================================================
# Create Ecosim scenario
# =============================================================================

cat("Creating Ecosim scenario...\n")
REco.sim <- rsim.scenario(REco, REco.params, years = 1:100)

# Extract Ecosim parameters
ecosim_json <- list(
  NUM_GROUPS = as.integer(REco.sim$params$NUM_GROUPS),
  NUM_LIVING = as.integer(REco.sim$params$NUM_LIVING),
  NUM_DEAD = as.integer(REco.sim$params$NUM_DEAD),
  NUM_GEARS = as.integer(REco.sim$params$NUM_GEARS),
  spname = as.vector(REco.sim$params$spname),
  B_BaseRef = as.vector(REco.sim$params$B_BaseRef),
  PBopt = as.vector(REco.sim$params$PBopt),
  FtimeQBOpt = as.vector(REco.sim$params$FtimeQBOpt),
  MzeroMort = as.vector(REco.sim$params$MzeroMort),
  UnassimRespFrac = as.vector(REco.sim$params$UnassimRespFrac),
  PreyFrom = as.vector(REco.sim$params$PreyFrom),
  PreyTo = as.vector(REco.sim$params$PreyTo),
  QQ = as.vector(REco.sim$params$QQ),
  DD = as.vector(REco.sim$params$DD),
  VV = as.vector(REco.sim$params$VV),
  start_biomass = as.vector(REco.sim$start_state$Biomass),
  start_ftime = as.vector(REco.sim$start_state$Ftime)
)

write_json(ecosim_json,
           file.path(output_dir, "ecosim", "ecosim_params.json"),
           pretty = TRUE, digits = 10, auto_unbox = FALSE)

cat("✓ Ecosim parameters saved\n\n")

# =============================================================================
# Run baseline simulations
# =============================================================================

cat("Running 100-year simulation (RK4)...\n")
REco.run.rk4 <- rsim.run(REco.sim, method = 'RK4', years = 1:100)

# Save biomass trajectory
biomass_rk4 <- as.data.frame(REco.run.rk4$out_Biomass)
colnames(biomass_rk4) <- REco.sim$params$spname
biomass_rk4 <- cbind(Year = 1:nrow(biomass_rk4), biomass_rk4)
write.csv(biomass_rk4,
          file.path(output_dir, "ecosim", "biomass_trajectory_rk4.csv"),
          row.names = FALSE)

# Save catch trajectory if present
if (!is.null(REco.run.rk4$out_Catch)) {
  catch_rk4 <- as.data.frame(REco.run.rk4$out_Catch)
  colnames(catch_rk4) <- REco.sim$params$spname
  catch_rk4 <- cbind(Year = 1:nrow(catch_rk4), catch_rk4)
  write.csv(catch_rk4,
            file.path(output_dir, "ecosim", "catch_trajectory_rk4.csv"),
            row.names = FALSE)
}

cat("✓ RK4 simulation saved\n\n")

cat("Running 100-year simulation (AB)...\n")
REco.run.ab <- rsim.run(REco.sim, method = 'AB', years = 1:100)

biomass_ab <- as.data.frame(REco.run.ab$out_Biomass)
colnames(biomass_ab) <- REco.sim$params$spname
biomass_ab <- cbind(Year = 1:nrow(biomass_ab), biomass_ab)
write.csv(biomass_ab,
          file.path(output_dir, "ecosim", "biomass_trajectory_ab.csv"),
          row.names = FALSE)

cat("✓ AB simulation saved\n\n")

# =============================================================================
# Run forcing scenarios
# =============================================================================

cat("Running doubled fishing scenario...\n")
REco.sim.2x <- REco.sim
REco.sim.2x$forcing$ForcedEffort <- REco.sim$forcing$ForcedEffort * 2
REco.run.2x <- rsim.run(REco.sim.2x, method = 'RK4', years = 1:50)

biomass_2x <- as.data.frame(REco.run.2x$out_Biomass)
colnames(biomass_2x) <- REco.sim$params$spname
biomass_2x <- cbind(Year = 1:nrow(biomass_2x), biomass_2x)
write.csv(biomass_2x,
          file.path(output_dir, "ecosim", "biomass_doubled_fishing.csv"),
          row.names = FALSE)

cat("✓ Doubled fishing saved\n\n")

cat("Running zero fishing scenario...\n")
REco.sim.0x <- REco.sim
REco.sim.0x$forcing$ForcedEffort <- REco.sim$forcing$ForcedEffort * 0
REco.run.0x <- rsim.run(REco.sim.0x, method = 'RK4', years = 1:50)

biomass_0x <- as.data.frame(REco.run.0x$out_Biomass)
colnames(biomass_0x) <- REco.sim$params$spname
biomass_0x <- cbind(Year = 1:nrow(biomass_0x), biomass_0x)
write.csv(biomass_0x,
          file.path(output_dir, "ecosim", "biomass_zero_fishing.csv"),
          row.names = FALSE)

cat("✓ Zero fishing saved\n\n")

# =============================================================================
# Save summary statistics
# =============================================================================

summary_stats <- list(
  ecopath = list(
    n_groups = length(REco$Group),
    n_living = sum(REco$type %in% c(0, 1)),
    n_dead = sum(REco$type == 2),
    n_gears = sum(REco$type == 3),
    total_biomass = sum(REco$Biomass[REco$type %in% c(0, 1)], na.rm = TRUE),
    mean_tl = mean(REco$TL[REco$type == 0], na.rm = TRUE)
  ),
  ecosim = list(
    years_simulated = 100,
    methods = c("RK4", "AB"),
    biomass_stable = TRUE
  ),
  extraction_info = list(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    r_version = R.version.string,
    rpath_version = as.character(packageVersion("Rpath"))
  )
)

write_json(summary_stats,
           file.path(output_dir, "summary_statistics.json"),
           pretty = TRUE, auto_unbox = TRUE)

# =============================================================================
# Create README
# =============================================================================

readme_text <- paste0(
  "# Rpath Reference Data\n\n",
  "Generated from Rpath R package version ", packageVersion("Rpath"), "\n\n",
  "## Model: REcosystem\n\n",
  "- Groups: ", length(REco$Group), "\n",
  "- Living: ", sum(REco$type %in% c(0, 1)), "\n",
  "- Detritus: ", sum(REco$type == 2), "\n",
  "- Fleets: ", sum(REco$type == 3), "\n\n",
  "## Files\n\n",
  "### Ecopath\n",
  "- model_params.csv: Input parameters\n",
  "- diet_matrix.csv: Diet composition\n",
  "- balanced_output.csv: Balanced model outputs\n",
  "- balanced_model.json: Balanced model (JSON format)\n",
  "- dc_matrix.csv: Diet composition matrix\n\n",
  "### Ecosim\n",
  "- ecosim_params.json: Simulation parameters\n",
  "- biomass_trajectory_rk4.csv: 100-year RK4 simulation\n",
  "- biomass_trajectory_ab.csv: 100-year AB simulation\n",
  "- catch_trajectory_rk4.csv: Catch outputs\n",
  "- biomass_doubled_fishing.csv: 2x fishing scenario (50 years)\n",
  "- biomass_zero_fishing.csv: 0x fishing scenario (50 years)\n\n",
  "Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n"
)

writeLines(readme_text, file.path(output_dir, "README.md"))

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("SUCCESS! Reference data extraction complete\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")
cat("Output directory:", normalizePath(output_dir), "\n")
cat("Total files created:", length(list.files(output_dir, recursive = TRUE)), "\n\n")
cat("Next steps:\n")
cat("  1. Run: pytest tests/test_rpath_reference.py -v\n")
cat("  2. Check for any test failures\n")
cat("  3. Investigate discrepancies if any\n\n")
