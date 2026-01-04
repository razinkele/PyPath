# Simplified extraction with debugging

library(Rpath)
library(jsonlite)

output_dir <- "tests/data/rpath_reference"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "ecopath"), showWarnings = FALSE)
dir.create(file.path(output_dir, "ecosim"), showWarnings = FALSE)

cat("Loading data...\n")
data(REco.params)

cat("Saving input files...\n")
write.csv(REco.params$model,
          file.path(output_dir, "ecopath", "model_params.csv"),
          row.names = FALSE)
write.csv(REco.params$diet,
          file.path(output_dir, "ecopath", "diet_matrix.csv"),
          row.names = FALSE)

cat("Running rpath()...\n")
REco <- rpath(REco.params, eco.name = "REcosystem")

cat("Creating data.frame field by field...\n")

# Start with just one field
df <- data.frame(
  Group = as.vector(REco$Group),
  stringsAsFactors = FALSE
)
cat("  Group: OK (", nrow(df), "rows )\n")

# Add Type
df$Type <- as.vector(REco$type)
cat("  Type: OK\n")

# Add Biomass
df$Biomass <- as.vector(REco$Biomass)
cat("  Biomass: OK\n")

# Add PB
df$PB <- as.vector(REco$PB)
cat("  PB: OK\n")

# Add QB
df$QB <- as.vector(REco$QB)
cat("  QB: OK\n")

# Add EE
df$EE <- as.vector(REco$EE)
cat("  EE: OK\n")

# Add GE
df$GE <- as.vector(REco$GE)
cat("  GE: OK\n")

# Add M0 - calculate if not available in REco
cat("  Checking M0...\n")
if (!is.null(REco$M0) && length(REco$M0) > 0) {
  cat("    Using REco$M0\n")
  df$M0 <- as.vector(REco$M0)
} else {
  cat("    REco$M0 is NULL, calculating from PB * (1 - EE)\n")
  df$M0 <- as.vector(REco$PB) * (1 - as.vector(REco$EE))
}
cat("  M0: OK\n")

# Add TL
df$TL <- as.vector(REco$TL)
cat("  TL: OK\n")

cat("\nFinal dataframe:\n")
cat("  Rows:", nrow(df), "\n")
cat("  Cols:", ncol(df), "\n")

# Save it
write.csv(df,
          file.path(output_dir, "ecopath", "balanced_output.csv"),
          row.names = FALSE)
cat("\n✓ Saved balanced_output.csv\n")

# Save as JSON - calculate M0 if not available
m0_values <- if (!is.null(REco$M0) && length(REco$M0) > 0) {
  as.vector(REco$M0)
} else {
  as.vector(REco$PB) * (1 - as.vector(REco$EE))
}

json_list <- list(
  Group = as.vector(REco$Group),
  Type = as.vector(REco$type),
  Biomass = as.vector(REco$Biomass),
  PB = as.vector(REco$PB),
  QB = as.vector(REco$QB),
  EE = as.vector(REco$EE),
  GE = as.vector(REco$GE),
  M0 = m0_values,
  TL = as.vector(REco$TL)
)

write_json(json_list,
           file.path(output_dir, "ecopath", "balanced_model.json"),
           pretty = TRUE, digits = 10, auto_unbox = FALSE)
cat("✓ Saved balanced_model.json\n")

# Save DC matrix
write.csv(REco$DC,
          file.path(output_dir, "ecopath", "dc_matrix.csv"),
          row.names = TRUE)
cat("✓ Saved dc_matrix.csv\n\n")

# Now do Ecosim
cat("Creating Ecosim scenario...\n")
REco.sim <- rsim.scenario(REco, REco.params, years = 1:100)
cat("✓ Scenario created\n")

# Save ecosim params as JSON
ecosim_list <- list(
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

write_json(ecosim_list,
           file.path(output_dir, "ecosim", "ecosim_params.json"),
           pretty = TRUE, digits = 10, auto_unbox = FALSE)
cat("✓ Saved ecosim_params.json\n\n")

# Run simulations
cat("Running RK4 simulation (100 years)...\n")
REco.run.rk4 <- rsim.run(REco.sim, method = 'RK4', years = 1:100)

biomass_rk4 <- as.data.frame(REco.run.rk4$out_Biomass)
# Use only living + dead group names (exclude gears)
n_cols <- ncol(biomass_rk4)
colnames(biomass_rk4) <- REco.sim$params$spname[1:n_cols]
biomass_rk4 <- cbind(Year = 1:nrow(biomass_rk4), biomass_rk4)
write.csv(biomass_rk4,
          file.path(output_dir, "ecosim", "biomass_trajectory_rk4.csv"),
          row.names = FALSE)
cat("✓ Saved biomass_trajectory_rk4.csv\n")

if (!is.null(REco.run.rk4$out_Catch)) {
  catch_rk4 <- as.data.frame(REco.run.rk4$out_Catch)
  n_cols_catch <- ncol(catch_rk4)
  colnames(catch_rk4) <- REco.sim$params$spname[1:n_cols_catch]
  catch_rk4 <- cbind(Year = 1:nrow(catch_rk4), catch_rk4)
  write.csv(catch_rk4,
            file.path(output_dir, "ecosim", "catch_trajectory_rk4.csv"),
            row.names = FALSE)
  cat("✓ Saved catch_trajectory_rk4.csv\n")
}

cat("\nRunning AB simulation (100 years)...\n")
REco.run.ab <- rsim.run(REco.sim, method = 'AB', years = 1:100)

biomass_ab <- as.data.frame(REco.run.ab$out_Biomass)
n_cols_ab <- ncol(biomass_ab)
colnames(biomass_ab) <- REco.sim$params$spname[1:n_cols_ab]
biomass_ab <- cbind(Year = 1:nrow(biomass_ab), biomass_ab)
write.csv(biomass_ab,
          file.path(output_dir, "ecosim", "biomass_trajectory_ab.csv"),
          row.names = FALSE)
cat("✓ Saved biomass_trajectory_ab.csv\n\n")

# Forcing scenarios
cat("Running doubled fishing scenario (50 years)...\n")
REco.sim.2x <- REco.sim
REco.sim.2x$forcing$ForcedEffort <- REco.sim$forcing$ForcedEffort * 2
REco.run.2x <- rsim.run(REco.sim.2x, method = 'RK4', years = 1:50)

biomass_2x <- as.data.frame(REco.run.2x$out_Biomass)
n_cols_2x <- ncol(biomass_2x)
colnames(biomass_2x) <- REco.sim$params$spname[1:n_cols_2x]
biomass_2x <- cbind(Year = 1:nrow(biomass_2x), biomass_2x)
write.csv(biomass_2x,
          file.path(output_dir, "ecosim", "biomass_doubled_fishing.csv"),
          row.names = FALSE)
cat("✓ Saved biomass_doubled_fishing.csv\n")

cat("\nRunning zero fishing scenario (50 years)...\n")
REco.sim.0x <- REco.sim
REco.sim.0x$forcing$ForcedEffort <- REco.sim$forcing$ForcedEffort * 0
REco.run.0x <- rsim.run(REco.sim.0x, method = 'RK4', years = 1:50)

biomass_0x <- as.data.frame(REco.run.0x$out_Biomass)
n_cols_0x <- ncol(biomass_0x)
colnames(biomass_0x) <- REco.sim$params$spname[1:n_cols_0x]
biomass_0x <- cbind(Year = 1:nrow(biomass_0x), biomass_0x)
write.csv(biomass_0x,
          file.path(output_dir, "ecosim", "biomass_zero_fishing.csv"),
          row.names = FALSE)
cat("✓ Saved biomass_zero_fishing.csv\n\n")

# Summary
cat(paste(rep("=", 70), collapse=""), "\n")
cat("SUCCESS! Reference data extraction complete\n")
cat(paste(rep("=", 70), collapse=""), "\n")
cat("Output: ", normalizePath(output_dir), "\n")
cat("Files created: ", length(list.files(output_dir, recursive = TRUE)), "\n\n")
