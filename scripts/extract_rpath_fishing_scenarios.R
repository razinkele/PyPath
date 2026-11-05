# Rpath fishing-scenario references for PyPath (REcosystem, 50 years).
#
# Writes biomass_{doubled,zero}_fishing{,_ab}.csv into <output_dir>/ecosim:
# monthly out_Biomass with effort x2 / x0 for every fleet, for RK4 and AB.
# Effort must be changed in sim$fishing$ForcedEffort; rsim.run never reads
# sim$forcing$ForcedEffort (the earlier references set that and came out flat).
#
# Usage: Rscript scripts/extract_rpath_fishing_scenarios.R \
#          packages/pypath/tests/data/rpath_reference

suppressMessages({
  library(Rpath)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[1] else "packages/pypath/tests/data/rpath_reference"
dir.create(file.path(output_dir, "ecosim"), recursive = TRUE, showWarnings = FALSE)

REco <- rpath(REco.params, eco.name = "R Ecosystem")
base <- rsim.scenario(REco, REco.params, years = 1:50)

scenarios <- list(doubled = 2, zero = 0)
methods <- c(RK4 = "", AB = "_ab")

for (sc in names(scenarios)) {
  for (meth in names(methods)) {
    sim <- base
    sim$fishing$ForcedEffort <- sim$fishing$ForcedEffort * scenarios[[sc]]
    run <- rsim.run(sim, method = meth, years = 1:50)
    bio <- as.data.frame(run$out_Biomass)
    colnames(bio) <- sim$params$spname[seq_len(ncol(bio))]
    # Rpath row m is the state at the end of month m
    out <- cbind(Month = seq_len(nrow(bio)), bio)
    file <- file.path(output_dir, "ecosim", sprintf("biomass_%s_fishing%s.csv", sc, methods[[meth]]))
    write.csv(out, file, row.names = FALSE)

    # Guard against another silently flat reference
    rel_range <- apply(run$out_Biomass[, -1], 2, function(x) (max(x) - min(x)) / mean(x))
    if (max(rel_range, na.rm = TRUE) < 0.01) {
      stop(sprintf("%s %s: biomass did not respond to the effort change", sc, meth))
    }
    cat(sprintf("%-8s %-4s -> %s (max relative range %.3g)\n",
                sc, meth, basename(file), max(rel_range, na.rm = TRUE)))
  }
}

cat("Rpath", as.character(packageVersion("Rpath")), "\n")
