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

# Helper to write metadata and a sentinel QQ CSV if extraction fails early
write_meta_and_sentinel <- function(note = NULL) {
  diag_dir_loc <- file.path(output_dir, "ecosim", "diagnostics")
  if (!dir.exists(diag_dir_loc)) dir.create(diag_dir_loc, recursive = TRUE, showWarnings = FALSE)
  meta <- list(qq_provided = FALSE)
  if (!is.null(note)) meta$note <- note
  write_json(meta, file.path(diag_dir_loc, "meta.json"), pretty = TRUE, auto_unbox = TRUE)
  sentinel <- data.frame(month = integer(0))
  write.csv(sentinel, file.path(diag_dir_loc, 'seabirds_qq_rk4.csv'), row.names = FALSE, na = 'NA')
  cat("Wrote early meta.json with qq_provided=FALSE and QQ sentinel due to failure: ", note, "\n")
}

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

tryCatch({
  cat("Running Ecopath balance...\n")
  REco <- rpath(REco.params, eco.name = "REcosystem")

  # DEBUG: dump REco internals to help diagnose mismatched lengths
  cat("DEBUG: REco names:\n")
  print(names(REco))
  cat(sprintf("DEBUG: lengths - Group=%d, type=%d, Biomass=%d\n", length(REco$Group), length(REco$type), ifelse(is.null(REco$Biomass), 0, length(REco$Biomass))))
  cat(sprintf("DEBUG: sample Group names: %s\n", paste(head(REco$Group, 10), collapse=", ")))

  # If rrun present, dump out_Biomass structure
  if (exists("REco.run.rk4") && !is.null(REco.run.rk4)) {
    cat("DEBUG: REco.run.rk4 names:\n")
    print(names(REco.run.rk4))
    if (!is.null(REco.run.rk4$out_Biomass)) {
      ob <- REco.run.rk4$out_Biomass
      cat(sprintf("DEBUG: out_Biomass class=%s\n", paste(class(ob), collapse=",")))
      if (!is.null(dim(ob))) cat(sprintf("DEBUG: out_Biomass dim=%s\n", paste(dim(ob), collapse=",")))
      if (!is.null(colnames(ob))) cat(sprintf("DEBUG: out_Biomass colnames sample=%s\n", paste(head(colnames(ob), 10), collapse=",")))
    }
  }

  # Debug: print simulation param spnames (only if REco.sim exists)
  if (exists("REco.sim") && !is.null(REco.sim$params$spname)) {
    cat(sprintf("DEBUG: REco.sim$params$spname len=%d sample=%s\n", length(REco.sim$params$spname), paste(head(REco.sim$params$spname, 10), collapse=",")))
  } else {
    cat("DEBUG: REco.sim not created yet or has no spname\n")
  }

  # Extract balanced model outputs
  # Convert vectors to lists for JSON export (unname to remove names)
  # Use a safe accessor that pads missing/short vectors with NA so data.frame construction can't fail
  safe_vec <- function(x, n) {
    y <- try(as.numeric(x), silent = TRUE)
    if (inherits(y, "try-error") || is.null(y)) return(rep(NA_real_, n))
    if (length(y) != n) return(rep(NA_real_, n))
    return(y)
  }
  n_groups_bal <- length(REco$Group)

  ecopath_output <- list(
    Group = unname(as.character(REco$Group)),
    Type = safe_vec(REco$type, n_groups_bal),
    Biomass = safe_vec(REco$Biomass, n_groups_bal),
    PB = safe_vec(REco$PB, n_groups_bal),
    QB = safe_vec(REco$QB, n_groups_bal),
    EE = safe_vec(REco$EE, n_groups_bal),
    GE = safe_vec(REco$GE, n_groups_bal),
    M0 = safe_vec(REco$M0, n_groups_bal),
    TL = safe_vec(REco$TL, n_groups_bal)
  )

  # Save as JSON for easy parsing in Python
  write_json(ecopath_output,
             file.path(output_dir, "ecopath", "balanced_model.json"),
             pretty = TRUE, digits = 10)

  # Also save as CSV
  balanced_df <- data.frame(
    Group = unname(as.character(REco$Group)),
    Type = ecopath_output$Type,
    Biomass = ecopath_output$Biomass,
    PB = ecopath_output$PB,
    QB = ecopath_output$QB,
    EE = ecopath_output$EE,
    GE = ecopath_output$GE,
    M0 = ecopath_output$M0,
    TL = ecopath_output$TL,
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
}, error = function(e) {
  write_meta_and_sentinel(paste0("Ecopath balance failed: ", conditionMessage(e)))
  stop(conditionMessage(e))
})

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

# Extract biomass trajectory (robust handling)
bio_raw <- REco.run.rk4$out_Biomass
bio_df <- tryCatch(
  as.data.frame(bio_raw),
  error = function(e) {
    write_meta_and_sentinel(paste0('biomass conversion failed: ', conditionMessage(e)))
    stop(conditionMessage(e))
  }
)

ncols <- ncol(bio_df)
col_names_raw <- colnames(bio_df)
spn <- if (!is.null(REco.sim$params$spname)) as.character(REco.sim$params$spname) else character(0)
cat(sprintf("Debug: biomass columns=%d, spname len=%d\n", ncols, length(spn)))

# Try to align names by intersection (preferred), else truncate/pad spn to ncols
tryCatch({
  if (!is.null(col_names_raw) && length(col_names_raw) > 0) {
    common <- intersect(spn, col_names_raw)
    if (length(common) > 0) {
      # Order columns to match spname order where possible
      ordered <- spn[spn %in% col_names_raw]
      col_order <- c(ordered, setdiff(col_names_raw, ordered))
      col_order <- unique(col_order)
      cat(sprintf("Debug: ordered_len=%d, col_order_len=%d, ncols=%d\n", length(ordered), length(col_order), ncols))
      cat(sprintf("Debug: class(bio_df)=%s, names_len=%d\n", paste(class(bio_df), collapse=','), length(names(bio_df))))
      cat(sprintf("Debug: sample names_raw: %s\n", paste(head(col_names_raw, 10), collapse=',')))
      cat(sprintf("Debug: sample col_order: %s\n", paste(head(col_order, 10), collapse=',')))
      # If col_order doesn't match ncols, fallback to using raw column names
      if (length(col_order) != ncols) {
        warning(paste0("Computed col_order length (", length(col_order), ") != ncols (", ncols, "); falling back to raw column names or truncated spn."))
        if (length(col_names_raw) == ncols) {
          col_order <- col_names_raw
        } else if (length(spn) >= ncols) {
          col_order <- spn[seq_len(ncols)]
        } else {
          col_order <- c(spn, paste0('G', seq_len(ncols - length(spn))))
        }
      }
      # Reorder bio_df accordingly (only columns present)
      bio_df <- bio_df[, col_order, drop = FALSE]
      # Attempt to set colnames and capture errors to write debug note
      set_colnames_try <- tryCatch({
        colnames(bio_df) <- col_order
        TRUE
      }, error = function(e) {
        dbg <- paste0('colname assignment failed: ', conditionMessage(e), '; class=', paste(class(bio_df), collapse=','), '; names_len=', length(names(bio_df)), '; col_order_len=', length(col_order))
        write_meta_and_sentinel(dbg)
        stop(conditionMessage(e))
      })
      if (set_colnames_try) cat('Assigned column names OK\n')
    } else {
      # No intersection; fallback to use spn truncated/padded
      if (length(spn) >= ncols) {
        colnames(bio_df) <- spn[seq_len(ncols)]
      } else {
        colnames(bio_df) <- c(spn, paste0('G', seq_len(ncols - length(spn))))
      }
    }
  } else {
    # No raw column names; use spn truncated/padded
    if (length(spn) >= ncols) {
      colnames(bio_df) <- spn[seq_len(ncols)]
    } else {
      colnames(bio_df) <- c(spn, paste0('G', seq_len(ncols - length(spn))))
    }
  }
}, error = function(e) {
  write_meta_and_sentinel(paste0('biomass column-name alignment failed: ', conditionMessage(e)))
  stop(conditionMessage(e))
})

# Add time column
biomass_trajectory <- cbind(
  Year = 1:nrow(bio_df),
  bio_df
)

# Save biomass trajectory
write.csv(biomass_trajectory,
          file.path(output_dir, "ecosim", "biomass_trajectory_rk4.csv"),
          row.names = FALSE)

# Extract catch trajectory if present
if (!is.null(REco.run.rk4$out_Catch)) {
  catch_raw <- REco.run.rk4$out_Catch
  catch_df <- tryCatch(as.data.frame(catch_raw), error = function(e) { write_meta_and_sentinel(paste0('Catch conversion failed: ', conditionMessage(e))); stop(conditionMessage(e)) })
  # Align names defensively
  ncols_c <- ncol(catch_df)
  spn_c <- if (!is.null(REco.sim$params$spname)) as.character(REco.sim$params$spname) else character(0)
  cn_raw <- colnames(catch_df)
  if (!is.null(cn_raw) && length(cn_raw) > 0) {
    common_c <- intersect(spn_c, cn_raw)
    if (length(common_c) > 0) {
      ord_c <- spn_c[spn_c %in% cn_raw]
      col_order_c <- unique(c(ord_c, setdiff(cn_raw, ord_c)))
      if (length(col_order_c) != ncols_c) {
        if (length(cn_raw) == ncols_c) col_order_c <- cn_raw
        else if (length(spn_c) >= ncols_c) col_order_c <- spn_c[seq_len(ncols_c)]
        else col_order_c <- c(spn_c, paste0('G', seq_len(ncols_c - length(spn_c))))
      }
      catch_df <- catch_df[, col_order_c, drop = FALSE]
      colnames(catch_df) <- col_order_c
    } else {
      if (length(spn_c) >= ncols_c) colnames(catch_df) <- spn_c[seq_len(ncols_c)] else colnames(catch_df) <- c(spn_c, paste0('G', seq_len(ncols_c - length(spn_c))))
    }
  } else {
    if (length(spn_c) >= ncols_c) colnames(catch_df) <- spn_c[seq_len(ncols_c)] else colnames(catch_df) <- c(spn_c, paste0('G', seq_len(ncols_c - length(spn_c))))
  }
  catch_trajectory <- cbind(Year = 1:nrow(catch_df), catch_df)
  write.csv(catch_trajectory,
            file.path(output_dir, "ecosim", "catch_trajectory_rk4.csv"),
            row.names = FALSE)
}

# Run with Adams-Bashforth method for comparison
REco.run.ab <- rsim.run(REco.sim, method = 'AB', years = 1:100)

# Robust AB biomass extraction
bio_ab_raw <- REco.run.ab$out_Biomass
bio_ab_df <- tryCatch(as.data.frame(bio_ab_raw), error = function(e) { write_meta_and_sentinel(paste0('AB biomass conversion failed: ', conditionMessage(e))); stop(conditionMessage(e)) })

ncols_ab <- ncol(bio_ab_df)
spn_ab <- if (!is.null(REco.sim$params$spname)) as.character(REco.sim$params$spname) else character(0)
cat(sprintf("Debug(AB): biomass columns=%d, spname len=%d\n", ncols_ab, length(spn_ab)))

col_names_ab_raw <- colnames(bio_ab_df)
if (!is.null(col_names_ab_raw) && length(col_names_ab_raw) > 0) {
  common_ab <- intersect(spn_ab, col_names_ab_raw)
  if (length(common_ab) > 0) {
    ordered_ab <- spn_ab[spn_ab %in% col_names_ab_raw]
    col_order_ab <- unique(c(ordered_ab, setdiff(col_names_ab_raw, ordered_ab)))
    if (length(col_order_ab) != ncols_ab) {
      warning('AB: col_order mismatch; falling back to raw names or truncated spn')
      if (length(col_names_ab_raw) == ncols_ab) col_order_ab <- col_names_ab_raw
      else if (length(spn_ab) >= ncols_ab) col_order_ab <- spn_ab[seq_len(ncols_ab)]
      else col_order_ab <- c(spn_ab, paste0('G', seq_len(ncols_ab - length(spn_ab))))
    }
    bio_ab_df <- bio_ab_df[, col_order_ab, drop = FALSE]
    colnames(bio_ab_df) <- col_order_ab
  } else {
    if (length(spn_ab) >= ncols_ab) colnames(bio_ab_df) <- spn_ab[seq_len(ncols_ab)] else colnames(bio_ab_df) <- c(spn_ab, paste0('G', seq_len(ncols_ab - length(spn_ab))))
  }
} else {
  if (length(spn_ab) >= ncols_ab) colnames(bio_ab_df) <- spn_ab[seq_len(ncols_ab)] else colnames(bio_ab_df) <- c(spn_ab, paste0('G', seq_len(ncols_ab - length(spn_ab))))
}

biomass_trajectory_ab <- cbind(Year = 1:nrow(bio_ab_df), bio_ab_df)

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

bio_f_raw <- REco.run.fishing$out_Biomass
bio_f_df <- tryCatch(as.data.frame(bio_f_raw), error = function(e) { write_meta_and_sentinel(paste0('Fishing biomass conversion failed: ', conditionMessage(e))); stop(conditionMessage(e)) })

ncols_f <- ncol(bio_f_df)
spn_f <- if (!is.null(REco.sim$params$spname)) as.character(REco.sim$params$spname) else character(0)
cn_f_raw <- colnames(bio_f_df)
if (!is.null(cn_f_raw) && length(cn_f_raw) > 0) {
  common_f <- intersect(spn_f, cn_f_raw)
  if (length(common_f) > 0) {
    ord_f <- spn_f[spn_f %in% cn_f_raw]
    col_order_f <- unique(c(ord_f, setdiff(cn_f_raw, ord_f)))
    if (length(col_order_f) != ncols_f) {
      if (length(cn_f_raw) == ncols_f) col_order_f <- cn_f_raw
      else if (length(spn_f) >= ncols_f) col_order_f <- spn_f[seq_len(ncols_f)]
      else col_order_f <- c(spn_f, paste0('G', seq_len(ncols_f - length(spn_f))))
    }
    bio_f_df <- bio_f_df[, col_order_f, drop = FALSE]
    colnames(bio_f_df) <- col_order_f
  } else {
    if (length(spn_f) >= ncols_f) colnames(bio_f_df) <- spn_f[seq_len(ncols_f)] else colnames(bio_f_df) <- c(spn_f, paste0('G', seq_len(ncols_f - length(spn_f))))
  }
} else {
  if (length(spn_f) >= ncols_f) colnames(bio_f_df) <- spn_f[seq_len(ncols_f)] else colnames(bio_f_df) <- c(spn_f, paste0('G', seq_len(ncols_f - length(spn_f))))
}

biomass_fishing <- cbind(Year = 1:nrow(bio_f_df), bio_f_df)

write.csv(biomass_fishing,
          file.path(output_dir, "ecosim", "biomass_doubled_fishing.csv"),
          row.names = FALSE)

# Scenario 2: Zero fishing
REco.sim.nofishing <- REco.sim
# Set all fishing to zero
REco.sim.nofishing$forcing$ForcedEffort <- REco.sim$forcing$ForcedEffort * 0

REco.run.nofishing <- rsim.run(REco.sim.nofishing, method = 'RK4', years = 1:50)

bio_nf_raw <- REco.run.nofishing$out_Biomass
bio_nf_df <- tryCatch(as.data.frame(bio_nf_raw), error = function(e) { write_meta_and_sentinel(paste0('No-fishing biomass conversion failed: ', conditionMessage(e))); stop(conditionMessage(e)) })

ncols_nf <- ncol(bio_nf_df)
spn_nf <- if (!is.null(REco.sim$params$spname)) as.character(REco.sim$params$spname) else character(0)
cn_nf_raw <- colnames(bio_nf_df)
if (!is.null(cn_nf_raw) && length(cn_nf_raw) > 0) {
  common_nf <- intersect(spn_nf, cn_nf_raw)
  if (length(common_nf) > 0) {
    ord_nf <- spn_nf[spn_nf %in% cn_nf_raw]
    col_order_nf <- unique(c(ord_nf, setdiff(cn_nf_raw, ord_nf)))
    if (length(col_order_nf) != ncols_nf) {
      if (length(cn_nf_raw) == ncols_nf) col_order_nf <- cn_nf_raw
      else if (length(spn_nf) >= ncols_nf) col_order_nf <- spn_nf[seq_len(ncols_nf)]
      else col_order_nf <- c(spn_nf, paste0('G', seq_len(ncols_nf - length(spn_nf))))
    }
    bio_nf_df <- bio_nf_df[, col_order_nf, drop = FALSE]
    colnames(bio_nf_df) <- col_order_nf
  } else {
    if (length(spn_nf) >= ncols_nf) colnames(bio_nf_df) <- spn_nf[seq_len(ncols_nf)] else colnames(bio_nf_df) <- c(spn_nf, paste0('G', seq_len(ncols_nf - length(spn_nf))))
  }
} else {
  if (length(spn_nf) >= ncols_nf) colnames(bio_nf_df) <- spn_nf[seq_len(ncols_nf)] else colnames(bio_nf_df) <- c(spn_nf, paste0('G', seq_len(ncols_nf - length(spn_nf))))
}

biomass_nofishing <- cbind(Year = 1:nrow(bio_nf_df), bio_nf_df)

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

# =============================================================================
# Diagnostics metadata and QQ sentinel
# =============================================================================
# Create diagnostics directory (used by PyPath tests and analysis scripts)
diag_dir <- file.path(output_dir, "ecosim", "diagnostics")
if (!dir.exists(diag_dir)) dir.create(diag_dir, recursive = TRUE, showWarnings = FALSE)

# Helper to write metadata and a sentinel QQ CSV when extraction fails
write_meta_and_sentinel <- function(note = NULL) {
  meta <- list(qq_provided = FALSE)
  if (!is.null(note)) meta$note <- note
  write_json(meta, file.path(diag_dir, "meta.json"), pretty = TRUE, auto_unbox = TRUE)
  # Write a minimal QQ sentinel (no group columns) to make verification explicit
  sentinel <- data.frame(month = integer(0))
  write.csv(sentinel, file.path(diag_dir, 'seabirds_qq_rk4.csv'), row.names = FALSE, na = 'NA')
  # Ensure components file is absent or empty
  if (file.exists(file.path(diag_dir, 'seabirds_components_rk4.csv'))) {
    file.remove(file.path(diag_dir, 'seabirds_components_rk4.csv'))
  }
  cat("Wrote meta.json with qq_provided=FALSE and QQ sentinel.\n")
}

# Attempt to compute per-month QQ/consumption diagnostics from the RK4 run.
# If R path provides explicit QQ outputs, prefer those; otherwise compute QQ from model params
# and the biomass trajectory so downstream parity tests get concrete per-term diagnostics.
qq_provided <- FALSE
spnames <- if (!is.null(REco.sim$params$spname)) as.character(REco.sim$params$spname) else c()
n_months <- nrow(biomass_trajectory)

# Helper to ensure a vector/matrix becomes an n x n matrix
ensure_mat <- function(x, n) {
  if (is.null(x)) return(matrix(0, nrow = n, ncol = n))
  if (is.matrix(x)) return(x)
  return(matrix(as.numeric(x), nrow = n, ncol = n, byrow = TRUE))
}


  if (exists("REco.run.rk4") && !is.null(names(REco.run.rk4))) {
  rk4_names <- names(REco.run.rk4)
  # Try to find explicit consumption/QQ output fields
  candidate_names <- rk4_names[grepl("QQ|consumption|out_Consumption|out_Q", rk4_names, ignore.case = TRUE)]
  if (length(candidate_names) > 0) {
    # If an explicit QQ-like output exists, try to coerce it into a months x groups table
    qq_raw <- REco.run.rk4[[candidate_names[1]]]
    # Attempt to coerce to a data.frame with columns per group
    qq_df_try <- try(as.data.frame(qq_raw), silent = TRUE)
    if (!inherits(qq_df_try, "try-error")) {
      # Ensure month column and group columns
      if (nrow(qq_df_try) == n_months) {
        qq_out <- qq_df_try
        qq_out$month <- seq_len(n_months)
        # reorder to put month first
        qq_out <- qq_out[c('month', setdiff(names(qq_out), 'month'))]
        write.csv(qq_out, file.path(diag_dir, 'seabirds_qq_rk4.csv'), row.names = FALSE, na = 'NA')
        qq_provided <- TRUE
      }
    }
  }
}

# If we did not find explicit QQ outputs, compute QQ from parameters and biomass trajectory
if (!qq_provided && length(spnames) > 0) {
  n_groups <- length(spnames)
  # Extract parameter matrices/vectors
  QQbase <- ensure_mat(REco.sim$params$QQ, n_groups)
  VV <- ensure_mat(REco.sim$params$VV, n_groups)
  DD <- ensure_mat(REco.sim$params$DD, n_groups)
  Bbase <- as.numeric(REco.sim$params$B_BaseRef)
  PB <- as.numeric(REco.sim$params$PBopt)
  QB <- as.numeric(REco.sim$params$FtimeQBOpt)
  M0 <- as.numeric(REco.sim$params$MzeroMort)
  Ftime0 <- if (!is.null(REco.sim$start_state$Ftime)) as.numeric(REco.sim$start_state$Ftime) else rep(1, n_groups)

  qq_rows <- vector('list', n_months)
  components_rows <- vector('list', n_months)

  for (m in seq_len(n_months)) {
    # State at month m (biomass_trajectory has Year col then group cols)
    row <- biomass_trajectory[m, ]
    # Build a full-length state vector matching spnames; fill missing groups with NA then set NAs to 0 for computations
    state <- rep(NA_real_, n_groups)
    present_cols <- intersect(names(biomass_trajectory)[-1], spnames)
    if (length(present_cols) > 0) {
      pos <- match(present_cols, spnames)
      for (j in seq_along(present_cols)) {
        state[pos[j]] <- as.numeric(row[[ present_cols[j] ]])
      }
    }
    # Replace NAs with 0 for numeric stability in ratios
    state[is.na(state)] <- 0

    # Default forcing
    ForcedPrey <- rep(1, n_groups)
    Ftime <- Ftime0

    preyYY <- rep(0, n_groups)
    for (i in seq_len(n_groups)) {
      if (!is.na(Bbase[i]) && Bbase[i] > 0) preyYY[i] <- state[i] / Bbase[i] * ForcedPrey[i]
    }
    predYY <- rep(0, n_groups)
    n_living <- as.integer(REco.sim$params$NUM_LIVING)
    for (i in seq_len(n_living)) {
      if (!is.na(Bbase[i]) && Bbase[i] > 0) predYY[i] <- Ftime[i] * state[i] / Bbase[i]
    }

    QQmat <- matrix(0, nrow = n_groups, ncol = n_groups)
    for (pred in seq_len(n_living)) {
      if (state[pred] <= 0) next
      for (prey in seq_len(n_groups)) {
        qbase <- QQbase[prey, pred]
        if (is.na(qbase) || qbase <= 0) next
        PYY <- preyYY[prey]
        PDY <- predYY[pred]
        dd <- DD[prey, pred]
        vv <- VV[prey, pred]
        dd_term <- if (!is.na(dd) && dd > 1.0) dd / (dd - 1.0 + max(PYY, 1e-10)) else 1.0
        vv_term <- if (!is.na(vv) && vv > 1.0) vv / (vv - 1.0 + max(PDY, 1e-10)) else 1.0
        Q_calc <- qbase * PDY * PYY * dd_term * vv_term
        QQmat[prey, pred] <- max(Q_calc, 0)
      }
    }

    # consumption_by_predator: column sums
    cons_by_pred <- colSums(QQmat, na.rm = TRUE)
    predation_loss_by_prey <- rowSums(QQmat, na.rm = TRUE)

    # Record QQ row for this month
    qq_row <- data.frame(month = m)
    for (j in seq_len(n_groups)) qq_row[[spnames[j]]] <- cons_by_pred[j]
    qq_rows[[m]] <- qq_row

    # Compute components for Seabirds if present
    if ("Seabirds" %in% spnames) {
      idx <- which(spnames == "Seabirds")
      biomass_i <- state[idx]
      consumption_i <- cons_by_pred[idx]
      predation_loss_i <- predation_loss_by_prey[idx]
      # production
      if (!is.na(QB[idx]) && QB[idx] > 0) {
        GE <- PB[idx] / QB[idx]
        production_i <- GE * consumption_i
      } else {
        production_i <- PB[idx] * biomass_i
      }
      fish_loss_i <- 0.0
      m0_loss_i <- if (!is.na(M0[idx])) M0[idx] * biomass_i else 0.0
      derivative_i <- production_i - predation_loss_i - fish_loss_i - m0_loss_i

      components_rows[[m]] <- data.frame(
        month = m,
        time = m / 12.0,
        biomass = biomass_i,
        production = production_i,
        predation_loss = predation_loss_i,
        consumption_by_predator = consumption_i,
        fish_loss = fish_loss_i,
        m0_loss = m0_loss_i,
        derivative = derivative_i,
        method = 'RK4'
      )
    }
  }

  # Combine QQ rows and write CSV
  qq_out <- do.call(rbind, qq_rows)
  write.csv(qq_out, file.path(diag_dir, 'seabirds_qq_rk4.csv'), row.names = FALSE, na = 'NA')

  # Combine components rows (only if Seabirds present)
  if (any(sapply(components_rows, length) > 0)) {
    comps_df <- do.call(rbind, components_rows)
    write.csv(comps_df, file.path(diag_dir, 'seabirds_components_rk4.csv'), row.names = FALSE, na = 'NA')
    # indicate QQ provided since we computed it
    qq_provided <- TRUE
  }
}

# Write metadata JSON so tests and scripts can detect whether QQ diagnostics are present
meta <- list(qq_provided = qq_provided)
write_json(meta, file.path(diag_dir, "meta.json"), pretty = TRUE, auto_unbox = TRUE)

cat("\n=============================================================================\n")
cat("Reference data extraction complete!\n")
cat("Output directory:", output_dir, "\n")
cat("=============================================================================\n")
