# Compute intermediate matrices (A and b) for REco balanced model
# and save them to tests/data/rpath_reference/ecopath for comparison.

if (!require("Rpath", quietly = TRUE)) {
  install.packages("Rpath")
}
if (!require("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}

library(Rpath)
library(jsonlite)

output_dir <- "tests/data/rpath_reference/ecopath"

# Load params and run rpath
data(REco.params)
REco <- rpath(REco.params, eco.name = "REcosystem")

model <- REco.params$model
diet <- REco.params$diet

groups <- as.character(model$Group)
types <- as.numeric(model$Type)
ngroups <- length(groups)

# Indices
living_idx <- which(types < 2)
dead_idx <- which(types == 2)
fleet_idx <- which(types == 3)
nliving <- length(living_idx)

# Build diet_values matrix (rows = all groups + Import, cols = predators that are living)
diet_prey_names <- as.character(diet$Group)
pred_cols <- intersect(groups, colnames(diet))
n_pred <- length(pred_cols)
diet_values <- matrix(0, nrow = ngroups + 1, ncol = n_pred)
rownames(diet_values) <- c(groups, "Import")
colnames(diet_values) <- pred_cols

for (i in seq_along(groups)) {
  g <- groups[i]
  if (g %in% diet_prey_names) {
    row_idx <- which(diet_prey_names == g)
    diet_values[i, ] <- as.numeric(diet[row_idx, pred_cols])
  }
}
# Import row
if ("Import" %in% diet_prey_names) {
  import_row_idx <- which(diet_prey_names == "Import")
  if (length(diet_prey_names) > ngroups) {
    diet_values[ngroups + 1, ] <- as.numeric(diet[import_row_idx, pred_cols])
  }
}
# Replace NA with 0
diet_values[is.na(diet_values)] <- 0

# Adjust mixotrophs (type between 0 and 1)
for (j_local in seq_along(pred_cols)) {
  global_pred_idx <- which(groups == pred_cols[j_local])
  t <- types[global_pred_idx]
  if (t > 0 && t < 1) {
    mix_q <- 1 - t
    diet_values[, j_local] <- diet_values[, j_local] * mix_q
  }
}

# Build nodetrdiet (rows = living prey only, cols = living predators)
nodetrdiet <- matrix(0, nrow = nliving, ncol = nliving)
for (j in seq_len(nliving)) {
  pred_name <- groups[living_idx[j]]
  # find column index for pred_name in pred_cols
  col_idx <- which(pred_cols == pred_name)
  import_frac <- diet_values[ngroups + 1, col_idx]
  denom <- if ((1 - import_frac) > 0) (1 - import_frac) else 1
  for (i in seq_len(nliving)) {
    prey_global <- living_idx[i]
    nodetrdiet[i, j] <- diet_values[prey_global, col_idx] / denom
  }
}

# Balanced values from REco
B_bal <- as.numeric(REco$Biomass)
PB_bal <- as.numeric(REco$PB)
QB_bal <- as.numeric(REco$QB)
EE_bal <- as.numeric(REco$EE)

# Original input masks
orig_no_b <- is.na(model$Biomass)
orig_no_ee <- is.na(model$EE)

# Compute bio_qb using balanced values
bio_qb <- B_bal[living_idx] * QB_bal[living_idx]
# Replace NA with 0
bio_qb[is.na(bio_qb)] <- 0

# Consumption contributions
cons <- matrix(0, nrow = nliving, ncol = nliving)
for (j in seq_len(nliving)) {
  cons[, j] <- nodetrdiet[, j] * bio_qb[j]
}

# Landings/discards matrices from input model if available
landmat <- matrix(0, nrow = ngroups, ncol = length(fleet_idx))
discardmat <- matrix(0, nrow = ngroups, ncol = length(fleet_idx))
if (length(fleet_idx) > 0) {
  fleet_names <- groups[fleet_idx]
  for (g in seq_along(fleet_names)) {
    col_name <- fleet_names[g]
    disc_name <- paste0(col_name, ".disc")
    if (col_name %in% colnames(model)) {
      landmat[, g] <- as.numeric(model[[col_name]])
    }
    if (disc_name %in% colnames(model)) {
      discardmat[, g] <- as.numeric(model[[disc_name]])
    }
  }
}
landmat[is.na(landmat)] <- 0
discardmat[is.na(discardmat)] <- 0

totcatch <- rowSums(landmat + discardmat)

bioacc <- as.numeric(model$BioAcc)
bioacc[is.na(bioacc)] <- 0

living_catch <- totcatch[living_idx]
living_bioacc <- bioacc[living_idx]

# b_vec
b_vec <- living_catch + living_bioacc + rowSums(cons)

# Build A matrix
A <- matrix(0, nrow = nliving, ncol = nliving)
for (i in seq_len(nliving)) {
  idx_global <- living_idx[i]
  if (orig_no_ee[idx_global]) {
    if (!is.na(B_bal[idx_global])) {
      A[i, i] <- B_bal[idx_global] * PB_bal[idx_global]
    } else {
      A[i, i] <- PB_bal[idx_global] * EE_bal[idx_global]
    }
  } else {
    A[i, i] <- PB_bal[idx_global] * EE_bal[idx_global]
  }
}
qb_dc <- matrix(0, nrow = nliving, ncol = nliving)
for (j in seq_len(nliving)) {
  qb_dc[, j] <- nodetrdiet[, j] * QB_bal[living_idx[j]]
}
for (j in seq_len(nliving)) {
  if (orig_no_b[living_idx[j]]) {
    A[, j] <- A[, j] - qb_dc[, j]
  }
}

# Save outputs
write.csv(A, file = file.path(output_dir, "r_A.csv"), row.names = FALSE)
write.csv(as.data.frame(b_vec), file = file.path(output_dir, "r_b_vec.csv"), row.names = FALSE)
write.csv(nodetrdiet, file = file.path(output_dir, "r_nodetrdiet.csv"), row.names = FALSE)
write.csv(as.data.frame(bio_qb), file = file.path(output_dir, "r_bio_qb.csv"), row.names = FALSE)

json_write <- list(
  orig_no_b = as.logical(orig_no_b),
  orig_no_ee = as.logical(orig_no_ee),
  living_idx = as.integer(living_idx),
  groups = groups
)
write_json(json_write, path = file.path(output_dir, "r_intermediates_meta.json"), pretty = TRUE)
cat("Saved R intermediates to", output_dir, "\n")
