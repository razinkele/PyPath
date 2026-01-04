# Check what data is available in Rpath package

if (!require("Rpath", quietly = TRUE)) {
  cat("Rpath package not installed. Installing...\n")
  install.packages("Rpath")
  library(Rpath)
}

cat("Rpath version:", as.character(packageVersion("Rpath")), "\n\n")

# List available datasets
cat("Available datasets in Rpath:\n")
data_list <- data(package = "Rpath")$results
print(data_list)

cat("\n\nTrying to load REco data...\n")
# Try different ways to load REco data
tryCatch({
  data("REco.params", package = "Rpath")
  cat("✓ REco.params loaded\n")
  cat("Structure:\n")
  str(REco.params, max.level = 1)
}, error = function(e) {
  cat("✗ Error loading REco.params:", conditionMessage(e), "\n")
})

tryCatch({
  data("REco.groups", package = "Rpath")
  cat("✓ REco.groups loaded\n")
}, error = function(e) {
  cat("✗ Error loading REco.groups:", conditionMessage(e), "\n")
})

# Check for example models in Rpath
cat("\n\nLooking for example models...\n")
search_path <- find.package("Rpath")
cat("Package path:", search_path, "\n")
