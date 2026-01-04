library(Rpath)
library(jsonlite)

# Load data
data(REco.params)

cat("REco.params structure:\n")
cat("Names:", names(REco.params), "\n\n")

cat("Model DataFrame:\n")
print(head(REco.params$model))
cat("\nDim:", dim(REco.params$model), "\n\n")

cat("Diet DataFrame:\n")
print(head(REco.params$diet))
cat("\nDim:", dim(REco.params$diet), "\n\n")

cat("Stanzas:\n")
print(names(REco.params$stanzas))

# Try to run rpath
cat("\n\nTrying to run rpath()...\n")
tryCatch({
  REco <- rpath(REco.params, eco.name = "REcosystem")
  cat("✓ rpath() succeeded\n")
  cat("Output class:", class(REco), "\n")
  cat("Output names:", names(REco), "\n")
}, error = function(e) {
  cat("✗ rpath() failed:", conditionMessage(e), "\n")
  traceback()
})
