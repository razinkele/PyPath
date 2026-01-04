library(Rpath)

data(REco.params)

cat("Running rpath()...\n")
REco <- rpath(REco.params, eco.name = "REcosystem")

cat("\nREco class:", class(REco), "\n")
cat("REco names:", names(REco), "\n\n")

cat("Group:\n")
print(REco$Group)
cat("Length:", length(REco$Group), "\n\n")

cat("Type:\n")
print(REco$type)
cat("Length:", length(REco$type), "\n\n")

cat("Biomass:\n")
print(REco$Biomass)
cat("Length:", length(REco$Biomass), "\n\n")

cat("All lengths:\n")
for (name in names(REco)) {
  if (is.vector(REco[[name]])) {
    cat(sprintf("%-15s : %d\n", name, length(REco[[name]])))
  } else if (is.matrix(REco[[name]])) {
    cat(sprintf("%-15s : matrix %dx%d\n", name, nrow(REco[[name]]), ncol(REco[[name]])))
  } else {
    cat(sprintf("%-15s : %s\n", name, class(REco[[name]])[1]))
  }
}
