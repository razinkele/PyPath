library(Rpath)

data(REco.params)
cat("REco.params loaded successfully\n\n")

cat("Running rpath()...\n")
REco <- rpath(REco.params, eco.name = "REcosystem")
cat("rpath() completed\n\n")

cat("Checking field types and lengths:\n")
for (field in names(REco)) {
  val <- REco[[field]]
  if (is.vector(val) && !is.list(val)) {
    cat(sprintf("%-20s: vector, length=%d\n", field, length(val)))
  } else if (is.matrix(val)) {
    cat(sprintf("%-20s: matrix, dim=%dx%d\n", field, nrow(val), ncol(val)))
  } else if (is.list(val)) {
    cat(sprintf("%-20s: list, length=%d\n", field, length(val)))
  } else {
    cat(sprintf("%-20s: %s\n", field, class(val)[1]))
  }
}

cat("\nTrying different extraction methods:\n\n")

cat("Method 1: as.vector(REco$type)\n")
type1 <- as.vector(REco$type)
cat("  Length:", length(type1), "\n")
cat("  Class:", class(type1), "\n")
cat("  First 5:", type1[1:5], "\n\n")

cat("Method 2: c(REco$type)\n")
type2 <- c(REco$type)
cat("  Length:", length(type2), "\n")
cat("  Has names:", !is.null(names(type2)), "\n\n")

cat("Method 3: unname(REco$type)\n")
type3 <- unname(REco$type)
cat("  Length:", length(type3), "\n")
cat("  Has names:", !is.null(names(type3)), "\n\n")

cat("Method 4: REco$type[]  (bracket subsetting)\n")
type4 <- REco$type[]
cat("  Length:", length(type4), "\n")
cat("  Has names:", !is.null(names(type4)), "\n\n")

cat("Creating data.frame with different methods:\n\n")

cat("Trying method 1: as.vector\n")
tryCatch({
  df1 <- data.frame(
    Group = as.vector(REco$Group),
    Type = as.vector(REco$type),
    stringsAsFactors = FALSE
  )
  cat("  SUCCESS! Rows:", nrow(df1), "\n")
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
})

cat("\nTrying method 2: unname\n")
tryCatch({
  df2 <- data.frame(
    Group = unname(REco$Group),
    Type = unname(REco$type),
    stringsAsFactors = FALSE
  )
  cat("  SUCCESS! Rows:", nrow(df2), "\n")
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
})

cat("\nTrying method 3: c()\n")
tryCatch({
  df3 <- data.frame(
    Group = c(REco$Group),
    Type = c(REco$type),
    stringsAsFactors = FALSE
  )
  cat("  SUCCESS! Rows:", nrow(df3), "\n")
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
})

cat("\nTrying method 4: Bracket subsetting\n")
tryCatch({
  df4 <- data.frame(
    Group = REco$Group[],
    Type = REco$type[],
    stringsAsFactors = FALSE
  )
  cat("  SUCCESS! Rows:", nrow(df4), "\n")
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
})

cat("\nChecking if Groupis character:\n")
cat("  Class:", class(REco$Group), "\n")
cat("  Is character:", is.character(REco$Group), "\n")
cat("  First 3:", REco$Group[1:3], "\n")
