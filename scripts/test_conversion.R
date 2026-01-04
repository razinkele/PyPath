library(Rpath)

data(REco.params)
REco <- rpath(REco.params, eco.name = "REcosystem")

cat("Original type:\n")
print(REco$type)
cat("Class:", class(REco$type), "\n")
cat("Length:", length(REco$type), "\n\n")

cat("as.numeric(REco$type):\n")
type_num <- as.numeric(REco$type)
print(type_num)
cat("Length:", length(type_num), "\n\n")

cat("unname(as.numeric(REco$type)):\n")
type_unname <- unname(as.numeric(REco$type))
print(type_unname)
cat("Length:", length(type_unname), "\n\n")

cat("c(REco$type):\n")
type_c <- c(REco$type)
print(type_c)
cat("Length:", length(type_c), "\n\n")

cat("as.vector(REco$type):\n")
type_vec <- as.vector(REco$type)
print(type_vec)
cat("Length:", length(type_vec), "\n")
