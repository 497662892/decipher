#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

cran <- Sys.getenv("CRAN_REPO", "https://cloud.r-project.org")
lib <- Sys.getenv("R_LIBS_USER", file.path(getwd(), "data/resources/R/4.5.3-library"))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

packages <- c("data.table", "Matrix", "bigsparser", "bigsnpr")
installed <- vapply(packages, requireNamespace, logical(1), quietly = TRUE)
missing <- packages[!installed]

cat("R version:", R.version.string, "\n")
cat("Library:", lib, "\n")
cat("CRAN:", cran, "\n")
cat("Missing:", if (length(missing) == 0L) "<none>" else paste(missing, collapse = ", "), "\n")

if (length(missing) > 0L) {
  install.packages(missing, lib = lib, repos = cran, Ncpus = max(1L, parallel::detectCores(logical = FALSE)))
}

status <- vapply(packages, requireNamespace, logical(1), quietly = TRUE)
print(data.frame(package = packages, installed = status), row.names = FALSE)

if (!all(status)) {
  stop("Some LDpred2 R dependencies are still missing.", call. = FALSE)
}
