#!/usr/bin/env Rscript
# OADP is implemented by the mature bigutilsr package, also used by bigsnpr.
suppressPackageStartupMessages(library(data.table))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
out <- args[2]
data.table::setDTthreads(1)
pcs <- paste0("PC", 1:10)
bridge <- jsonlite::read_json(file.path(out, "geometry.json"), simplifyVector = TRUE)
stopifnot(bridge$orthonormal_loadings_verified)
data <- fread(file.path(out, "private", "target_geometry.tsv"), colClasses = list(character = "sampleid"))
XV <- as.matrix(data[, ..pcs]); norms <- bridge$weight_norms; d <- bridge$singular_values
stopifnot(all(is.finite(XV)), all(is.finite(data$X_norm)), !anyDuplicated(data$sampleid))
corrected_svd <- bigutilsr::pca_OADP_proj2(XV, data$X_norm, d)
stopifnot(all(is.finite(corrected_svd)))
check_ind <- unique(c(1L, as.integer(nrow(data)/2), nrow(data)))
check <- bigutilsr::pca_OADP_proj2(XV[rev(check_ind), ], data$X_norm[rev(check_ind)], d)
stopifnot(max(abs(check - corrected_svd[rev(check_ind), ])) < 1e-9)
raw <- sweep(XV, 2, norms, "*")
corrected <- sweep(corrected_svd, 2, norms, "*")
output <- data.table(sampleid = data$sampleid)
rows <- list()
for (j in seq_along(pcs)) {
  output[[paste0("refPC", j)]] <- corrected[, j]
  fit <- lm(corrected[, j] ~ raw[, j])
  rows[[j]] <- data.table(pc = pcs[j], simple_sd = sd(raw[, j]), corrected_sd = sd(corrected[, j]),
    sd_ratio = sd(corrected[, j]) / sd(raw[, j]), correlation = cor(raw[, j], corrected[, j]),
    affine_slope = unname(coef(fit)[2]), non_affine_rms_over_simple_sd = sqrt(mean(residuals(fit)^2))/sd(raw[, j]),
    rms_change_over_simple_sd = sqrt(mean((corrected[, j] - raw[, j])^2))/sd(raw[, j]))
}
fwrite(output, file.path(out, "target_pcs.tsv"), sep = "\t")
fwrite(rbindlist(rows), file.path(out, "correction_by_pc.csv"))
writeLines(capture.output(sessionInfo()), file.path(out, "R_session.txt"))
jsonlite::write_json(list(orthonormal_loadings_verified = TRUE,
  max_orthogonality_error = bridge$max_orthogonality_error,
  eigenvalue_scale_relative_error = bridge$eigenvalue_scale_relative_error,
  target_order_invariant = TRUE, dosage_replay = bridge$dosage_replay,
  packages = list(bigutilsr = as.character(packageVersion("bigutilsr")), pgenlib = bridge$pgenlib_version),
  note = "OADP correction magnitude; no claim of independently measured true shrinkage bias."),
  file.path(out, "diagnostics.json"), pretty = TRUE, auto_unbox = TRUE)
cat("OADP complete:", nrow(output), "patients\n")
