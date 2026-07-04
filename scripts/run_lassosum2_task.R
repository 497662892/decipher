#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function(argv) {
  args <- list()
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) {
      stop("Unexpected positional argument: ", key, call. = FALSE)
    }
    name <- sub("^--", "", key)
    if (i == length(argv) || startsWith(argv[[i + 1L]], "--")) {
      args[[name]] <- "true"
      i <- i + 1L
    } else {
      args[[name]] <- argv[[i + 1L]]
      i <- i + 2L
    }
  }
  args
}

arg_value <- function(args, name, default = NULL, required = FALSE) {
  value <- args[[name]]
  if (is.null(value)) {
    if (required) {
      stop("Missing required argument --", name, call. = FALSE)
    }
    return(default)
  }
  value
}

parse_bool <- function(value, default = FALSE) {
  if (is.null(value)) {
    return(default)
  }
  lowered <- tolower(as.character(value))
  if (lowered %in% c("1", "true", "t", "yes", "y")) {
    return(TRUE)
  }
  if (lowered %in% c("0", "false", "f", "no", "n")) {
    return(FALSE)
  }
  stop("Cannot parse boolean value: ", value, call. = FALSE)
}

parse_numeric_vector <- function(value) {
  parsed <- as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
  parsed <- parsed[is.finite(parsed)]
  if (length(parsed) == 0L) {
    stop("Numeric grid is empty: ", value, call. = FALSE)
  }
  parsed
}

require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("R package '", pkg, "' is required for lassosum2 tasks.", call. = FALSE)
  }
}

first_column <- function(frame, candidates, context) {
  for (candidate in candidates) {
    if (candidate %in% names(frame)) {
      return(candidate)
    }
  }
  stop(context, " is missing one of: ", paste(candidates, collapse = ", "), call. = FALSE)
}

read_keep <- function(path) {
  keep <- data.table::fread(path, data.table = FALSE, showProgress = FALSE, check.names = FALSE)
  if (nrow(keep) == 0L || ncol(keep) == 0L) {
    stop("Keep file has no samples: ", path, call. = FALSE)
  }
  col <- first_column(keep, c("#IID", "IID", "sampleid", "patient_id", "sample_id"), paste("Keep file", path))
  unique(as.character(keep[[col]]))
}

fam_sample_ids <- function(fam) {
  candidates <- c("sample.ID", "sample_id", "IID", "iid", "V2")
  for (candidate in candidates) {
    if (candidate %in% names(fam)) {
      return(as.character(fam[[candidate]]))
    }
  }
  if (ncol(fam) >= 2L) {
    return(as.character(fam[[2L]]))
  }
  as.character(fam[[1L]])
}

map_for_lassosum2 <- function(map) {
  chr_col <- first_column(map, c("chromosome", "chr", "CHR"), "PLINK map")
  rsid_col <- first_column(map, c("marker.ID", "rsid", "SNP", "ID"), "PLINK map")
  pos_col <- first_column(map, c("physical.pos", "pos", "BP"), "PLINK map")
  a1_col <- first_column(map, c("allele1", "a1", "A1"), "PLINK map")
  a0_col <- first_column(map, c("allele2", "a0", "A2"), "PLINK map")
  data.frame(
    chr = suppressWarnings(as.integer(map[[chr_col]])),
    rsid = as.character(map[[rsid_col]]),
    pos = suppressWarnings(as.numeric(map[[pos_col]])),
    a1 = toupper(as.character(map[[a1_col]])),
    a0 = toupper(as.character(map[[a0_col]]))
  )
}

complement_allele <- function(allele) {
  mapping <- c(A = "T", T = "A", C = "G", G = "C")
  unname(mapping[allele])
}

allele_orientation <- function(ref_a1, ref_a0, target_a1, target_a0) {
  sign <- rep(NA_real_, length(ref_a1))
  direct <- target_a1 == ref_a1 & target_a0 == ref_a0
  reversed <- target_a1 == ref_a0 & target_a0 == ref_a1
  comp_a1 <- complement_allele(target_a1)
  comp_a0 <- complement_allele(target_a0)
  comp_direct <- comp_a1 == ref_a1 & comp_a0 == ref_a0
  comp_reversed <- comp_a1 == ref_a0 & comp_a0 == ref_a1
  sign[direct | comp_direct] <- 1
  sign[reversed | comp_reversed] <- -1
  sign
}

map_for_ld_reference <- function(path) {
  ref <- readRDS(path)
  chr_col <- first_column(ref, c("chr", "chromosome", "CHR"), "LD reference map")
  rsid_col <- first_column(ref, c("rsid", "marker.ID", "SNP", "ID"), "LD reference map")
  pos_col <- first_column(ref, c("pos_hg38", "pos", "BP", "physical.pos"), "LD reference map")
  a1_col <- first_column(ref, c("a1", "allele1", "A1"), "LD reference map")
  a0_col <- first_column(ref, c("a0", "allele2", "A0", "A2"), "LD reference map")
  out <- data.frame(
    chr = suppressWarnings(as.integer(ref[[chr_col]])),
    rsid = as.character(ref[[rsid_col]]),
    pos = suppressWarnings(as.numeric(ref[[pos_col]])),
    a1 = toupper(as.character(ref[[a1_col]])),
    a0 = toupper(as.character(ref[[a0_col]])),
    stringsAsFactors = FALSE
  )
  if ("ld" %in% names(ref)) {
    out$ld <- suppressWarnings(as.numeric(ref$ld))
  }
  if ("af_UKBB" %in% names(ref)) {
    out$af_UKBB <- suppressWarnings(as.numeric(ref$af_UKBB))
  }
  keep <- stats::complete.cases(out[, c("chr", "rsid", "pos", "a1", "a0")]) &
    out$chr %in% seq_len(22L) &
    out$a1 %in% c("A", "C", "G", "T") &
    out$a0 %in% c("A", "C", "G", "T")
  out[keep, ]
}

infer_beta_se_from_p <- function(beta, pval) {
  clipped_p <- pmin(pmax(pval, .Machine$double.xmin), 1)
  z <- stats::qnorm(clipped_p / 2, lower.tail = FALSE)
  inferred <- abs(beta) / z
  inferred[!is.finite(inferred) | inferred <= 0] <- NA_real_
  inferred
}

sumstats_for_lassosum2 <- function(path, default_n = NA_real_) {
  ss <- data.table::fread(path, data.table = FALSE, showProgress = FALSE)
  required <- c("SNP", "CHR", "BP", "A1", "A2", "BETA", "P")
  missing <- setdiff(required, names(ss))
  if (length(missing) > 0L) {
    stop("Harmonized sumstats missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  beta <- suppressWarnings(as.numeric(ss$BETA))
  pval <- suppressWarnings(as.numeric(ss$P))
  beta_se_from_p <- infer_beta_se_from_p(beta, pval)
  se_col <- intersect(c("SE", "BETA_SE", "SEBETA", "STDERR"), names(ss))
  if (length(se_col) > 0L) {
    beta_se_from_col <- suppressWarnings(as.numeric(ss[[se_col[[1L]]]]))
    valid_col <- is.finite(beta_se_from_col) & beta_se_from_col > 0
    valid_p <- is.finite(beta_se_from_p) & beta_se_from_p > 0
    beta_se <- beta_se_from_col
    beta_se[!valid_col & valid_p] <- beta_se_from_p[!valid_col & valid_p]
    n_beta_se_from_col <- sum(valid_col, na.rm = TRUE)
    n_beta_se_from_p <- sum(!valid_col & valid_p, na.rm = TRUE)
  } else {
    message("SE column is absent; deriving approximate beta_se from BETA and P.")
    beta_se <- beta_se_from_p
    n_beta_se_from_col <- 0L
    n_beta_se_from_p <- sum(is.finite(beta_se) & beta_se > 0, na.rm = TRUE)
  }

  n_col <- intersect(c("N", "N_EFF", "NEFF", "N_eff", "n_eff"), names(ss))
  if (length(n_col) > 0L) {
    n_eff_from_col <- suppressWarnings(as.numeric(ss[[n_col[[1L]]]]))
    valid_n_col <- is.finite(n_eff_from_col) & n_eff_from_col > 0
    n_eff <- n_eff_from_col
    n_eff_from_default <- 0L
    if (is.finite(default_n) && default_n > 0) {
      n_eff[!valid_n_col] <- default_n
      n_eff_from_default <- sum(!valid_n_col, na.rm = TRUE)
    }
    n_eff_from_column <- sum(valid_n_col, na.rm = TRUE)
  } else if (is.finite(default_n) && default_n > 0) {
    n_eff <- rep(default_n, nrow(ss))
    n_eff_from_column <- 0L
    n_eff_from_default <- length(n_eff)
  } else {
    stop("Harmonized sumstats need N/N_eff, or pass --default-n.", call. = FALSE)
  }

  out <- data.frame(
    chr = suppressWarnings(as.integer(ss$CHR)),
    rsid = as.character(ss$SNP),
    pos = suppressWarnings(as.numeric(ss$BP)),
    a1 = toupper(as.character(ss$A1)),
    a0 = toupper(as.character(ss$A2)),
    beta = beta,
    beta_se = beta_se,
    n_eff = n_eff,
    stringsAsFactors = FALSE
  )
  keep <- stats::complete.cases(out) &
    out$chr %in% seq_len(22L) &
    out$a1 %in% c("A", "C", "G", "T") &
    out$a0 %in% c("A", "C", "G", "T") &
    out$beta_se > 0 &
    out$n_eff > 0
  out <- out[keep, ]
  attr(out, "input_summary") <- list(
    n_beta_se_from_column = as.integer(n_beta_se_from_col),
    n_beta_se_from_p = as.integer(n_beta_se_from_p),
    n_eff_from_column = as.integer(n_eff_from_column),
    n_eff_from_default = as.integer(n_eff_from_default)
  )
  out
}

match_reference_to_target <- function(ref_matched, target_map) {
  ref <- data.table::data.table(
    ref_row = seq_len(nrow(ref_matched)),
    chr = as.integer(ref_matched$chr),
    pos = as.numeric(ref_matched$pos),
    ref_a1 = as.character(ref_matched$a1),
    ref_a0 = as.character(ref_matched$a0)
  )
  target <- data.table::data.table(
    target_num_id = seq_len(nrow(target_map)),
    chr = as.integer(target_map$chr),
    pos = as.numeric(target_map$pos),
    target_a1 = as.character(target_map$a1),
    target_a0 = as.character(target_map$a0)
  )
  ref[, key := paste(chr, pos, sep = ":")]
  target[, key := paste(chr, pos, sep = ":")]
  ref <- ref[!is.na(chr) & !is.na(pos)]
  target <- target[!is.na(chr) & !is.na(pos)]
  ref[, n_ref_key := .N, by = key]
  target[, n_target_key := .N, by = key]
  ref <- ref[n_ref_key == 1L]
  target <- target[n_target_key == 1L]
  merged <- merge(
    ref[, .(key, ref_row, ref_a1, ref_a0)],
    target[, .(key, target_num_id, target_a1, target_a0)],
    by = "key",
    all = FALSE,
    sort = FALSE
  )
  if (nrow(merged) == 0L) {
    return(data.frame())
  }
  merged[, target_beta_sign := allele_orientation(ref_a1, ref_a0, target_a1, target_a0)]
  merged <- merged[is.finite(target_beta_sign)]
  merged[, ref_dup := .N, by = ref_row]
  merged[, target_dup := .N, by = target_num_id]
  merged <- merged[ref_dup == 1L & target_dup == 1L]
  data.frame(
    ref_row = merged$ref_row,
    target_num_id = merged$target_num_id,
    target_beta_sign = merged$target_beta_sign
  )
}

prepare_reference_matched <- function(sumstats, target_map, ld_ref_map, join_by_pos) {
  ref_matched <- bigsnpr::snp_match(
    sumstats,
    ld_ref_map[, intersect(c("chr", "rsid", "pos", "a1", "a0", "ld", "af_UKBB"), names(ld_ref_map))],
    strand_flip = TRUE,
    join_by_pos = join_by_pos
  )
  if (nrow(ref_matched) == 0L) {
    stop("No variants matched between harmonized sumstats and LD reference map.", call. = FALSE)
  }
  ref_matched$ld_ref_num_id <- ref_matched[["_NUM_ID_"]]
  target_match <- match_reference_to_target(ref_matched, target_map)
  if (nrow(target_match) == 0L) {
    stop("No LD reference variants could be matched to target genotype variants for scoring.", call. = FALSE)
  }
  matched <- ref_matched[target_match$ref_row, ]
  matched$ld_ref_num_id <- ref_matched$ld_ref_num_id[target_match$ref_row]
  matched$target_beta_sign <- target_match$target_beta_sign
  matched[["_NUM_ID_"]] <- target_match$target_num_id
  matched <- matched[order(matched$chr, matched$pos), ]
  matched
}

ld_ref_file <- function(ld_ref_dir, ld_ref_pattern, chr) {
  file.path(ld_ref_dir, gsub("{chr}", as.character(chr), ld_ref_pattern, fixed = TRUE))
}

ld_cache_paths <- function(ld_backing_prefix) {
  list(
    sbk = paste0(ld_backing_prefix, ".sbk"),
    rds = paste0(ld_backing_prefix, ".rds"),
    metadata = paste0(ld_backing_prefix, ".metadata.rds")
  )
}

same_vector <- function(left, right) {
  isTRUE(all.equal(left, right, check.attributes = FALSE, tolerance = 0))
}

same_ld_cache_metadata <- function(left, right) {
  if (is.null(left) || is.null(right)) {
    return(FALSE)
  }
  scalar_fields <- c("version", "disease", "fold", "model_scope", "n_matched", "ld_reference_mode", "ld_ref_map", "ld_ref_dir", "ld_ref_pattern")
  for (field in scalar_fields) {
    if (!identical(left[[field]], right[[field]])) {
      return(FALSE)
    }
  }
  vector_fields <- c("num_id", "rsid", "chr", "pos", "a1", "a0", "ld_ref_num_id", "target_beta_sign")
  for (field in vector_fields) {
    if (!same_vector(left[[field]], right[[field]])) {
      return(FALSE)
    }
  }
  TRUE
}

ld_cache_metadata <- function(disease, fold, model_scope, matched, num_id, ld_reference_mode, ld_ref_map_path, ld_ref_dir, ld_ref_pattern) {
  list(
    version = 1L,
    disease = disease,
    fold = fold,
    model_scope = model_scope,
    n_matched = nrow(matched),
    num_id = as.integer(num_id),
    rsid = as.character(matched$rsid),
    chr = as.integer(matched$chr),
    pos = as.numeric(matched$pos),
    a1 = as.character(matched$a1),
    a0 = as.character(matched$a0),
    ld_reference_mode = ld_reference_mode,
    ld_ref_map = normalizePath(ld_ref_map_path, mustWork = FALSE),
    ld_ref_dir = normalizePath(ld_ref_dir, mustWork = FALSE),
    ld_ref_pattern = ld_ref_pattern,
    ld_ref_num_id = as.integer(matched$ld_ref_num_id),
    target_beta_sign = as.numeric(matched$target_beta_sign)
  )
}

load_ld_cache <- function(ld_backing_prefix, expected_metadata, reuse_ld_corr) {
  if (!reuse_ld_corr) {
    message("LD correlation cache reuse is disabled; rebuilding ", ld_backing_prefix)
    return(NULL)
  }
  paths <- ld_cache_paths(ld_backing_prefix)
  if (!all(file.exists(unlist(paths)))) {
    return(NULL)
  }
  if (file.info(paths$sbk)$size <= 0 || file.info(paths$rds)$size <= 0) {
    message("Ignoring incomplete LD correlation cache at ", ld_backing_prefix)
    return(NULL)
  }
  cached_metadata <- tryCatch(readRDS(paths$metadata), error = function(err) NULL)
  if (!same_ld_cache_metadata(expected_metadata, cached_metadata)) {
    message("Existing LD correlation cache metadata does not match; rebuilding ", ld_backing_prefix)
    return(NULL)
  }
  message("Reusing LD correlation cache from ", ld_backing_prefix)
  readRDS(paths$rds)
}

build_external_ld_cache <- function(ld_ref_map, matched, ld_ref_dir, ld_ref_pattern, ld_backing_prefix, metadata) {
  paths <- ld_cache_paths(ld_backing_prefix)
  unlink(unlist(paths))
  corr <- NULL
  for (chr in sort(unique(matched$chr))) {
    idx <- which(matched$chr == chr)
    ref_chr_rows <- which(ld_ref_map$chr == chr)
    ind_chr <- match(matched$ld_ref_num_id[idx], ref_chr_rows)
    if (anyNA(ind_chr)) {
      stop("Could not map matched variants to LD reference chromosome ", chr, " matrix indices.", call. = FALSE)
    }
    path <- ld_ref_file(ld_ref_dir, ld_ref_pattern, chr)
    if (!file.exists(path)) {
      stop("Missing LD reference matrix for chr", chr, ": ", path, call. = FALSE)
    }
    message("Reading UKB LD reference chr", chr, " for ", length(ind_chr), " variants.")
    corr_chr <- readRDS(path)[ind_chr, ind_chr]
    if (is.null(corr)) {
      corr <- bigsparser::as_SFBM(corr_chr, backingfile = ld_backing_prefix, compact = TRUE)
    } else {
      corr$add_columns(corr_chr, nrow(corr))
      saveRDS(corr, paths$rds)
    }
  }
  saveRDS(corr, paths$rds)
  saveRDS(metadata, paths$metadata)
  corr
}

model_tag <- function(delta, lambda) {
  clean <- function(value) {
    gsub("\\+", "", gsub("\\.", "p", gsub("-", "m", format(value, scientific = FALSE, trim = TRUE))))
  }
  paste0("delta", clean(delta), "_lambda", clean(lambda))
}

optional_formal_arg <- function(fun, args, name, value) {
  if (name %in% names(formals(fun))) {
    args[[name]] <- value
  }
  args
}

replace_missing_by_col <- function(block, fill_values) {
  missing <- is.na(block)
  n_missing <- sum(missing)
  if (n_missing > 0L) {
    for (j in which(colSums(missing) > 0L)) {
      block[missing[, j], j] <- fill_values[[j]]
    }
  }
  list(block = block, n_missing = as.numeric(n_missing))
}

training_imputation_means <- function(G, ind_train, ind_col, block_size) {
  means <- numeric(length(ind_col))
  n_missing <- 0
  n_without_observed <- 0
  for (start in seq(1L, length(ind_col), by = block_size)) {
    idx <- seq.int(start, min(start + block_size - 1L, length(ind_col)))
    block <- G[ind_train, ind_col[idx], drop = FALSE]
    missing <- is.na(block)
    n_missing <- n_missing + as.numeric(sum(missing))
    n_observed <- colSums(!missing)
    block_means <- colSums(block, na.rm = TRUE) / n_observed
    without_observed <- !is.finite(block_means)
    if (any(without_observed)) {
      n_without_observed <- n_without_observed + sum(without_observed)
      block_means[without_observed] <- 0
    }
    means[idx] <- block_means
  }
  attr(means, "n_missing") <- n_missing
  attr(means, "n_without_observed") <- as.integer(n_without_observed)
  means
}

score_with_training_mean_imputation <- function(G, beta_grid, ind_row, ind_col, impute_values, block_size) {
  out <- matrix(0, nrow = length(ind_row), ncol = ncol(beta_grid))
  n_missing <- 0
  for (start in seq(1L, length(ind_col), by = block_size)) {
    idx <- seq.int(start, min(start + block_size - 1L, length(ind_col)))
    block <- G[ind_row, ind_col[idx], drop = FALSE]
    replaced <- replace_missing_by_col(block, impute_values[idx])
    n_missing <- n_missing + replaced$n_missing
    out <- out + replaced$block %*% beta_grid[idx, , drop = FALSE]
  }
  attr(out, "n_missing") <- n_missing
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

for (pkg in c("data.table", "bigsparser", "bigsnpr", "bigstatsr")) {
  require_pkg(pkg)
}

disease <- arg_value(args, "disease", required = TRUE)
fold <- as.integer(arg_value(args, "fold", "0"))
model_scope <- arg_value(args, "model-scope", "full_train")
bed_prefix <- arg_value(args, "bed-prefix", required = TRUE)
score_bed_prefix <- arg_value(args, "score-bed-prefix", bed_prefix)
fit_score_bed_prefix <- arg_value(args, "fit-score-bed-prefix", bed_prefix)
harmonized <- arg_value(args, "harmonized", required = TRUE)
train_keep <- arg_value(args, "train-keep", required = TRUE)
score_keep <- arg_value(args, "score-keep", required = TRUE)
fit_score_keep <- arg_value(args, "fit-score-keep", NULL)
patient_col <- arg_value(args, "patient-col", "sampleid")
score_out <- arg_value(args, "score-out", required = TRUE)
fit_score_out <- arg_value(args, "fit-score-out", NULL)
summary_out <- arg_value(args, "summary-out", required = TRUE)
model_out <- arg_value(args, "model-out", required = TRUE)
ld_backing_prefix <- arg_value(args, "ld-backing-prefix", required = TRUE)
delta_grid <- parse_numeric_vector(arg_value(args, "delta-grid", "0.001,0.01,0.1,1"))
nlambda <- as.integer(arg_value(args, "nlambda", "30"))
lambda_min_ratio <- as.numeric(arg_value(args, "lambda-min-ratio", "0.01"))
dfmax <- as.integer(arg_value(args, "dfmax", "200000"))
maxiter <- as.integer(arg_value(args, "maxiter", "1000"))
tol <- as.numeric(arg_value(args, "tol", "1e-5"))
threads <- as.integer(arg_value(args, "threads", "1"))
default_n <- as.numeric(arg_value(args, "default-n", "NA"))
join_by_pos <- parse_bool(arg_value(args, "join-by-pos", "true"))
reuse_ld_corr <- parse_bool(arg_value(args, "reuse-ld-corr", "true"))
save_beta_grid <- parse_bool(arg_value(args, "save-beta-grid", "false"))
ld_reference_mode <- tolower(arg_value(args, "ld-reference-mode", "ukb_hm3_plus"))
ld_ref_map_path <- arg_value(args, "ld-ref-map", required = TRUE)
ld_ref_dir <- arg_value(args, "ld-ref-dir", required = TRUE)
ld_ref_pattern <- arg_value(args, "ld-ref-pattern", "LD_with_blocks_chr{chr}.rds")
score_block_size <- as.integer(arg_value(args, "score-block-size", "5000"))

if (!ld_reference_mode %in% c("ukb_hm3_plus", "external")) {
  stop("--ld-reference-mode must be 'ukb_hm3_plus' or 'external' for lassosum2.", call. = FALSE)
}
if (is.null(fit_score_keep) != is.null(fit_score_out)) {
  stop("--fit-score-keep and --fit-score-out must be provided together.", call. = FALSE)
}
if (!is.finite(nlambda) || nlambda < 1L) {
  stop("--nlambda must be positive.", call. = FALSE)
}
if (!is.finite(lambda_min_ratio) || lambda_min_ratio <= 0 || lambda_min_ratio >= 1) {
  stop("--lambda-min-ratio must be in (0, 1).", call. = FALSE)
}
if (!is.finite(dfmax) || dfmax < 1L) {
  stop("--dfmax must be positive.", call. = FALSE)
}
if (!is.finite(maxiter) || maxiter < 1L) {
  stop("--maxiter must be positive.", call. = FALSE)
}
if (!is.finite(tol) || tol <= 0) {
  stop("--tol must be positive.", call. = FALSE)
}
if (!is.finite(threads) || threads < 1L) {
  threads <- 1L
}
if (!is.finite(score_block_size) || score_block_size < 1L) {
  score_block_size <- 5000L
}

out_dirs <- c(dirname(score_out), dirname(summary_out), dirname(model_out), dirname(ld_backing_prefix))
if (!is.null(fit_score_out)) {
  out_dirs <- c(out_dirs, dirname(fit_score_out))
}
for (path in out_dirs) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

attach_bigsnp <- function(prefix, context) {
  bed_file <- paste0(prefix, ".bed")
  if (!file.exists(bed_file)) {
    stop("Missing PLINK bed file for ", context, ": ", bed_file, call. = FALSE)
  }
  rds_file <- paste0(prefix, ".rds")
  if (!file.exists(rds_file)) {
    message("Creating bigSNP backing files from ", bed_file)
    rds_file <- bigsnpr::snp_readBed(bed_file, backingfile = prefix)
  }
  bigsnpr::snp_attach(rds_file)
}

obj <- attach_bigsnp(bed_prefix, "training")
G <- obj$genotypes
sample_ids <- fam_sample_ids(obj$fam)
map <- map_for_lassosum2(obj$map)

score_obj <- obj
if (!identical(normalizePath(score_bed_prefix, mustWork = FALSE), normalizePath(bed_prefix, mustWork = FALSE))) {
  score_obj <- attach_bigsnp(score_bed_prefix, "scoring")
}
G_score <- score_obj$genotypes
score_sample_ids_all <- fam_sample_ids(score_obj$fam)
score_map <- map_for_lassosum2(score_obj$map)

fit_score_obj <- obj
if (!identical(normalizePath(fit_score_bed_prefix, mustWork = FALSE), normalizePath(bed_prefix, mustWork = FALSE))) {
  if (identical(normalizePath(fit_score_bed_prefix, mustWork = FALSE), normalizePath(score_bed_prefix, mustWork = FALSE))) {
    fit_score_obj <- score_obj
  } else {
    fit_score_obj <- attach_bigsnp(fit_score_bed_prefix, "model-fit scoring")
  }
}
G_fit_score <- fit_score_obj$genotypes
fit_score_sample_ids_all <- fam_sample_ids(fit_score_obj$fam)
fit_score_map <- map_for_lassosum2(fit_score_obj$map)

train_ids <- read_keep(train_keep)
score_ids <- read_keep(score_keep)
ind_train <- match(train_ids, sample_ids)
ind_score <- match(score_ids, score_sample_ids_all)
if (anyNA(ind_train)) {
  stop("Training keep samples absent from PLINK bed: ", paste(head(train_ids[is.na(ind_train)]), collapse = ", "), call. = FALSE)
}
if (anyNA(ind_score)) {
  stop("Scoring keep samples absent from PLINK bed: ", paste(head(score_ids[is.na(ind_score)]), collapse = ", "), call. = FALSE)
}
fit_score_ids <- NULL
ind_fit_score <- NULL
if (!is.null(fit_score_keep)) {
  fit_score_ids <- read_keep(fit_score_keep)
  ind_fit_score <- match(fit_score_ids, fit_score_sample_ids_all)
  if (anyNA(ind_fit_score)) {
    stop("Model-fit scoring keep samples absent from PLINK bed: ", paste(head(fit_score_ids[is.na(ind_fit_score)]), collapse = ", "), call. = FALSE)
  }
}

sumstats <- sumstats_for_lassosum2(harmonized, default_n = default_n)
input_summary <- attr(sumstats, "input_summary")
if (nrow(sumstats) == 0L) {
  stop("No usable summary-statistic rows remain after lassosum2 validation.", call. = FALSE)
}
ld_ref_map <- map_for_ld_reference(ld_ref_map_path)
matched <- prepare_reference_matched(
  sumstats = sumstats,
  target_map = map,
  ld_ref_map = ld_ref_map,
  join_by_pos = join_by_pos
)
num_id <- matched[["_NUM_ID_"]]
ld_metadata <- ld_cache_metadata(
  disease = disease,
  fold = fold,
  model_scope = model_scope,
  matched = matched,
  num_id = num_id,
  ld_reference_mode = ld_reference_mode,
  ld_ref_map_path = ld_ref_map_path,
  ld_ref_dir = ld_ref_dir,
  ld_ref_pattern = ld_ref_pattern
)
corr <- load_ld_cache(ld_backing_prefix, ld_metadata, reuse_ld_corr = reuse_ld_corr)
ld_corr_reused <- !is.null(corr)
if (!ld_corr_reused) {
  message("Building LD correlation cache from external UKB LD reference at ", ld_backing_prefix)
  corr <- build_external_ld_cache(
    ld_ref_map = ld_ref_map,
    matched = matched,
    ld_ref_dir = ld_ref_dir,
    ld_ref_pattern = ld_ref_pattern,
    ld_backing_prefix = ld_backing_prefix,
    metadata = ld_metadata
  )
  gc()
}

lasso_args <- list(
  corr = corr,
  df_beta = matched,
  delta = delta_grid,
  nlambda = nlambda,
  lambda.min.ratio = lambda_min_ratio,
  dfmax = dfmax,
  maxiter = maxiter,
  tol = tol,
  ncores = threads
)
lasso_args <- optional_formal_arg(bigsnpr::snp_lassosum2, lasso_args, "ind.corr", seq_len(nrow(matched)))
beta_grid <- do.call(bigsnpr::snp_lassosum2, lasso_args)
params <- attr(beta_grid, "grid_param")
if (is.null(params) || nrow(params) != ncol(beta_grid)) {
  stop("lassosum2 did not return a usable grid_param attribute.", call. = FALSE)
}
valid_cols <- colSums(!is.finite(beta_grid)) == 0
if (!all(valid_cols)) {
  message("Skipping ", sum(!valid_cols), " lassosum2 models with non-finite betas.")
  beta_grid <- beta_grid[, valid_cols, drop = FALSE]
  params <- params[valid_cols, , drop = FALSE]
}
if (ncol(beta_grid) == 0L) {
  stop("No usable lassosum2 beta vectors were produced.", call. = FALSE)
}
params$model <- mapply(model_tag, params$delta, params$lambda)

score_impute_values <- training_imputation_means(G, ind_train, num_id, score_block_size)
message(
  "Using training-mean genotype imputation for lassosum2 PRS scoring; ",
  "observed ", format(attr(score_impute_values, "n_missing"), scientific = FALSE),
  " missing training genotypes while estimating per-variant means."
)
if (attr(score_impute_values, "n_without_observed") > 0L) {
  message(
    "Imputing ", attr(score_impute_values, "n_without_observed"),
    " variants with no observed training genotypes to dosage 0."
  )
}

score_match_for_map <- function(target_map, context) {
  target_match <- match_reference_to_target(matched, target_map)
  if (nrow(target_match) == 0L) {
    stop("No matched variants could be mapped to ", context, " genotype variants for scoring.", call. = FALSE)
  }
  target_match[order(target_match$ref_row), ]
}

write_scores <- function(G_target, target_map, ind_row, ids, out_path, context) {
  target_match <- score_match_for_map(target_map, context)
  ref_rows <- target_match$ref_row
  score_beta_grid <- beta_grid[ref_rows, , drop = FALSE] * target_match$target_beta_sign
  score_matrix <- score_with_training_mean_imputation(
    G_target,
    score_beta_grid,
    ind_row = ind_row,
    ind_col = target_match$target_num_id,
    impute_values = score_impute_values[ref_rows],
    block_size = score_block_size
  )
  if (is.null(dim(score_matrix))) {
    score_matrix <- matrix(score_matrix, ncol = 1L)
  }
  score_frames <- vector("list", nrow(params))
  for (j in seq_len(nrow(params))) {
    frame <- data.frame(
      sampleid = ids,
      disease = disease,
      fold = fold,
      model_scope = model_scope,
      delta = params$delta[[j]],
      lambda = params$lambda[[j]],
      model = params$model[[j]],
      sparsity = if ("sparsity" %in% names(params)) params$sparsity[[j]] else NA_real_,
      num_iter = if ("num_iter" %in% names(params)) params$num_iter[[j]] else NA_integer_,
      prs = as.numeric(score_matrix[, j])
    )
    names(frame)[[1L]] <- patient_col
    score_frames[[j]] <- frame
  }
  data.table::fwrite(do.call(rbind, score_frames), out_path)
  n_missing <- attr(score_matrix, "n_missing")
  attr(n_missing, "n_matched") <- nrow(target_match)
  n_missing
}

n_missing_scored_genotypes <- write_scores(G_score, score_map, ind_score, score_ids, score_out, "scoring")
n_score_matched <- attr(n_missing_scored_genotypes, "n_matched")
n_missing_fit_scored_genotypes <- NA_real_
n_fit_score_matched <- NA_integer_
if (!is.null(fit_score_out)) {
  n_missing_fit_scored_genotypes <- write_scores(
    G_fit_score,
    fit_score_map,
    ind_fit_score,
    fit_score_ids,
    fit_score_out,
    "model-fit scoring"
  )
  n_fit_score_matched <- attr(n_missing_fit_scored_genotypes, "n_matched")
}

summary <- data.frame(
  disease = disease,
  fold = fold,
  model_scope = model_scope,
  ld_reference_mode = ld_reference_mode,
  n_sumstats = nrow(sumstats),
  n_matched = nrow(matched),
  n_score_matched = n_score_matched,
  n_fit_score_matched = n_fit_score_matched,
  n_train = length(ind_train),
  n_scored = length(ind_score),
  n_fit_scored = if (!is.null(ind_fit_score)) length(ind_fit_score) else NA_integer_,
  n_models = nrow(params),
  n_beta_se_from_column = input_summary$n_beta_se_from_column,
  n_beta_se_from_p = input_summary$n_beta_se_from_p,
  n_eff_from_column = input_summary$n_eff_from_column,
  n_eff_from_default = input_summary$n_eff_from_default,
  score_imputation = "training_mean",
  score_block_size = score_block_size,
  n_missing_train_genotypes_for_imputation = attr(score_impute_values, "n_missing"),
  n_variants_without_observed_train_genotype = attr(score_impute_values, "n_without_observed"),
  n_missing_scored_genotypes = n_missing_scored_genotypes,
  n_missing_fit_scored_genotypes = n_missing_fit_scored_genotypes,
  delta_grid = paste(delta_grid, collapse = ","),
  nlambda = nlambda,
  lambda_min_ratio = lambda_min_ratio,
  dfmax = dfmax,
  maxiter = maxiter,
  tol = tol,
  join_by_pos = join_by_pos,
  reuse_ld_corr = reuse_ld_corr,
  ld_corr_reused = ld_corr_reused,
  ld_backing_prefix = ld_backing_prefix,
  ld_cache_metadata = ld_cache_paths(ld_backing_prefix)$metadata,
  ld_ref_map = ld_ref_map_path,
  ld_ref_dir = ld_ref_dir,
  ld_ref_pattern = ld_ref_pattern,
  score_out = score_out,
  fit_score_out = if (!is.null(fit_score_out)) fit_score_out else NA_character_
)
data.table::fwrite(summary, summary_out)

model <- list(
  disease = disease,
  fold = fold,
  params = params,
  summary = summary,
  matched = matched[, intersect(c("chr", "pos", "rsid", "a1", "a0", "_NUM_ID_", "ld_ref_num_id", "target_beta_sign"), names(matched))]
)
if (save_beta_grid) {
  model$beta_grid <- beta_grid
}
saveRDS(model, model_out)

message("Wrote lassosum2 scores to ", score_out)
if (!is.null(fit_score_out)) {
  message("Wrote lassosum2 model-fit scores to ", fit_score_out)
}
message("Wrote lassosum2 summary to ", summary_out)
