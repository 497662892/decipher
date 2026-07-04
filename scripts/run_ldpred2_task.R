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

parse_numeric_vector <- function(value) {
  parsed <- as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
  parsed <- parsed[is.finite(parsed)]
  if (length(parsed) == 0L) {
    stop("Numeric grid is empty: ", value, call. = FALSE)
  }
  parsed
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

parse_bool_vector <- function(value) {
  parts <- strsplit(value, ",", fixed = TRUE)[[1]]
  parsed <- vapply(parts, parse_bool, logical(1), USE.NAMES = FALSE)
  if (length(parsed) == 0L) {
    stop("Sparse grid is empty: ", value, call. = FALSE)
  }
  parsed
}

infer_beta_se_from_p <- function(beta, pval) {
  clipped_p <- pmin(pmax(pval, .Machine$double.xmin), 1)
  z <- stats::qnorm(clipped_p / 2, lower.tail = FALSE)
  inferred <- abs(beta) / z
  inferred[!is.finite(inferred) | inferred <= 0] <- NA_real_
  inferred
}

require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("R package '", pkg, "' is required for LDpred2 tasks.", call. = FALSE)
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

map_for_ldpred2 <- function(map) {
  chr_col <- first_column(map, c("chromosome", "chr", "CHR"), "PLINK map")
  rsid_col <- first_column(map, c("marker.ID", "rsid", "SNP", "ID"), "PLINK map")
  pos_col <- first_column(map, c("physical.pos", "pos", "BP"), "PLINK map")
  a1_col <- first_column(map, c("allele1", "a1", "A1"), "PLINK map")
  a0_col <- first_column(map, c("allele2", "a0", "A2"), "PLINK map")
  genetic_col <- NULL
  for (candidate in c("genetic.dist", "genetic_pos", "cM")) {
    if (candidate %in% names(map)) {
      genetic_col <- candidate
      break
    }
  }
  genetic <- rep(NA_real_, nrow(map))
  if (!is.null(genetic_col)) {
    genetic <- suppressWarnings(as.numeric(map[[genetic_col]]))
  }
  data.frame(
    chr = suppressWarnings(as.integer(map[[chr_col]])),
    rsid = as.character(map[[rsid_col]]),
    genetic = genetic,
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

ld_ref_file <- function(ld_ref_dir, ld_ref_pattern, chr) {
  file.path(ld_ref_dir, gsub("{chr}", as.character(chr), ld_ref_pattern, fixed = TRUE))
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

sumstats_for_ldpred2 <- function(path, default_n = NA_real_) {
  ss <- data.table::fread(path, data.table = FALSE, showProgress = FALSE)
  required <- c("SNP", "CHR", "BP", "A1", "A2", "BETA", "P")
  missing <- setdiff(required, names(ss))
  if (length(missing) > 0L) {
    stop("Harmonized sumstats are missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  se_col <- intersect(c("SE", "BETA_SE", "SEBETA", "STDERR"), names(ss))
  beta <- suppressWarnings(as.numeric(ss$BETA))
  pval <- suppressWarnings(as.numeric(ss$P))
  beta_se_from_p <- infer_beta_se_from_p(beta, pval)
  if (length(se_col) > 0L) {
    beta_se_from_col <- suppressWarnings(as.numeric(ss[[se_col[[1L]]]]))
    valid_col <- is.finite(beta_se_from_col) & beta_se_from_col > 0
    valid_p <- is.finite(beta_se_from_p) & beta_se_from_p > 0
    beta_se <- beta_se_from_col
    beta_se[!valid_col & valid_p] <- beta_se_from_p[!valid_col & valid_p]
    n_beta_se_from_col <- sum(valid_col, na.rm = TRUE)
    n_beta_se_from_p <- sum(!valid_col & valid_p, na.rm = TRUE)
    if (n_beta_se_from_p > 0L) {
      message("Derived beta_se from BETA and P for ", n_beta_se_from_p, " rows with missing/invalid SE.")
    }
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
      if (n_eff_from_default > 0L) {
        message("Filled n_eff from --default-n for ", n_eff_from_default, " rows with missing/invalid N.")
      }
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
    pos = suppressWarnings(as.numeric(ss$BP)),
    rsid = as.character(ss$SNP),
    a1 = toupper(as.character(ss$A1)),
    a0 = toupper(as.character(ss$A2)),
    beta = beta,
    beta_se = beta_se,
    n_eff = n_eff
  )
  keep <- stats::complete.cases(out) &
    out$chr %in% seq_len(22L) &
    is.finite(out$pos) &
    is.finite(out$beta) &
    is.finite(out$beta_se) &
    out$beta_se > 0 &
    is.finite(out$n_eff) &
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

genetic_positions_for_ld <- function(map, genetic_map_dir, genetic_map_type, threads, allow_mb_fallback) {
  chr <- map$chr
  pos <- map$pos
  if (!is.null(genetic_map_dir) && nzchar(genetic_map_dir)) {
    dir.create(genetic_map_dir, recursive = TRUE, showWarnings = FALSE)
    result <- tryCatch(
      {
        genetic_map <- bigsnpr::download_genetic_map(
          type = genetic_map_type,
          dir = genetic_map_dir,
          ncores = threads
        )
        bigsnpr::snp_asGeneticPos2(chr, pos, genetic_map)
      },
      error = function(err) {
        message("Could not compute genetic positions from ", genetic_map_type, ": ", conditionMessage(err))
        NULL
      }
    )
    if (!is.null(result)) {
      return(list(pos = as.numeric(result), unit = "cM"))
    }
  }

  genetic <- map$genetic
  if (any(is.finite(genetic)) && stats::sd(genetic, na.rm = TRUE) > 0) {
    return(list(pos = genetic, unit = "cM_from_bim"))
  }

  if (allow_mb_fallback) {
    message("Falling back to physical Mb positions for LD windows because genetic map is unavailable.")
    return(list(pos = map$pos / 1e6, unit = "Mb"))
  }

  stop(
    "Could not compute genetic positions for LDpred2. Provide --genetic-map-dir ",
    "with a downloadable/cached map, or pass --allow-mb-fallback true.",
    call. = FALSE
  )
}

ld_cache_paths <- function(ld_backing_prefix) {
  list(
    sbk = paste0(ld_backing_prefix, ".sbk"),
    rds = paste0(ld_backing_prefix, ".rds"),
    metadata = paste0(ld_backing_prefix, ".metadata.rds")
  )
}

file_fingerprint <- function(paths) {
  existing <- file.exists(paths)
  info <- file.info(paths[existing])
  out <- data.frame(
    path = paths,
    exists = existing,
    size = NA_real_,
    mtime = NA_real_
  )
  if (any(existing)) {
    out$size[existing] <- as.numeric(info$size)
    out$mtime[existing] <- as.numeric(info$mtime)
  }
  out
}

ld_cache_metadata <- function(
    disease,
    fold,
    model_scope,
    bed_prefix,
    harmonized,
    train_ids,
    matched,
    num_id,
    ld_positions,
    window_size,
    genetic_map_type,
    ld_window_cm,
    ld_window_mb) {
  list(
    version = 1L,
    disease = disease,
    fold = fold,
    model_scope = model_scope,
    bed_prefix = normalizePath(bed_prefix, mustWork = FALSE),
    bed_files = file_fingerprint(paste0(bed_prefix, c(".bed", ".bim", ".fam"))),
    harmonized = normalizePath(harmonized, mustWork = FALSE),
    train_ids = sort(unique(as.character(train_ids))),
    n_train = length(unique(train_ids)),
    n_matched = nrow(matched),
    num_id = as.integer(num_id),
    rsid = as.character(matched$rsid),
    chr = as.integer(matched$chr),
    pos = as.numeric(matched$pos),
    a1 = as.character(matched$a1),
    a0 = as.character(matched$a0),
    ld_pos = as.numeric(ld_positions$pos[num_id]),
    ld_position_unit = ld_positions$unit,
    genetic_map_type = genetic_map_type,
    window_size = as.numeric(window_size),
    ld_window_cm = as.numeric(ld_window_cm),
    ld_window_mb = as.numeric(ld_window_mb)
  )
}

same_vector <- function(left, right) {
  isTRUE(all.equal(left, right, check.attributes = FALSE, tolerance = 0))
}

same_ld_cache_metadata <- function(left, right) {
  if (is.null(left) || is.null(right)) {
    return(FALSE)
  }
  scalar_fields <- c(
    "version", "disease", "fold", "model_scope", "n_train", "n_matched",
    "ld_position_unit", "genetic_map_type", "window_size", "ld_window_cm",
    "ld_window_mb", "ld_reference_mode", "ld_ref_map", "ld_ref_dir",
    "ld_ref_pattern"
  )
  for (field in scalar_fields) {
    if (!identical(left[[field]], right[[field]])) {
      return(FALSE)
    }
  }
  vector_fields <- c(
    "train_ids", "num_id", "rsid", "chr", "pos", "a1", "a0", "ld_pos",
    "ld_ref_num_id", "target_beta_sign"
  )
  for (field in vector_fields) {
    if (!same_vector(left[[field]], right[[field]])) {
      return(FALSE)
    }
  }
  TRUE
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

save_ld_cache <- function(corr_matrix, ld_backing_prefix, metadata) {
  paths <- ld_cache_paths(ld_backing_prefix)
  unlink(unlist(paths))
  corr <- bigsparser::as_SFBM(corr_matrix, backingfile = ld_backing_prefix)
  saveRDS(corr, paths$rds)
  saveRDS(metadata, paths$metadata)
  corr
}

load_external_ld_cache <- function(ld_backing_prefix, expected_metadata, reuse_ld_corr) {
  load_ld_cache(ld_backing_prefix, expected_metadata, reuse_ld_corr = reuse_ld_corr)
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

model_tag <- function(h2, p, sparse) {
  clean <- function(value) {
    gsub("\\+", "", gsub("\\.", "p", gsub("-", "m", format(value, scientific = FALSE, trim = TRUE))))
  }
  paste0("h2", clean(h2), "_p", clean(p), "_sparse", as.integer(sparse))
}

auto_model_tag <- function(chain, p_init, sparse) {
  clean <- function(value) {
    gsub("\\+", "", gsub("\\.", "p", gsub("-", "m", format(value, scientific = FALSE, trim = TRUE))))
  }
  paste0("auto_chain", chain, "_pinit", clean(p_init), "_sparse", as.integer(sparse))
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

for (pkg in c("data.table", "Matrix", "bigsparser", "bigsnpr", "bigstatsr")) {
  require_pkg(pkg)
}

disease <- arg_value(args, "disease", required = TRUE)
fold <- as.integer(arg_value(args, "fold", required = TRUE))
bed_prefix <- arg_value(args, "bed-prefix", required = TRUE)
score_bed_prefix <- arg_value(args, "score-bed-prefix", NULL)
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
h2_grid <- parse_numeric_vector(arg_value(args, "h2-grid", required = TRUE))
p_grid <- parse_numeric_vector(arg_value(args, "p-grid", required = TRUE))
sparse_grid <- parse_bool_vector(arg_value(args, "sparse-grid", required = TRUE))
ldpred2_mode <- tolower(arg_value(args, "ldpred2-mode", "grid"))
auto_h2_init <- as.numeric(arg_value(args, "auto-h2-init", as.character(h2_grid[[1L]])))
auto_p_init <- parse_numeric_vector(arg_value(args, "auto-p-init", "0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.2"))
auto_burn_in <- as.integer(arg_value(args, "auto-burn-in", "500"))
auto_num_iter <- as.integer(arg_value(args, "auto-num-iter", "200"))
auto_sparse <- parse_bool(arg_value(args, "auto-sparse", "false"))
auto_verbose <- parse_bool(arg_value(args, "auto-verbose", "false"))
auto_allow_jump_sign <- parse_bool(arg_value(args, "auto-allow-jump-sign", "false"))
auto_shrink_corr <- as.numeric(arg_value(args, "auto-shrink-corr", "0.95"))
auto_use_mle <- parse_bool(arg_value(args, "auto-use-mle", "false"))
auto_seed <- as.integer(arg_value(args, "auto-seed", "42"))
ld_window_cm <- as.numeric(arg_value(args, "ld-window-cm", "3"))
ld_window_mb <- as.numeric(arg_value(args, "ld-window-mb", "3"))
genetic_map_dir <- arg_value(args, "genetic-map-dir", "data/resources/genetic_maps")
genetic_map_type <- arg_value(args, "genetic-map-type", "hg38_price")
allow_mb_fallback <- parse_bool(arg_value(args, "allow-mb-fallback", "false"))
threads <- as.integer(arg_value(args, "threads", "1"))
default_n <- as.numeric(arg_value(args, "default-n", "NA"))
join_by_pos <- parse_bool(arg_value(args, "join-by-pos", "false"))
save_beta_grid <- parse_bool(arg_value(args, "save-beta-grid", "false"))
reuse_ld_corr <- parse_bool(arg_value(args, "reuse-ld-corr", "true"))
model_scope <- arg_value(args, "model-scope", "cv")
ld_reference_mode <- tolower(arg_value(args, "ld-reference-mode", "target"))
ld_ref_map_path <- arg_value(args, "ld-ref-map", NULL)
ld_ref_dir <- arg_value(args, "ld-ref-dir", NULL)
ld_ref_pattern <- arg_value(args, "ld-ref-pattern", "LD_with_blocks_chr{chr}.rds")
score_block_size <- as.integer(arg_value(args, "score-block-size", "5000"))

if (is.null(fit_score_keep) != is.null(fit_score_out)) {
  stop("--fit-score-keep and --fit-score-out must be provided together.", call. = FALSE)
}
if (!ldpred2_mode %in% c("grid", "auto")) {
  stop("--ldpred2-mode must be 'grid' or 'auto'.", call. = FALSE)
}
if (!ld_reference_mode %in% c("target", "ukb_hm3_plus", "external")) {
  stop("--ld-reference-mode must be 'target', 'ukb_hm3_plus', or 'external'.", call. = FALSE)
}
if (ld_reference_mode != "target" && (is.null(ld_ref_map_path) || is.null(ld_ref_dir))) {
  stop("--ld-ref-map and --ld-ref-dir are required when --ld-reference-mode is not 'target'.", call. = FALSE)
}
if (!is.finite(auto_h2_init) || auto_h2_init <= 0) {
  stop("--auto-h2-init must be positive.", call. = FALSE)
}
if (!is.finite(auto_burn_in) || auto_burn_in < 0L) {
  stop("--auto-burn-in must be a non-negative integer.", call. = FALSE)
}
if (!is.finite(auto_num_iter) || auto_num_iter < 1L) {
  stop("--auto-num-iter must be a positive integer.", call. = FALSE)
}
if (!is.finite(auto_seed)) {
  auto_seed <- 42L
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

bed_file <- paste0(bed_prefix, ".bed")
if (!file.exists(bed_file)) {
  stop("Missing PLINK bed file: ", bed_file, call. = FALSE)
}

rds_file <- paste0(bed_prefix, ".rds")
if (!file.exists(rds_file)) {
  message("Creating bigSNP backing files from ", bed_file)
  rds_file <- bigsnpr::snp_readBed(bed_file, backingfile = bed_prefix)
}
obj <- bigsnpr::snp_attach(rds_file)
G <- obj$genotypes
sample_ids <- fam_sample_ids(obj$fam)
map <- map_for_ldpred2(obj$map)

use_score_bed <- !is.null(score_bed_prefix) && score_bed_prefix != bed_prefix
score_obj <- obj
G_score <- G
score_sample_ids <- sample_ids
score_map <- map
if (use_score_bed) {
  score_bed_file <- paste0(score_bed_prefix, ".bed")
  if (!file.exists(score_bed_file)) {
    stop("Missing PLINK score bed file: ", score_bed_file, call. = FALSE)
  }
  score_rds_file <- paste0(score_bed_prefix, ".rds")
  if (!file.exists(score_rds_file)) {
    message("Creating score bigSNP backing files from ", score_bed_file)
    score_rds_file <- bigsnpr::snp_readBed(score_bed_file, backingfile = score_bed_prefix)
  }
  score_obj <- bigsnpr::snp_attach(score_rds_file)
  G_score <- score_obj$genotypes
  score_sample_ids <- fam_sample_ids(score_obj$fam)
  score_map <- map_for_ldpred2(score_obj$map)
}

train_ids <- read_keep(train_keep)
score_ids <- read_keep(score_keep)
ind_train <- match(train_ids, sample_ids)
ind_score <- match(score_ids, score_sample_ids)
if (anyNA(ind_train)) {
  stop("Training keep samples absent from PLINK bed: ", paste(head(train_ids[is.na(ind_train)]), collapse = ", "), call. = FALSE)
}
if (anyNA(ind_score)) {
  stop("Scoring keep samples absent from PLINK score bed: ", paste(head(score_ids[is.na(ind_score)]), collapse = ", "), call. = FALSE)
}
fit_score_ids <- NULL
ind_fit_score <- NULL
if (!is.null(fit_score_keep)) {
  fit_score_ids <- read_keep(fit_score_keep)
  ind_fit_score <- match(fit_score_ids, sample_ids)
  if (anyNA(ind_fit_score)) {
    stop(
      "Model-fit scoring keep samples absent from PLINK bed: ",
      paste(head(fit_score_ids[is.na(ind_fit_score)]), collapse = ", "),
      call. = FALSE
    )
  }
}

sumstats <- sumstats_for_ldpred2(harmonized, default_n = default_n)
input_summary <- attr(sumstats, "input_summary")
if (nrow(sumstats) == 0L) {
  stop("No usable summary-statistic rows remain after LDpred2 validation.", call. = FALSE)
}
ld_ref_map <- NULL
if (ld_reference_mode == "target") {
  matched <- bigsnpr::snp_match(
    sumstats,
    map[, c("chr", "rsid", "pos", "a1", "a0")],
    strand_flip = TRUE,
    join_by_pos = join_by_pos
  )
  if (nrow(matched) == 0L) {
    stop("No variants matched between harmonized sumstats and PLINK bed.", call. = FALSE)
  }
  matched <- matched[order(matched$chr, matched$pos), ]

  num_id <- matched[["_NUM_ID_"]]
  matched$target_beta_sign <- 1
  ld_positions <- genetic_positions_for_ld(
    map,
    genetic_map_dir = genetic_map_dir,
    genetic_map_type = genetic_map_type,
    threads = threads,
    allow_mb_fallback = allow_mb_fallback
  )
  valid_ld_pos <- is.finite(ld_positions$pos[num_id])
  n_dropped_missing_ld_pos <- sum(!valid_ld_pos)
  if (!all(valid_ld_pos)) {
    message("Dropping ", n_dropped_missing_ld_pos, " matched variants with missing LD map positions.")
    matched <- matched[valid_ld_pos, ]
    num_id <- matched[["_NUM_ID_"]]
  }
  if (nrow(matched) == 0L) {
    stop("No matched variants remain after LD map-position filtering.", call. = FALSE)
  }
  if (ld_positions$unit %in% c("cM", "cM_from_bim")) {
    window_size <- ld_window_cm
  } else {
    window_size <- ld_window_mb
  }
} else {
  ld_ref_map <- map_for_ld_reference(ld_ref_map_path)
  matched <- prepare_reference_matched(
    sumstats = sumstats,
    target_map = map,
    ld_ref_map = ld_ref_map,
    join_by_pos = join_by_pos
  )
  num_id <- matched[["_NUM_ID_"]]
  ld_positions <- list(pos = rep(NA_real_, nrow(map)), unit = "external_ld_reference")
  n_dropped_missing_ld_pos <- 0L
  window_size <- NA_real_
}

score_beta_row <- seq_len(nrow(matched))
score_num_id <- num_id
score_target_beta_sign <- matched$target_beta_sign
if (use_score_bed) {
  score_match <- match_reference_to_target(matched, score_map)
  if (nrow(score_match) == 0L) {
    stop("No matched LDpred2 variants could be mapped to the PLINK score bed.", call. = FALSE)
  }
  if (nrow(score_match) < nrow(matched)) {
    message(
      "Mapped ", nrow(score_match), " of ", nrow(matched),
      " LDpred2 variants to the PLINK score bed."
    )
  }
  score_beta_row <- score_match$ref_row
  score_num_id <- score_match$target_num_id
  score_target_beta_sign <- score_match$target_beta_sign
}

ld_metadata <- ld_cache_metadata(
  disease = disease,
  fold = fold,
  model_scope = model_scope,
  bed_prefix = bed_prefix,
  harmonized = harmonized,
  train_ids = train_ids,
  matched = matched,
  num_id = num_id,
  ld_positions = ld_positions,
  window_size = window_size,
  genetic_map_type = genetic_map_type,
  ld_window_cm = ld_window_cm,
  ld_window_mb = ld_window_mb
)
ld_metadata$version <- 2L
ld_metadata$ld_reference_mode <- ld_reference_mode
ld_metadata$ld_ref_map <- if (!is.null(ld_ref_map_path)) normalizePath(ld_ref_map_path, mustWork = FALSE) else NA_character_
ld_metadata$ld_ref_dir <- if (!is.null(ld_ref_dir)) normalizePath(ld_ref_dir, mustWork = FALSE) else NA_character_
ld_metadata$ld_ref_pattern <- ld_ref_pattern
ld_metadata$ld_ref_num_id <- if ("ld_ref_num_id" %in% names(matched)) as.integer(matched$ld_ref_num_id) else NA_integer_
ld_metadata$target_beta_sign <- as.numeric(matched$target_beta_sign)
corr <- load_ld_cache(ld_backing_prefix, ld_metadata, reuse_ld_corr = reuse_ld_corr)
ld_corr_reused <- !is.null(corr)
if (!ld_corr_reused) {
  if (ld_reference_mode == "target") {
    message(
      "Building LD correlation cache for ", nrow(matched), " variants and ",
      length(ind_train), " training samples at ", ld_backing_prefix
    )
    blocks <- list()
    for (chr in sort(unique(matched$chr))) {
      idx <- which(matched$chr == chr)
      ind_col <- num_id[idx]
      message("Computing LD correlations for chr", chr, " with ", length(ind_col), " variants.")
      blocks[[length(blocks) + 1L]] <- bigsnpr::snp_cor(
        G,
        ind.row = ind_train,
        ind.col = ind_col,
        infos.pos = ld_positions$pos[ind_col],
        size = window_size,
        ncores = threads
      )
    }

    corr_matrix <- if (length(blocks) == 1L) blocks[[1L]] else Matrix::bdiag(blocks)
    corr <- save_ld_cache(corr_matrix, ld_backing_prefix, ld_metadata)
    rm(corr_matrix, blocks)
  } else {
    message("Building LD correlation cache from external UKB LD reference at ", ld_backing_prefix)
    corr <- build_external_ld_cache(
      ld_ref_map = ld_ref_map,
      matched = matched,
      ld_ref_dir = ld_ref_dir,
      ld_ref_pattern = ld_ref_pattern,
      ld_backing_prefix = ld_backing_prefix,
      metadata = ld_metadata
    )
  }
  gc()
}

auto_summary <- NULL
if (ldpred2_mode == "grid") {
  params <- expand.grid(
    h2 = h2_grid,
    p = p_grid,
    sparse = sparse_grid,
    KEEP.OUT.ATTRS = FALSE
  )
  params <- params[, c("p", "h2", "sparse")]
  params$model <- mapply(model_tag, params$h2, params$p, params$sparse)
  beta_grid <- bigsnpr::snp_ldpred2_grid(corr, matched, params[, c("p", "h2", "sparse")], ncores = threads)
} else {
  set.seed(auto_seed)
  auto_args <- list(
    corr = corr,
    df_beta = matched,
    h2_init = auto_h2_init,
    vec_p_init = auto_p_init,
    burn_in = auto_burn_in,
    num_iter = auto_num_iter,
    sparse = auto_sparse,
    verbose = auto_verbose,
    ncores = threads
  )
  auto_args <- optional_formal_arg(bigsnpr::snp_ldpred2_auto, auto_args, "allow_jump_sign", auto_allow_jump_sign)
  auto_args <- optional_formal_arg(bigsnpr::snp_ldpred2_auto, auto_args, "shrink_corr", auto_shrink_corr)
  auto_args <- optional_formal_arg(bigsnpr::snp_ldpred2_auto, auto_args, "use_MLE", auto_use_mle)
  multi_auto <- do.call(bigsnpr::snp_ldpred2_auto, auto_args)

  beta_cols <- list()
  param_rows <- list()
  summary_rows <- list()
  for (chain in seq_along(multi_auto)) {
    fit <- multi_auto[[chain]]
    p_init <- auto_p_init[[chain]]
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      chain = chain,
      p_init = p_init,
      p_est = fit$p_est,
      h2_est = fit$h2_est,
      has_beta_est = !is.null(fit$beta_est),
      has_beta_est_sparse = !is.null(fit$beta_est_sparse)
    )
    beta_est <- if (!is.null(fit$beta_est)) as.numeric(fit$beta_est) else NULL
    if (!is.null(beta_est) && length(beta_est) == nrow(matched) && all(is.finite(beta_est))) {
      beta_cols[[length(beta_cols) + 1L]] <- beta_est
      param_rows[[length(param_rows) + 1L]] <- data.frame(
        p = fit$p_est,
        h2 = fit$h2_est,
        sparse = FALSE,
        model = auto_model_tag(chain, p_init, FALSE),
        p_init = p_init,
        chain = chain
      )
    } else {
      message("Skipping LDpred2-auto dense chain ", chain, " because beta_est is missing or non-finite.")
    }
    beta_est_sparse <- if (!is.null(fit$beta_est_sparse)) as.numeric(fit$beta_est_sparse) else NULL
    if (!is.null(beta_est_sparse) && length(beta_est_sparse) == nrow(matched) && all(is.finite(beta_est_sparse))) {
      beta_cols[[length(beta_cols) + 1L]] <- beta_est_sparse
      param_rows[[length(param_rows) + 1L]] <- data.frame(
        p = fit$p_est,
        h2 = fit$h2_est,
        sparse = TRUE,
        model = auto_model_tag(chain, p_init, TRUE),
        p_init = p_init,
        chain = chain
      )
    } else if (auto_sparse) {
      message("Skipping LDpred2-auto sparse chain ", chain, " because beta_est_sparse is missing or non-finite.")
    }
  }
  if (length(beta_cols) == 0L) {
    stop("No usable LDpred2-auto beta vectors were produced.", call. = FALSE)
  }
  beta_grid <- do.call(cbind, beta_cols)
  params <- do.call(rbind, param_rows)
  auto_summary <- do.call(rbind, summary_rows)
}

score_impute_values <- training_imputation_means(G, ind_train, num_id, score_block_size)
message(
  "Using training-mean genotype imputation for LDpred2 PRS scoring; ",
  "observed ", format(attr(score_impute_values, "n_missing"), scientific = FALSE),
  " missing training genotypes while estimating per-variant means."
)
if (attr(score_impute_values, "n_without_observed") > 0L) {
  message(
    "Imputing ", attr(score_impute_values, "n_without_observed"),
    " variants with no observed training genotypes to dosage 0."
  )
}

write_scores <- function(G_source, ind_row, ids, out_path, ind_col, beta_row, beta_sign) {
  score_beta_grid <- beta_grid[beta_row, , drop = FALSE] * beta_sign
  score_matrix <- score_with_training_mean_imputation(
    G_source,
    score_beta_grid,
    ind_row = ind_row,
    ind_col = ind_col,
    impute_values = score_impute_values[beta_row],
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
      h2 = params$h2[[j]],
      p = params$p[[j]],
      sparse = as.integer(params$sparse[[j]]),
      model = params$model[[j]],
      prs = as.numeric(score_matrix[, j])
    )
    if ("p_init" %in% names(params)) {
      frame$p_init <- params$p_init[[j]]
    }
    if ("chain" %in% names(params)) {
      frame$chain <- params$chain[[j]]
    }
    names(frame)[[1L]] <- patient_col
    score_frames[[j]] <- frame
  }
  data.table::fwrite(do.call(rbind, score_frames), out_path)
  attr(score_matrix, "n_missing")
}

n_missing_scored_genotypes <- write_scores(
  G_score,
  ind_score,
  score_ids,
  score_out,
  score_num_id,
  score_beta_row,
  score_target_beta_sign
)
n_missing_fit_scored_genotypes <- NA_real_
if (!is.null(fit_score_out)) {
  n_missing_fit_scored_genotypes <- write_scores(
    G,
    ind_fit_score,
    fit_score_ids,
    fit_score_out,
    num_id,
    seq_len(nrow(matched)),
    matched$target_beta_sign
  )
}

summary <- data.frame(
  disease = disease,
  fold = fold,
  model_scope = model_scope,
  ldpred2_mode = ldpred2_mode,
  ld_reference_mode = ld_reference_mode,
  n_sumstats = nrow(sumstats),
  n_matched = nrow(matched),
  n_dropped_missing_ld_pos = n_dropped_missing_ld_pos,
  n_train = length(ind_train),
  n_scored = length(ind_score),
  n_fit_scored = if (!is.null(ind_fit_score)) length(ind_fit_score) else NA_integer_,
  n_score_matched = length(score_num_id),
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
  join_by_pos = join_by_pos,
  ld_position_unit = ld_positions$unit,
  genetic_map_type = genetic_map_type,
  ld_window_cm = ld_window_cm,
  ld_window_mb = ld_window_mb,
  reuse_ld_corr = reuse_ld_corr,
  ld_corr_reused = ld_corr_reused,
  ld_backing_prefix = ld_backing_prefix,
  bed_prefix = bed_prefix,
  score_bed_prefix = if (use_score_bed) score_bed_prefix else bed_prefix,
  ld_cache_metadata = ld_cache_paths(ld_backing_prefix)$metadata,
  ld_ref_map = if (!is.null(ld_ref_map_path)) ld_ref_map_path else NA_character_,
  ld_ref_dir = if (!is.null(ld_ref_dir)) ld_ref_dir else NA_character_,
  ld_ref_pattern = if (!is.null(ld_ref_pattern)) ld_ref_pattern else NA_character_,
  auto_h2_init = if (ldpred2_mode == "auto") auto_h2_init else NA_real_,
  auto_burn_in = if (ldpred2_mode == "auto") auto_burn_in else NA_integer_,
  auto_num_iter = if (ldpred2_mode == "auto") auto_num_iter else NA_integer_,
  auto_sparse = if (ldpred2_mode == "auto") auto_sparse else NA,
  auto_shrink_corr = if (ldpred2_mode == "auto") auto_shrink_corr else NA_real_,
  auto_allow_jump_sign = if (ldpred2_mode == "auto") auto_allow_jump_sign else NA,
  auto_use_mle = if (ldpred2_mode == "auto") auto_use_mle else NA,
  auto_seed = if (ldpred2_mode == "auto") auto_seed else NA_integer_,
  score_out = score_out,
  fit_score_out = if (!is.null(fit_score_out)) fit_score_out else NA_character_
)
data.table::fwrite(summary, summary_out)

model <- list(
  disease = disease,
  fold = fold,
  params = params,
  summary = summary,
  matched = matched[, intersect(
    c("chr", "pos", "rsid", "a1", "a0", "_NUM_ID_", "ld_ref_num_id", "target_beta_sign"),
    names(matched)
  )]
)
if (!is.null(auto_summary)) {
  model$auto_summary <- auto_summary
}
if (save_beta_grid) {
  model$beta_grid <- beta_grid
}
saveRDS(model, model_out)

message("Wrote LDpred2 scores to ", score_out)
if (!is.null(fit_score_out)) {
  message("Wrote LDpred2 model-fit scores to ", fit_score_out)
}
message("Wrote LDpred2 summary to ", summary_out)
