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
  if (length(se_col) > 0L) {
    beta_se <- suppressWarnings(as.numeric(ss[[se_col[[1L]]]]))
  } else {
    message("SE column is absent; deriving approximate beta_se from BETA and P.")
    clipped_p <- pmin(pmax(pval, .Machine$double.xmin), 1)
    z <- stats::qnorm(clipped_p / 2, lower.tail = FALSE)
    beta_se <- abs(beta) / z
  }

  n_col <- intersect(c("N", "N_EFF", "NEFF", "N_eff", "n_eff"), names(ss))
  if (length(n_col) > 0L) {
    n_eff <- suppressWarnings(as.numeric(ss[[n_col[[1L]]]]))
  } else if (is.finite(default_n) && default_n > 0) {
    n_eff <- rep(default_n, nrow(ss))
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
  out[keep, ]
}

model_tag <- function(h2, p, sparse) {
  clean <- function(value) {
    gsub("\\+", "", gsub("\\.", "p", gsub("-", "m", format(value, scientific = FALSE, trim = TRUE))))
  }
  paste0("h2", clean(h2), "_p", clean(p), "_sparse", as.integer(sparse))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

for (pkg in c("data.table", "Matrix", "bigsparser", "bigsnpr")) {
  require_pkg(pkg)
}

disease <- arg_value(args, "disease", required = TRUE)
fold <- as.integer(arg_value(args, "fold", required = TRUE))
bed_prefix <- arg_value(args, "bed-prefix", required = TRUE)
harmonized <- arg_value(args, "harmonized", required = TRUE)
train_keep <- arg_value(args, "train-keep", required = TRUE)
score_keep <- arg_value(args, "score-keep", required = TRUE)
patient_col <- arg_value(args, "patient-col", "sampleid")
score_out <- arg_value(args, "score-out", required = TRUE)
summary_out <- arg_value(args, "summary-out", required = TRUE)
model_out <- arg_value(args, "model-out", required = TRUE)
ld_backing_prefix <- arg_value(args, "ld-backing-prefix", required = TRUE)
h2_grid <- parse_numeric_vector(arg_value(args, "h2-grid", required = TRUE))
p_grid <- parse_numeric_vector(arg_value(args, "p-grid", required = TRUE))
sparse_grid <- parse_bool_vector(arg_value(args, "sparse-grid", required = TRUE))
ld_window_cm <- as.numeric(arg_value(args, "ld-window-cm", "3"))
ld_window_mb <- as.numeric(arg_value(args, "ld-window-mb", "3"))
threads <- as.integer(arg_value(args, "threads", "1"))
default_n <- as.numeric(arg_value(args, "default-n", "NA"))
join_by_pos <- parse_bool(arg_value(args, "join-by-pos", "false"))
save_beta_grid <- parse_bool(arg_value(args, "save-beta-grid", "false"))

if (!is.finite(threads) || threads < 1L) {
  threads <- 1L
}

for (path in c(dirname(score_out), dirname(summary_out), dirname(model_out), dirname(ld_backing_prefix))) {
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

train_ids <- read_keep(train_keep)
score_ids <- read_keep(score_keep)
ind_train <- match(train_ids, sample_ids)
ind_score <- match(score_ids, sample_ids)
if (anyNA(ind_train)) {
  stop("Training keep samples absent from PLINK bed: ", paste(head(train_ids[is.na(ind_train)]), collapse = ", "), call. = FALSE)
}
if (anyNA(ind_score)) {
  stop("Scoring keep samples absent from PLINK bed: ", paste(head(score_ids[is.na(ind_score)]), collapse = ", "), call. = FALSE)
}

sumstats <- sumstats_for_ldpred2(harmonized, default_n = default_n)
if (nrow(sumstats) == 0L) {
  stop("No usable summary-statistic rows remain after LDpred2 validation.", call. = FALSE)
}
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
blocks <- list()
for (chr in sort(unique(matched$chr))) {
  idx <- which(matched$chr == chr)
  ind_col <- num_id[idx]
  genetic_pos <- map$genetic[ind_col]
  if (any(is.finite(genetic_pos)) && stats::sd(genetic_pos, na.rm = TRUE) > 0) {
    infos_pos <- genetic_pos
    window_size <- ld_window_cm
  } else {
    infos_pos <- map$pos[ind_col] / 1e6
    window_size <- ld_window_mb
  }
  blocks[[length(blocks) + 1L]] <- bigsnpr::snp_cor(
    G,
    ind.row = ind_train,
    ind.col = ind_col,
    infos.pos = infos_pos,
    size = window_size,
    ncores = threads
  )
}

corr_matrix <- if (length(blocks) == 1L) blocks[[1L]] else Matrix::bdiag(blocks)
unlink(paste0(ld_backing_prefix, c(".sbk", ".rds")))
corr <- bigsparser::as_SFBM(corr_matrix, backingfile = ld_backing_prefix)

params <- expand.grid(
  h2 = h2_grid,
  p = p_grid,
  sparse = sparse_grid,
  KEEP.OUT.ATTRS = FALSE
)
params <- params[, c("p", "h2", "sparse")]
params$model <- mapply(model_tag, params$h2, params$p, params$sparse)

beta_grid <- bigsnpr::snp_ldpred2_grid(corr, matched, params[, c("p", "h2", "sparse")], ncores = threads)
score_matrix <- bigsnpr::big_prodMat(
  G,
  beta_grid,
  ind.row = ind_score,
  ind.col = num_id,
  ncores = threads
)
if (is.null(dim(score_matrix))) {
  score_matrix <- matrix(score_matrix, ncol = 1L)
}

score_frames <- vector("list", nrow(params))
for (j in seq_len(nrow(params))) {
  frame <- data.frame(
    sampleid = score_ids,
    disease = disease,
    fold = fold,
    h2 = params$h2[[j]],
    p = params$p[[j]],
    sparse = as.integer(params$sparse[[j]]),
    model = params$model[[j]],
    prs = as.numeric(score_matrix[, j])
  )
  names(frame)[[1L]] <- patient_col
  score_frames[[j]] <- frame
}
scores <- do.call(rbind, score_frames)
data.table::fwrite(scores, score_out)

summary <- data.frame(
  disease = disease,
  fold = fold,
  n_sumstats = nrow(sumstats),
  n_matched = nrow(matched),
  n_train = length(ind_train),
  n_scored = length(ind_score),
  n_models = nrow(params),
  join_by_pos = join_by_pos,
  ld_window_cm = ld_window_cm,
  ld_window_mb = ld_window_mb,
  score_out = score_out
)
data.table::fwrite(summary, summary_out)

model <- list(
  disease = disease,
  fold = fold,
  params = params,
  summary = summary,
  matched = matched[, intersect(c("chr", "pos", "rsid", "a1", "a0", "_NUM_ID_"), names(matched))]
)
if (save_beta_grid) {
  model$beta_grid <- beta_grid
}
saveRDS(model, model_out)

message("Wrote LDpred2 scores to ", score_out)
message("Wrote LDpred2 summary to ", summary_out)
