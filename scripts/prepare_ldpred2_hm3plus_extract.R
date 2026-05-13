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

require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("R package '", pkg, "' is required.", call. = FALSE)
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

normalize_chr <- function(x) {
  x <- sub("^chr", "", as.character(x), ignore.case = TRUE)
  suppressWarnings(as.integer(x))
}

normalize_allele <- function(x) {
  toupper(as.character(x))
}

variant_key <- function(chr, pos, allele_a, allele_b) {
  a <- normalize_allele(allele_a)
  b <- normalize_allele(allele_b)
  lo <- ifelse(a <= b, a, b)
  hi <- ifelse(a <= b, b, a)
  paste(chr, pos, lo, hi, sep = ":")
}

json_scalar <- function(value) {
  if (is.numeric(value)) {
    return(as.character(value))
  }
  if (is.logical(value)) {
    return(ifelse(value, "true", "false"))
  }
  paste0('"', gsub('"', '\\"', as.character(value), fixed = TRUE), '"')
}

write_json <- function(values, path) {
  lines <- c(
    "{",
    paste0(
      "  \"", names(values), "\": ", vapply(values, json_scalar, character(1)),
      c(rep(",", length(values) - 1L), "")
    ),
    "}"
  )
  writeLines(lines, path, useBytes = TRUE)
}

read_pvar_header <- function(path) {
  con <- file(path, open = "rt")
  on.exit(close(con))
  repeat {
    line <- readLines(con, n = 1L, warn = FALSE)
    if (length(line) == 0L) {
      stop("Could not find a pvar header in: ", path, call. = FALSE)
    }
    if (!startsWith(line, "##")) {
      header <- strsplit(line, "\t", fixed = TRUE)[[1L]]
      return(list(line = line, names = header))
    }
  }
}

stream_pvar_matches <- function(
  path,
  header_line,
  cols,
  hm3_unique,
  chunk_size,
  progress_every = 5000000L
) {
  con <- file(path, open = "rt")
  on.exit(close(con))
  repeat {
    line <- readLines(con, n = 1L, warn = FALSE)
    if (length(line) == 0L) {
      stop("Could not find a pvar header in: ", path, call. = FALSE)
    }
    if (!startsWith(line, "##")) {
      break
    }
  }

  selected_cols <- unname(unlist(cols, use.names = FALSE))
  matches <- list()
  n_target_input <- 0L
  n_target_acgt_autosomes <- 0L
  next_progress <- progress_every
  repeat {
    lines <- readLines(con, n = chunk_size, warn = FALSE)
    if (length(lines) == 0L) {
      break
    }
    n_target_input <- n_target_input + length(lines)
    chunk <- data.table::fread(
      text = paste(c(header_line, lines), collapse = "\n"),
      select = selected_cols,
      data.table = FALSE,
      showProgress = FALSE,
      check.names = FALSE
    )
    target <- data.table::data.table(
      target_chr = normalize_chr(chunk[[cols$chr]]),
      target_pos = suppressWarnings(as.integer(chunk[[cols$pos]])),
      target_snp = as.character(chunk[[cols$id]]),
      target_ref = normalize_allele(chunk[[cols$ref]]),
      target_alt = normalize_allele(chunk[[cols$alt]])
    )
    target <- target[
      target_chr %in% seq_len(22L) &
        !is.na(target_pos) &
        !is.na(target_snp) &
        target_snp != "" &
        target_ref %in% c("A", "C", "G", "T") &
        target_alt %in% c("A", "C", "G", "T")
    ]
    n_target_acgt_autosomes <- n_target_acgt_autosomes + nrow(target)
    if (nrow(target) > 0L) {
      target[, key := variant_key(target_chr, target_pos, target_ref, target_alt)]
      matched <- merge(target, hm3_unique, by = "key", all = FALSE, sort = FALSE)
      if (nrow(matched) > 0L) {
        matches[[length(matches) + 1L]] <- matched
      }
    }
    if (n_target_input >= next_progress) {
      message("Processed ", n_target_input, " target pvar rows.")
      next_progress <- next_progress + progress_every
    }
  }
  list(
    matches = data.table::rbindlist(matches, use.names = TRUE),
    n_target_input = n_target_input,
    n_target_acgt_autosomes = n_target_acgt_autosomes
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
require_pkg("data.table")

hm3_plus_map <- arg_value(args, "hm3-plus-map", required = TRUE)
target_pvar <- arg_value(args, "target-pvar", required = TRUE)
build <- tolower(arg_value(args, "build", "hg38"))
out_prefix <- arg_value(args, "out-prefix", required = TRUE)
chunk_size <- as.integer(arg_value(args, "chunk-size", "500000"))
if (!is.finite(chunk_size) || chunk_size <= 0L) {
  stop("--chunk-size must be a positive integer.", call. = FALSE)
}

if (!file.exists(hm3_plus_map)) {
  stop("Missing HapMap3+ map: ", hm3_plus_map, call. = FALSE)
}
if (!file.exists(target_pvar)) {
  stop("Missing target pvar: ", target_pvar, call. = FALSE)
}

pos_col <- switch(
  build,
  hg38 = "pos_hg38",
  grch38 = "pos_hg38",
  hg19 = "pos",
  grch37 = "pos",
  hg18 = "pos_hg18",
  stop("Unsupported build: ", build, call. = FALSE)
)

hm3 <- readRDS(hm3_plus_map)
missing_hm3 <- setdiff(c("chr", pos_col, "a0", "a1", "rsid"), names(hm3))
if (length(missing_hm3) > 0L) {
  stop("HapMap3+ map is missing columns: ", paste(missing_hm3, collapse = ", "), call. = FALSE)
}
hm3 <- data.table::data.table(
  hm3_chr = normalize_chr(hm3$chr),
  hm3_pos = suppressWarnings(as.integer(hm3[[pos_col]])),
  hm3_a0 = normalize_allele(hm3$a0),
  hm3_a1 = normalize_allele(hm3$a1),
  hm3_rsid = as.character(hm3$rsid)
)
hm3 <- hm3[hm3_chr %in% seq_len(22L) & !is.na(hm3_pos)]
hm3[, key := variant_key(hm3_chr, hm3_pos, hm3_a0, hm3_a1)]
hm3[, key_count := .N, by = key]
hm3_unique <- hm3[key_count == 1L, .(key, hm3_rsid, hm3_chr, hm3_pos, hm3_a0, hm3_a1)]

pvar_header <- read_pvar_header(target_pvar)
pvar_cols <- data.frame(setNames(as.list(pvar_header$names), pvar_header$names), check.names = FALSE)
chr_col <- first_column(pvar_cols, c("#CHROM", "CHROM", "CHR"), "target pvar")
pos_col_target <- first_column(pvar_cols, c("POS", "BP"), "target pvar")
id_col <- first_column(pvar_cols, c("ID", "SNP"), "target pvar")
ref_col <- first_column(pvar_cols, c("REF", "A2"), "target pvar")
alt_col <- first_column(pvar_cols, c("ALT", "A1"), "target pvar")

streamed <- stream_pvar_matches(
  target_pvar,
  header_line = pvar_header$line,
  cols = list(
    chr = chr_col,
    pos = pos_col_target,
    id = id_col,
    ref = ref_col,
    alt = alt_col
  ),
  hm3_unique = hm3_unique,
  chunk_size = chunk_size
)
matched <- streamed$matches
if (nrow(matched) == 0L) {
  stop("No HapMap3+ variants matched target pvar.", call. = FALSE)
}
matched[, target_key_count := .N, by = key]
matched <- matched[target_key_count == 1L]
if (nrow(matched) == 0L) {
  stop("No HapMap3+ variants remain after removing duplicate target keys.", call. = FALSE)
}
matched$match_order <- ifelse(
  matched$target_ref == matched$hm3_a0 & matched$target_alt == matched$hm3_a1,
  "same",
  "reverse"
)
data.table::setorder(matched, target_chr, target_pos, target_snp)

snplist_path <- paste0(out_prefix, ".snplist")
audit_path <- paste0(out_prefix, ".audit.tsv.gz")
summary_path <- paste0(out_prefix, ".summary.json")
dir.create(dirname(snplist_path), recursive = TRUE, showWarnings = FALSE)

writeLines(matched$target_snp, snplist_path, useBytes = TRUE)
audit <- matched[, c(
  "target_snp", "target_chr", "target_pos", "target_ref", "target_alt",
  "hm3_rsid", "hm3_chr", "hm3_pos", "hm3_a0", "hm3_a1", "match_order"
)]
con <- gzfile(audit_path, open = "wt")
utils::write.table(audit, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)

summary <- list(
  build = build,
  n_hm3_plus_input = nrow(hm3),
  n_hm3_plus_unique_keys = nrow(hm3_unique),
  n_target_input = streamed$n_target_input,
  n_target_acgt_autosomes = streamed$n_target_acgt_autosomes,
  n_target_hm3_key_matches_pre_unique = nrow(streamed$matches),
  n_target_candidate_unique_keys = data.table::uniqueN(matched$key),
  n_matched = nrow(matched),
  n_same_order = sum(matched$match_order == "same"),
  n_reverse_order = sum(matched$match_order == "reverse"),
  snplist = snplist_path,
  audit = audit_path
)
write_json(summary, summary_path)

message("Wrote HapMap3+ target snplist to ", snplist_path)
message("Wrote HapMap3+ target audit to ", audit_path)
message("Wrote HapMap3+ target summary to ", summary_path)
