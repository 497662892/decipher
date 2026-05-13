#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_long
#SBATCH --job-name=sbayesrc_resources
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sbayesrc_resources/%x-%J.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
cd "$REPO_DIR"

RESOURCE_DIR="${RESOURCE_DIR:-data/resources/sbayesrc}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$RESOURCE_DIR/downloads}"
EXTRACT_ARCHIVES="${EXTRACT_ARCHIVES:-1}"

DOWNLOAD_HM3="${DOWNLOAD_HM3:-0}"
DOWNLOAD_7M="${DOWNLOAD_7M:-1}"
DOWNLOAD_ANNOT="${DOWNLOAD_ANNOT:-0}"

BASE_URL="${BASE_URL:-https://gctbhub.cloud.edu.au/data/SBayesRC/resources/v2.0}"
HM3_URL="${HM3_URL:-$BASE_URL/LD/HapMap3/ukbEUR_HM3.zip}"
IMPUTED_URL="${IMPUTED_URL:-$BASE_URL/LD/Imputed/ukbEUR_Imputed.zip}"
ANNOT_URL="${ANNOT_URL:-$BASE_URL/Annotation/annot_baseline2.2.zip}"

HM3_BYTES="${HM3_BYTES:-3139079151}"
IMPUTED_BYTES="${IMPUTED_BYTES:-57939878464}"
ANNOT_BYTES="${ANNOT_BYTES:-307548449}"

mkdir -p "$DOWNLOAD_DIR" "$RESOURCE_DIR" logs/sbayesrc_resources

download_with_resume() {
  local url="$1"
  local out="$2"
  local expected_bytes="$3"

  mkdir -p "$(dirname "$out")"
  echo "Downloading $url -> $out"
  wget \
    --continue \
    --tries=20 \
    --waitretry=30 \
    --timeout=60 \
    --read-timeout=60 \
    --progress=dot:giga \
    -O "$out" \
    "$url"

  local actual_bytes
  actual_bytes="$(stat -c '%s' "$out")"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "Size check failed for $out: expected $expected_bytes bytes, got $actual_bytes bytes." >&2
    exit 1
  fi

  echo "Testing archive $out"
  unzip -t "$out" >/dev/null
}

extract_archive() {
  local archive="$1"
  local marker_dir="$2"

  if [[ "$EXTRACT_ARCHIVES" != "1" ]]; then
    echo "Skipping extraction for $archive because EXTRACT_ARCHIVES=$EXTRACT_ARCHIVES"
    return
  fi
  if [[ -d "$marker_dir" && -s "$marker_dir/ldm.info" && -s "$marker_dir/snp.info" ]]; then
    echo "Reusing extracted sBayesRC LD eigen directory: $marker_dir"
    return
  fi

  echo "Extracting $archive under $RESOURCE_DIR"
  unzip -q -o "$archive" -d "$RESOURCE_DIR"
}

extract_annotation() {
  local archive="$1"
  local out="$RESOURCE_DIR/annotation/annot_baseline2.2.txt"

  if [[ "$EXTRACT_ARCHIVES" != "1" ]]; then
    echo "Skipping annotation extraction because EXTRACT_ARCHIVES=$EXTRACT_ARCHIVES"
    return
  fi
  if [[ -s "$out" ]]; then
    echo "Reusing extracted sBayesRC annotation: $out"
    return
  fi

  echo "Extracting $archive under $RESOURCE_DIR"
  unzip -q -o "$archive" -d "$RESOURCE_DIR"
}

echo "Started sBayesRC resource download at $(date)"

if [[ "$DOWNLOAD_ANNOT" == "1" ]]; then
  download_with_resume "$ANNOT_URL" "$DOWNLOAD_DIR/annot_baseline2.2.zip" "$ANNOT_BYTES"
  extract_annotation "$DOWNLOAD_DIR/annot_baseline2.2.zip"
fi

if [[ "$DOWNLOAD_HM3" == "1" ]]; then
  download_with_resume "$HM3_URL" "$DOWNLOAD_DIR/ukbEUR_HM3.zip" "$HM3_BYTES"
  extract_archive "$DOWNLOAD_DIR/ukbEUR_HM3.zip" "$RESOURCE_DIR/ukbEUR_HM3"
fi

if [[ "$DOWNLOAD_7M" == "1" ]]; then
  download_with_resume "$IMPUTED_URL" "$DOWNLOAD_DIR/ukbEUR_Imputed.zip" "$IMPUTED_BYTES"
  extract_archive "$DOWNLOAD_DIR/ukbEUR_Imputed.zip" "$RESOURCE_DIR/ukbEUR_Imputed"
fi

echo "Finished sBayesRC resource download at $(date)"
