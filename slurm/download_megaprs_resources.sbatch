#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --partition=data_mover
#SBATCH --job-name=download_megaprs
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/resources/download_megaprs-%J.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
RESOURCE_DIR="${RESOURCE_DIR:-data/resources/megaprs}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$RESOURCE_DIR/downloads}"
LDAK_VERSION="${LDAK_VERSION:-6.2}"

cd "$REPO_DIR"
mkdir -p "$RESOURCE_DIR/BaselineLD" "$DOWNLOAD_DIR" logs/resources

if [[ ! -x "$RESOURCE_DIR/ldak${LDAK_VERSION}.linux" ]]; then
  wget -O "$RESOURCE_DIR/ldak${LDAK_VERSION}.linux" "https://github.com/dougspeed/LDAK/raw/main/ldak${LDAK_VERSION}.linux"
  chmod 755 "$RESOURCE_DIR/ldak${LDAK_VERSION}.linux"
fi
ln -sfn "ldak${LDAK_VERSION}.linux" "$RESOURCE_DIR/ldak"

if [[ ! -s "$RESOURCE_DIR/berisa.txt" ]]; then
  wget -O "$RESOURCE_DIR/berisa.txt" https://genetics.ghpc.au.dk/doug/berisa.txt
fi

if [[ ! -s "$RESOURCE_DIR/BaselineLD/BaselineLD1" ]]; then
  wget -O "$DOWNLOAD_DIR/BaselineLD.zip" https://genetics.ghpc.au.dk/doug/BaselineLD.zip
  unzip -o "$DOWNLOAD_DIR/BaselineLD.zip" -d "$RESOURCE_DIR/BaselineLD"
fi

test -s "$RESOURCE_DIR/berisa.txt" || { echo "Missing $RESOURCE_DIR/berisa.txt" >&2; exit 1; }
test -s "$RESOURCE_DIR/BaselineLD/BaselineLD1" || {
  echo "Missing $RESOURCE_DIR/BaselineLD/BaselineLD1 after extraction" >&2
  exit 1
}
for annot_index in $(seq 1 86); do
  test -s "$RESOURCE_DIR/BaselineLD/BaselineLD${annot_index}" || {
    echo "Missing $RESOURCE_DIR/BaselineLD/BaselineLD${annot_index} after extraction" >&2
    exit 1
  }
done
test -x "$RESOURCE_DIR/ldak" || { echo "Missing executable $RESOURCE_DIR/ldak" >&2; exit 1; }

echo "MegaPRS resources ready:"
echo "  ldak: $RESOURCE_DIR/ldak"
echo "  breakpoints: $RESOURCE_DIR/berisa.txt"
echo "  annotations: $RESOURCE_DIR/BaselineLD/BaselineLD[1..86]"
