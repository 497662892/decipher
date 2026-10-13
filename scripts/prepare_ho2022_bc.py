"""Prepare the two Ho 2022 breast-cancer components for the existing scoring pipeline."""

import argparse

from decipehr.ho_bc_weights import prepare_ho_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/external_pgs_ho2022_bc.yaml")
    parser.add_argument("--chain", default="data/resources/liftover/hg19ToHg38.over.chain.gz")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    prepare_ho_weights(args.manifest, args.chain, args.out_dir)
