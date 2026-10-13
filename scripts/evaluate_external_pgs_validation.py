"""Primary external-PGS benchmark: official train as context, official val as query."""

from evaluate_external_pgs_cv import main

if __name__ == "__main__":
    main(protocol="official_train_val")
