#!/usr/bin/env python3
import pandas as pd

# ==== USER SETTINGS ====
MRIO_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"   # set this
MRIO_FORMAT = "pkl"                    # "pkl" or "parquet"
HHI_PATH = "/home/user/Documents/University/Master Thesis/hhi_excl_self_all_region_sector.csv"  # set this
OUTPUT_REGION_PATH = "/home/user/Documents/University/Master Thesis/hhi_region_weighted.csv"     # set this
# ========================


def compute_region_weighted_hhi(mrio_df: pd.DataFrame, hhi_df: pd.DataFrame):
    """
    Combine existing (region, sector) HHIs with MRIO-based
    consumption shares to get a region-level weighted HHI.

    Assumes:
      - mrio_df columns: MultiIndex (consuming_region, sector)
      - hhi_df has columns: consuming_region, sector, hhi_excl_self
    """

    if not isinstance(mrio_df.columns, pd.MultiIndex) or mrio_df.columns.nlevels < 2:
        raise ValueError("MRIO columns must be a MultiIndex (region, sector).")

    # 1) Column sums: consumption of each (region, sector)
    col_sums = mrio_df.sum(axis=0)  # Series indexed by (region, sector)

    cons_df = col_sums.reset_index()
    cons_df.columns = ["consuming_region", "sector", "consumption"]

    # 2) Total consumption per region
    region_total_cons = cons_df.groupby("consuming_region")["consumption"].sum()

    # 3) Merge consumption into HHI file
    merged = hhi_df.merge(cons_df, on=["consuming_region", "sector"], how="left")

    # 4) Compute region weights
    merged["region_total_cons"] = merged["consuming_region"].map(region_total_cons)
    merged["region_weight"] = merged["consumption"] / merged["region_total_cons"]

    # 5) Region-level weighted HHI
    region_weighted = (
        merged.groupby("consuming_region")
        .apply(lambda g: (g["hhi_excl_self"] * g["region_weight"]).sum())
        .reset_index(name="weighted_hhi_excl_self")
        .sort_values("weighted_hhi_excl_self", ascending=False)
    )

    return merged, region_weighted


def main():
    # load MRIO
    if MRIO_FORMAT == "pkl":
        mrio_df = pd.read_pickle(MRIO_PATH)
    elif MRIO_FORMAT == "parquet":
        mrio_df = pd.read_parquet(MRIO_PATH)
    else:
        raise ValueError("MRIO_FORMAT must be 'pkl' or 'parquet'.")

    # load precomputed HHI
    hhi_df = pd.read_csv(HHI_PATH)

    merged, region_weighted = compute_region_weighted_hhi(mrio_df, hhi_df)

    # optional: full merged file with weights
    merged.to_csv(HHI_PATH.replace(".csv", "_with_weights.csv"), index=False)

    # region-level weighted HHI
    region_weighted.to_csv(OUTPUT_REGION_PATH, index=False)
    print(f"Written {OUTPUT_REGION_PATH}")


if __name__ == "__main__":
    main()
