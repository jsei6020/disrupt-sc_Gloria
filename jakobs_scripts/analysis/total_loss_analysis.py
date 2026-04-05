#!/usr/bin/env python3

import os
import pandas as pd

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final8weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status2.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
SUPPLY_FILE = "/home/user/Documents/University/Master Thesis/supply_concentrations/region_sectors_data_1pct.csv"

OUT_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/loss_rankings_12cols_supplysensitive.csv"

# thousands USD -> trillions USD
SCALE_TO_TRILLIONS = 1e-9


def main():
    # --- 1. Load runs, sector table, supply shares ---
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]

    runs["region"] = runs["region_sector"].str.split(pat="_", n=1).str[0]
    runs["sector"] = runs["region_sector"].str.split(pat="_", n=1).str[1]

    sector_table = pd.read_csv(SECTOR_TABLE_FILE)
    sector_type_map = sector_table.set_index("sector")["type"]
    runs["sector_type"] = runs["sector"].map(sector_type_map)

    supply_raw = pd.read_csv(SUPPLY_FILE, header=None)
    cols = ["region_sector", "supply_share_exports_raw"]
    if supply_raw.shape[1] > 2:
        cols += [f"extra_{i}" for i in range(2, supply_raw.shape[1])]
    supply_raw.columns = cols
    supply_raw["supply_share_exports"] = pd.to_numeric(
        supply_raw["supply_share_exports_raw"], errors="coerce"
    )
    supply = supply_raw[["region_sector", "supply_share_exports"]]

    # --- 2. Read loss_summary for each run and compute loss in trillions ---
    records = []
    for _, row in runs.iterrows():
        region_sector = row["region_sector"]
        region = row["region"]
        sector = row["sector"]
        sector_type = row["sector_type"]
        folder = row["folder_name"]

        loss_path = os.path.join(BASE_PATH, folder, "loss_summary.csv")
        if os.path.exists(loss_path):
            df_loss = pd.read_csv(loss_path)
            households_loss_thousands = df_loss.loc[0, "households"]
        else:
            households_loss_thousands = float("nan")

        households_loss_trillions = households_loss_thousands * SCALE_TO_TRILLIONS

        records.append(
            {
                "region_sector": region_sector,
                "region": region,
                "sector": sector,
                "sector_type": sector_type,
                "households_loss_trillions": households_loss_trillions,
            }
        )

    all_rs = pd.DataFrame(records)

    # --- 3. Attach export supply shares ---
    all_rs = all_rs.merge(supply, on="region_sector", how="left")
    all_rs["supply_share_exports"] = pd.to_numeric(
        all_rs["supply_share_exports"], errors="coerce"
    )

    # --- 4. Aggregations & metrics ---

    # 4.1 Region–sector level
    # per-run: each row is one run
    rs_per_run = (
        all_rs[["region_sector", "households_loss_trillions"]]
        .copy()
        .sort_values("households_loss_trillions", ascending=False)
        .reset_index(drop=True)
    )
    rs_per_run["region_sector_n_runs"] = 1
    rs_per_run.rename(
        columns={"households_loss_trillions": "region_sector_loss_per_run_trillions"},
        inplace=True,
    )

    # per-supply: aggregate by region_sector, then normalize by its supply share
    rs_group = (
        all_rs.groupby("region_sector", as_index=False)
        .agg(
            total_loss_trillions=("households_loss_trillions", "sum"),
            n_runs=("households_loss_trillions", "size"),
            supply_share=("supply_share_exports", "first"),
        )
    )
    rs_group["region_sector_loss_per_supply_trillions"] = (
        rs_group["total_loss_trillions"] / rs_group["supply_share"]
    )
    rs_per_supply = (
        rs_group[["region_sector", "region_sector_loss_per_supply_trillions", "supply_share"]]
        .sort_values("region_sector_loss_per_supply_trillions", ascending=False)
        .reset_index(drop=True)
    )

    # 4.2 Region level
    reg_group = (
        all_rs.groupby("region", as_index=False)
        .agg(
            total_loss_trillions=("households_loss_trillions", "sum"),
            n_runs=("households_loss_trillions", "size"),
            sum_supply_share=("supply_share_exports", "sum"),
        )
    )
    # per-run average
    reg_group["region_avg_loss_per_run_trillions"] = (
        reg_group["total_loss_trillions"] / reg_group["n_runs"]
    )
    reg_per_run = (
        reg_group[["region", "region_avg_loss_per_run_trillions", "n_runs"]]
        .sort_values("region_avg_loss_per_run_trillions", ascending=False)
        .reset_index(drop=True)
    )
    # per-supply average (using the per-run avg divided by supply share sum)
    reg_group["region_avg_loss_per_supply_trillions"] = (
        reg_group["region_avg_loss_per_run_trillions"] / reg_group["sum_supply_share"]
    )
    reg_per_supply = (
        reg_group[
            ["region", "region_avg_loss_per_supply_trillions", "sum_supply_share"]
        ]
        .sort_values("region_avg_loss_per_supply_trillions", ascending=False)
        .reset_index(drop=True)
    )

    # 4.3 Sector level
    sec_group = (
        all_rs.groupby("sector", as_index=False)
        .agg(
            total_loss_trillions=("households_loss_trillions", "sum"),
            n_runs=("households_loss_trillions", "size"),
            sum_supply_share=("supply_share_exports", "sum"),
        )
    )
    sec_group["sector_avg_loss_per_run_trillions"] = (
        sec_group["total_loss_trillions"] / sec_group["n_runs"]
    )
    sec_per_run = (
        sec_group[["sector", "sector_avg_loss_per_run_trillions", "n_runs"]]
        .sort_values("sector_avg_loss_per_run_trillions", ascending=False)
        .reset_index(drop=True)
    )
    sec_group["sector_avg_loss_per_supply_trillions"] = (
        sec_group["sector_avg_loss_per_run_trillions"] / sec_group["sum_supply_share"]
    )
    sec_per_supply = (
        sec_group[
            ["sector", "sector_avg_loss_per_supply_trillions", "sum_supply_share"]
        ]
        .sort_values("sector_avg_loss_per_supply_trillions", ascending=False)
        .reset_index(drop=True)
    )

    # 4.4 Sector-type level
    st_group = (
        all_rs.groupby("sector_type", as_index=False)
        .agg(
            total_loss_trillions=("households_loss_trillions", "sum"),
            n_runs=("households_loss_trillions", "size"),
            sum_supply_share=("supply_share_exports", "sum"),
        )
    )
    st_group["sector_type_avg_loss_per_run_trillions"] = (
        st_group["total_loss_trillions"] / st_group["n_runs"]
    )
    st_per_run = (
        st_group[
            ["sector_type", "sector_type_avg_loss_per_run_trillions", "n_runs"]
        ]
        .sort_values("sector_type_avg_loss_per_run_trillions", ascending=False)
        .reset_index(drop=True)
    )
    st_group["sector_type_avg_loss_per_supply_trillions"] = (
        st_group["sector_type_avg_loss_per_run_trillions"]
        / st_group["sum_supply_share"]
    )
    st_per_supply = (
        st_group[
            [
                "sector_type",
                "sector_type_avg_loss_per_supply_trillions",
                "sum_supply_share",
            ]
        ]
        .sort_values("sector_type_avg_loss_per_supply_trillions", ascending=False)
        .reset_index(drop=True)
    )

    # --- 5. Align lengths and build final layout ---
    max_len = max(
        len(rs_per_run),
        len(rs_per_supply),
        len(reg_per_run),
        len(reg_per_supply),
        len(sec_per_run),
        len(sec_per_supply),
        len(st_per_run),
        len(st_per_supply),
    )

    rs_per_run = rs_per_run.reindex(range(max_len))
    rs_per_supply = rs_per_supply.reindex(range(max_len))
    reg_per_run = reg_per_run.reindex(range(max_len))
    reg_per_supply = reg_per_supply.reindex(range(max_len))
    sec_per_run = sec_per_run.reindex(range(max_len))
    sec_per_supply = sec_per_supply.reindex(range(max_len))
    st_per_run = st_per_run.reindex(range(max_len))
    st_per_supply = st_per_supply.reindex(range(max_len))

    final_df = pd.DataFrame(
        {
            # Region-sector block:
            # 1) name + per-run average + n_runs
            "region_sector_per_run_name": rs_per_run["region_sector"],
            "region_sector_avg_loss_per_run_trillions": rs_per_run[
                "region_sector_loss_per_run_trillions"
            ],
            "region_sector_n_runs": rs_per_run["region_sector_n_runs"],
            # 2) name + per-supply average + supply share
            "region_sector_per_supply_name": rs_per_supply["region_sector"],
            "region_sector_loss_per_supply_trillions": rs_per_supply[
                "region_sector_loss_per_supply_trillions"
            ],
            "region_sector_supply_share": rs_per_supply["supply_share"],
            "sep1": pd.NA,

            # Region block:
            # 1) name + per-run average + n_runs
            "region_per_run_name": reg_per_run["region"],
            "region_avg_loss_per_run_trillions": reg_per_run[
                "region_avg_loss_per_run_trillions"
            ],
            "region_n_runs": reg_per_run["n_runs"],
            # 2) name + per-supply average + sum of supply shares
            "region_per_supply_name": reg_per_supply["region"],
            "region_avg_loss_per_supply_trillions": reg_per_supply[
                "region_avg_loss_per_supply_trillions"
            ],
            "region_sum_supply_share": reg_per_supply["sum_supply_share"],
            "sep2": pd.NA,

            # Sector block:
            # 1) name + per-run average + n_runs
            "sector_per_run_name": sec_per_run["sector"],
            "sector_avg_loss_per_run_trillions": sec_per_run[
                "sector_avg_loss_per_run_trillions"
            ],
            "sector_n_runs": sec_per_run["n_runs"],
            # 2) name + per-supply average + sum of supply shares
            "sector_per_supply_name": sec_per_supply["sector"],
            "sector_avg_loss_per_supply_trillions": sec_per_supply[
                "sector_avg_loss_per_supply_trillions"
            ],
            "sector_sum_supply_share": sec_per_supply["sum_supply_share"],
            "sep3": pd.NA,

            # Sector-type block:
            # 1) name + per-run average + n_runs
            "sector_type_per_run_name": st_per_run["sector_type"],
            "sector_type_avg_loss_per_run_trillions": st_per_run[
                "sector_type_avg_loss_per_run_trillions"
            ],
            "sector_type_n_runs": st_per_run["n_runs"],
            # 2) name + per-supply average + sum of supply shares
            "sector_type_per_supply_name": st_per_supply["sector_type"],
            "sector_type_avg_loss_per_supply_trillions": st_per_supply[
                "sector_type_avg_loss_per_supply_trillions"
            ],
            "sector_type_sum_supply_share": st_per_supply["sum_supply_share"],
        }
    )

    final_df.to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    main()
