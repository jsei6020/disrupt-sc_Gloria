"""
GLORIA MR‑SUT → USE matrix at purchasers' prices (consumer prices)
- Works on T-Results files for valuations m=1..5 (basic, trade, transport, taxes, subsidies)
- For each chunk index:
    * load the same row chunk from each valuation sheet
    * extract the USE block (products × industries) as in your existing logic
    * sum across valuations to obtain a purchasers-price use_chunk_cp
    * save each summed chunk as use_cp_chunk_XXXX.pkl
- After all chunks, stack them into use_cp.pkl (DataFrame with MultiIndex)
"""

import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

basepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/"
YEAR = 2023
TAG  = "20251217_120secMother_AllCountries_002"
MRIO_DIR = Path(basepath, "GLORIA_MRIOs_60_2023")

# pattern:  ..._T-Results_2023_060_Markup00M(full).csv with M=1..5
def t_results_path(m: int) -> Path:
    return MRIO_DIR / f"{TAG}_T-Results_{YEAR}_060_Markup00{m}(full).csv"

REGIONS_FILE = Path(basepath, "IO-Table/regions_table.csv")
SECTORS_FILE = Path(basepath, "IO-Table/sector_table_g.csv")

OUTPUT_DIR = Path(basepath, "IO-Table")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

R = 164        # regions
I = 120        # industries per region
P = 120        # products per region
ROWS_PER_REGION = I + P
CHUNK_SIZE = 960    # rows per chunk

# ============================================================================
# HELPERS
# ============================================================================

def load_metadata():
    regions_df = pd.read_csv(REGIONS_FILE)
    sectors_df = pd.read_csv(SECTORS_FILE)
    regions = regions_df["Region_acronyms"].tolist()
    sectors = sectors_df["sector"].tolist()
    multi_index = pd.MultiIndex.from_product(
        [regions, sectors], names=["region", "sector"]
    )
    return regions, sectors, multi_index

def extract_use_from_chunk(chunk_df: pd.DataFrame, global_row_start: int) -> np.ndarray:
    """
    From one T-sheet row-chunk (any valuation), extract USE block:
      - For each fully contained row-region:
          * rows: products of that region (I ... I+P-1)
          * cols: industries of all column-regions
      - Returns stacked array of shape (n_products_all_regions_in_chunk, R*I)
    """
    n_rows, n_cols = chunk_df.shape
    total_industries = R * I

    first_global_row = global_row_start
    last_global_row = global_row_start + n_rows - 1

    first_region = first_global_row // ROWS_PER_REGION
    last_region = last_global_row // ROWS_PER_REGION

    use_blocks = []

    for r_row in range(first_region, last_region + 1):
        region_row_start_global = r_row * ROWS_PER_REGION
        region_row_end_global   = region_row_start_global + ROWS_PER_REGION

        # only take regions fully inside this chunk
        if (region_row_start_global < first_global_row) or (region_row_end_global > last_global_row + 1):
            continue

        local_region_start = region_row_start_global - global_row_start
        local_prod_rows = slice(local_region_start + I, local_region_start + I + P)

        row_blocks = []
        for r_col in range(R):
            col_region_start = r_col * ROWS_PER_REGION
            col_ind_cols = slice(col_region_start, col_region_start + I)
            block = chunk_df.iloc[local_prod_rows, col_ind_cols].to_numpy()
            row_blocks.append(block)

        if row_blocks:
            use_row = np.hstack(row_blocks)
            use_blocks.append(use_row)

    if not use_blocks:
        return np.empty((0, total_industries))

    use_chunk = np.vstack(use_blocks)
    del use_blocks
    return use_chunk

# ============================================================================
# MAIN
# ============================================================================

def build_use_purchasers_chunked():
    print("=" * 70)
    print("GLORIA MR‑SUT: build USE at purchasers' prices (consumer prices)")
    print("=" * 70)
    if False:

        # open readers for each valuation sheet
        readers = {
            m: pd.read_csv(
                t_results_path(m),
                header=None,
                chunksize=CHUNK_SIZE
            )
            for m in range(1, 6)
        }

        chunk_idx = -1
        global_row_start = 0
        use_chunk_files = []

        while True:
            # get next chunk from each valuation
            chunks = {}
            for m, reader in readers.items():
                try:
                    chunks[m] = next(reader)
                except StopIteration:
                    chunks[m] = None

            # if all are None, we are done
            if all(ch is None for ch in chunks.values()):
                break

            # sanity: require all valuations to have same row count while not None
            valid_chunks = [ch for ch in chunks.values() if ch is not None]
            n_rows = valid_chunks[0].shape[0]
            if any(ch is not None and ch.shape[0] != n_rows for ch in valid_chunks):
                raise RuntimeError("Valuation sheets have inconsistent chunk sizes.")

            chunk_idx += 1
            print(f"\n--- Chunk {chunk_idx} (global rows {global_row_start}:{global_row_start + n_rows}) ---")

            use_chunk_cp = None

            # loop over valuations and accumulate USE block
            for m in range(1, 6):
                chunk_df = chunks[m]
                if chunk_df is None:
                    continue

                use_chunk_m = extract_use_from_chunk(chunk_df, global_row_start)

                if use_chunk_m.size == 0:
                    continue

                if use_chunk_cp is None:
                    use_chunk_cp = use_chunk_m.astype(np.float64)
                else:
                    use_chunk_cp += use_chunk_m

                del use_chunk_m, chunk_df
                gc.collect()

            if use_chunk_cp is not None:
                fpath = OUTPUT_DIR / f"use_cp_chunk_{chunk_idx:04d}.pkl"
                with open(fpath, "wb") as f:
                    pickle.dump(use_chunk_cp, f)
                use_chunk_files.append(fpath)
                print(f"  saved {fpath.name}, shape {use_chunk_cp.shape}")
                del use_chunk_cp
                gc.collect()

            global_row_start += n_rows

        # stack all chunks into a DataFrame
        print("\nAll chunks processed. Stacking into full USE matrix at purchasers' prices...")

    # collect all chunk files
    use_chunk_files = sorted(OUTPUT_DIR.glob("use_cp_chunk_*.pkl"))

    total_industries = R * I
    mats = []
    for fpath in use_chunk_files:
        print("Loading", fpath.name)
        with open(fpath, "rb") as f:
            mats.append(pickle.load(f))

    use_matrix_cp = np.vstack(mats) if mats else np.empty((0, total_industries))

    regions, sectors, multi_index = load_metadata()
    use_cp_df = pd.DataFrame(
        use_matrix_cp,
        index=multi_index[:use_matrix_cp.shape[0]],
        columns=multi_index[:total_industries]
    )

    out_file = OUTPUT_DIR / "use_cp.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(use_cp_df, f)

    print(f"\nSaved full USE at purchasers' prices to {out_file}")
    print("Done.")

if __name__ == "__main__":
    build_use_purchasers_chunked()