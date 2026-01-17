"""
Script: Chunked load of GLORIA MR‑SUT CSV and split into Supply and Use blocks
- Input: huge MR‑SUT CSV (no headers), cannot be loaded fully
- Reads row chunks from CSV (CHUNK_SIZE rows at a time)
- For each chunk, extracts product rows and their columns (industries across all regions)
- Saves each chunk separately: supply_chunk_XXXX.pkl, use_chunk_XXXX.pkl
- Uses progress.txt to resume after last completed chunk
- At the end, stacks all chunks into full supply.pkl and use.pkl
"""

import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/"
INPUT_FILE = filepath + "GLORIA_MRIOs_59_2022/20240110_120secMother_AllCountries_002_T-Results_2022_059_Markup001(full).csv"
REGIONS_FILE = filepath + "IO-Table/regions_table.csv"
SECTORS_FILE = filepath + "IO-Table/sector_table_g.csv"

OUTPUT_DIR = Path(filepath + "IO-Table")
PROGRESS_FILE = OUTPUT_DIR / "progress.txt"

R = 164          # regions
I = 120          # industries per region
P = 120          # products per region
ROWS_PER_REGION = I + P       # 240
CHUNK_SIZE = 960              # CSV rows per chunk

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPERS
# ============================================================================

def load_metadata():
    """Load region and sector lists and MultiIndex (for final DF)."""
    regions_df = pd.read_csv(REGIONS_FILE)
    sectors_df = pd.read_csv(SECTORS_FILE)
    regions = regions_df["Region_acronyms"].tolist()
    sectors = sectors_df["sector"].tolist()
    multi_index = pd.MultiIndex.from_product(
        [regions, sectors], names=["region", "sector"]
    )
    return regions, sectors, multi_index

def get_last_chunk():
    """Return last completed chunk index (0‑based), or -1 if none."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return -1

def save_progress(chunk_idx: int):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(chunk_idx))

def chunk_filenames(chunk_idx: int):
    s = OUTPUT_DIR / f"supply_chunk_{chunk_idx:04d}.pkl"
    u = OUTPUT_DIR / f"use_chunk_{chunk_idx:04d}.pkl"
    return s, u

# ============================================================================
# CORE: EXTRACT FROM ONE ROW‑CHUNK
# ============================================================================

def extract_supply_use_from_chunk_blockwise(chunk_df, global_row_start):
    """
    Block-wise extraction:
      - Supply: product rows × industry cols (P × I) per (row_region, col_region)
      - Use:    same orientation (products × industries), for now equal to supply
    For each row_region fully contained in this chunk, slice its 120 I + 120 P rows
    in one go and build corresponding global blocks.
    """
    n_rows, n_cols = chunk_df.shape
    print("Chunk DataFrame shape:", chunk_df.shape)
    #print(chunk_df.head())  # keep short for sanity check

    # Your debug prints (fixed to iloc to avoid column/row confusion)
    try:
        print(f"first supply {chunk_df.iloc[global_row_start % n_rows, I]}")
        print(f"first use {chunk_df.iloc[(global_row_start + I) % n_rows, 0]}")
    except Exception as e:
        print(f"Debug print failed: {e}")

    expected_cols = R * ROWS_PER_REGION
    if n_cols < expected_cols:
        print(f"Warning: expected at least {expected_cols} columns, got {n_cols}")

    total_industries = R * I

    # How many regions are fully inside this chunk?
    first_global_row = global_row_start
    last_global_row = global_row_start + n_rows - 1

    first_region = first_global_row // ROWS_PER_REGION
    last_region = last_global_row // ROWS_PER_REGION

    supply_blocks_per_row_region = []
    use_blocks_per_row_region = []

    for r_row in range(first_region, last_region + 1):
        # rows belonging to this region in global index
        region_row_start_global = r_row * ROWS_PER_REGION
        region_row_end_global = region_row_start_global + ROWS_PER_REGION  # exclusive

        # check if region fully contained in this chunk
        if (region_row_start_global < first_global_row) or (region_row_end_global > last_global_row + 1):
            continue

        # convert to local chunk indices
        local_region_start = region_row_start_global - global_row_start
        local_region_end = local_region_start + ROWS_PER_REGION
        print(f"{local_region_start} local, {region_row_start_global} global row")

        # in this region block:
        #   industry rows: [0:I]
        #   product rows:  [I:I+P]
        local_ind_rows = slice(local_region_start, local_region_start + I)
        local_prod_rows = slice(local_region_start + I, local_region_start + I + P)

        # collect blocks across all column regions
        supply_row_blocks = []
        use_row_blocks = []

        for r_col in range(R):
            col_region_start = r_col * ROWS_PER_REGION
            # industry columns of destination region
            col_ind_cols = slice(col_region_start, col_region_start + I)
            # product columns of destination region (if ever needed)
            col_prod_cols = slice(col_region_start + I, col_region_start + I + P)

            # block slices
            # supply: products (rows) × industries (cols)
            sup_block = chunk_df.iloc[local_ind_rows, col_prod_cols].to_numpy()  # shape P×I
            # use: same orientation for now (products × industries)
            use_block = chunk_df.iloc[local_prod_rows, col_ind_cols].to_numpy()  # shape P×I

            supply_row_blocks.append(sup_block)
            use_row_blocks.append(use_block)

        # concatenate horizontally over all column regions:
        #   supply: P × (R*I)
        #   use:    P × (R*I)
        supply_blocks_per_row_region.append(np.hstack(supply_row_blocks))
        use_blocks_per_row_region.append(np.hstack(use_row_blocks))

    if not supply_blocks_per_row_region and not use_blocks_per_row_region:
        return (np.empty((0, total_industries)),
                np.empty((0, total_industries)))

    # stack vertically over all row regions contained in this chunk
    supply_chunk = np.vstack(supply_blocks_per_row_region) if supply_blocks_per_row_region else np.empty((0, total_industries))
    use_chunk = np.vstack(use_blocks_per_row_region) if use_blocks_per_row_region else np.empty((0, total_industries))

    print("supply_chunk sample:\n", supply_chunk[:3, :5])
    print("use_chunk sample:\n", use_chunk[:3, :5])

    del supply_blocks_per_row_region, use_blocks_per_row_region
    return supply_chunk, use_chunk

# ============================================================================
# MAIN LOOP (CHUNKED CSV LOADING + FINAL COMPILE)
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("GLORIA MR‑SUT: CSV → Supply & Use chunks (separate pickles)")
    print("=" * 70)

    regions, sectors, multi_index = load_metadata()
    print(f"Regions: {len(regions)}, Sectors: {len(sectors)}")
    print(f"Rows per region in SUT: {ROWS_PER_REGION}")
    print(f"Chunk size (rows): {CHUNK_SIZE}\n")

    last_chunk = get_last_chunk()
    print(last_chunk)
    if not last_chunk == 41:
        if last_chunk > 0:
            print(f"Resuming from chunk index {last_chunk + 1}")
            current_chunk_idx = last_chunk - 1
            global_row_start = current_chunk_idx*960

        else:
            print("Starting fresh.")
            current_chunk_idx = 0
            global_row_start = current_chunk_idx*960

        # pandas chunk reader
        reader = pd.read_csv(
            INPUT_FILE,
            header=None,
            chunksize=CHUNK_SIZE
        )


        for chunk_df in reader:
            current_chunk_idx += 1

            # skip chunks already processed
            if current_chunk_idx <= last_chunk:
                print(current_chunk_idx)
                global_row_start += len(chunk_df)
                continue

            print(f"\n--- Chunk {current_chunk_idx} "
                f"(global rows {global_row_start}:{global_row_start + len(chunk_df)}) ---")

            try:
                supply_chunk, use_chunk = extract_supply_use_from_chunk_blockwise(
                    chunk_df, global_row_start
                )

                print(f"  product rows in chunk: {supply_chunk.shape[0]}")

                # save this chunk separately
                s_file, u_file = chunk_filenames(current_chunk_idx)
                with open(s_file, "wb") as f:
                    pickle.dump(supply_chunk, f)
                with open(u_file, "wb") as f:
                    pickle.dump(use_chunk, f)
                print(f"  saved {s_file.name}, {u_file.name}")

                save_progress(current_chunk_idx)

                # advance row offset and free memory
                global_row_start += len(chunk_df)
                del chunk_df, supply_chunk, use_chunk
                gc.collect()
                print("  ✓ chunk processed and saved, RAM cleared")

            except MemoryError:
                print("MemoryError in this chunk; partial results saved. "
                    "Re‑run script to continue.")
                return
            except Exception as e:
                print(f"Error in chunk {current_chunk_idx}: {e}")
                print("Partial results saved. Fix issue and re‑run to continue.")
                return

    # ================== COMPILE ALL CHUNKS =======================
    print("\nAll chunks processed. Stacking supply/use matrices from chunk files...")

    s_files = sorted(OUTPUT_DIR.glob("supply_chunk_*.pkl"))
    u_files = sorted(OUTPUT_DIR.glob("use_chunk_*.pkl"))
    total_industries = R * I

    # ------------------------------------------------------------------
    # 1) Build and save SUPPLY first, then clear its data from memory
    # ------------------------------------------------------------------
    if not (OUTPUT_DIR / "supply.pkl").exists():

        print("\nBuilding full supply matrix...")

        supply_mats = []
        for fpath in s_files:
            print(f"  loading {fpath.name}")
            with open(fpath, "rb") as f:
                supply_mats.append(pickle.load(f))

        #print(supply_mats)
        supply_matrix = np.vstack(supply_mats) if supply_mats else np.empty((0, total_industries))

        print(f"Supply matrix shape: {supply_matrix.shape}")

        regions, sectors, multi_index = load_metadata()  # reload to be safe

        supply_df = pd.DataFrame(
            supply_matrix,
            index=multi_index[:supply_matrix.shape[0]],
            columns=multi_index[:total_industries]
        )

        with open(OUTPUT_DIR / "supply.pkl", "wb") as f:
            pickle.dump(supply_df, f)

        print("\nSaved:")
        print(f"  {OUTPUT_DIR / 'supply.pkl'}            (full DataFrame)")

        # Free supply-related memory (keep supply_df only if needed later)
        del supply_mats
        del supply_matrix
        gc.collect()

    # ------------------------------------------------------------------
    # 2) Then build and save USE, then clear its data from memory
    # ------------------------------------------------------------------
    print("\nBuilding full use matrix...")

    use_mats = []
    for fpath in u_files:
        print(f"  loading {fpath.name}")
        with open(fpath, "rb") as f:
            use_mats.append(pickle.load(f))
    
    use_matrix = np.vstack(use_mats) if use_mats else np.empty((0, total_industries))

    print(f"Use matrix shape:    {use_matrix.shape}")

    use_df = pd.DataFrame(
        use_matrix,
        index=multi_index[:use_matrix.shape[0]],
        columns=multi_index[:total_industries]
    )

    with open(OUTPUT_DIR / "use.pkl", "wb") as f:
        pickle.dump(use_df, f)

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'use.pkl'}               (full DataFrame)")

    # Free use-related memory (keep use_df only if needed later)
    del use_mats
    del use_matrix
    gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
