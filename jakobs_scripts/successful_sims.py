import re
from datetime import datetime
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# paths (adjust base if needed)
# ------------------------------------------------------------
BASE = Path("/home/user/Documents/University/Master Thesis")
LOG_FILE = BASE / "disrupt-sc/output/Global4/final/logs_final/master.log"
GLOBAL4_FINAL = BASE / "disrupt-sc/output/Global4/final"
GLORIA = BASE / "Databases/Gloria"
SECTORS_FILE = GLORIA / "IO-Table/sector_table_g.csv"
TARGET_FILE = "loss_per_region_sector_time.csv"

# ------------------------------------------------------------
# 1) load sector table: short/sector label → full sector name
# ------------------------------------------------------------
def load_sector_pairs(path: Path):
    """
    Returns a list of (short_label, full_name) pairs from sector_table_g.csv.
    Assumes a column 'sector' plus a 'name'/'description'-like column.
    """
    df = pd.read_csv(path)
    cols = df.columns.str.lower()

    # column with sector labels (e.g. 'Growing beverage crops (coffee, tea etc)')
    sector_col = df.columns[cols == "sector"][0]

    # column with full descriptive name (or fall back to sector if nothing else)
    name_candidates = df.columns[cols.str.contains("name") | cols.str.contains("description")]
    full_col = name_candidates[0] if len(name_candidates) > 0 else sector_col

    pairs = list(zip(df[sector_col].astype(str), df[full_col].astype(str)))
    return pairs

def resolve_full_sector(sector_from_log: str, sector_pairs):
    """
    Map a chopped sector string from master.log to the full sector name.
    Strategy:
      1) exact match on short label
      2) full name startswith(log string)
      3) full name contains log string
    """
    s = sector_from_log.strip()

    # exact match
    for short, full in sector_pairs:
        if s == short:
            return full

    # startswith: log string is prefix of full
    for short, full in sector_pairs:
        if full.startswith(s):
            return full

    # containment
    for short, full in sector_pairs:
        if s in full:
            return full

    # fallback: return original
    return s

# ------------------------------------------------------------
# 2) parse master.log → list of runs
#    (dt, region_iso, sector_short_log, region_sector_full)
# ------------------------------------------------------------
def parse_log_to_runs(log_path: Path, sector_pairs):
    runs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(
                r"Starting run \d+ for ([A-Z]{3})_([^']+) at "
                r"(\w+ \w+  ?\d+ \d\d:\d\d:\d\d UTC \d{4})",
                line,
            )
            if not m:
                continue
            region_iso, sector_short, timestr = m.groups()
            dt = datetime.strptime(timestr, "%a %b %d %H:%M:%S UTC %Y")

            full_sector = resolve_full_sector(sector_short, sector_pairs)
            region_sector_full = f"{region_iso}_{full_sector}"

            runs.append((dt, region_iso, sector_short.strip(), region_sector_full))
    return runs

# ------------------------------------------------------------
# 3) list folders under Global4/final → list of (dt, folder_name)
# ------------------------------------------------------------
def list_folders_with_dt(base_dir: Path):
    folders = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name  # e.g. 'THA_Sugar re_20260203_050812'
        parts = name.split("_")
        if len(parts) < 3:
            continue
        datepart, timepart = parts[-2], parts[-1]
        try:
            dt = datetime.strptime(f"{datepart}_{timepart}", "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        folders.append((dt, name))
    return folders

# ------------------------------------------------------------
# 4) fuzzy match runs to folders within ±5 seconds
# ------------------------------------------------------------
def find_best_folder_for_run_dt(run_dt, folder_entries, max_delta=5):
    best_name = None
    best_abs = None
    for folder_dt, folder_name in folder_entries:
        delta = (folder_dt - run_dt).total_seconds()
        if abs(delta) <= max_delta:
            if best_abs is None or abs(delta) < best_abs:
                best_abs = abs(delta)
                best_name = folder_name
    return best_name

# ------------------------------------------------------------
# 5) main: build two CSVs
#    - runs_vs_folders_status.csv
#    - missing_loss_region_sector.csv
# ------------------------------------------------------------
def main():
    sector_pairs = load_sector_pairs(SECTORS_FILE)
    runs = parse_log_to_runs(LOG_FILE, sector_pairs)
    folder_entries = list_folders_with_dt(GLOBAL4_FINAL)

    rows = []
    missing_loss = []

    for dt, region_iso, sector_short, region_sector_full in runs:
        ts_str = dt.strftime("%Y%m%d_%H%M%S")
        folder_name = find_best_folder_for_run_dt(dt, folder_entries, max_delta=5)

        if folder_name is not None:
            folder_path = GLOBAL4_FINAL / folder_name
            exists = True
            has_loss = (folder_path / TARGET_FILE).is_file()
        else:
            folder_name = ""
            exists = False
            has_loss = False

        rows.append(
            {
                "region_sector": region_sector_full,
                "timestamp": ts_str,
                "folder_name": folder_name,
                "exists_in_global4_final": exists,
                "has_loss_per_region_sector_time": has_loss,
            }
        )

        if not has_loss:
            missing_loss.append({"region_sector": region_sector_full})

    status_df = pd.DataFrame(rows)
    status_df.to_csv("runs_vs_folders_status.csv", index=False)

    missing_df = pd.DataFrame(missing_loss)
    missing_df.to_csv("missing_loss_region_sector.csv", index=False)

if __name__ == "__main__":
    main()
