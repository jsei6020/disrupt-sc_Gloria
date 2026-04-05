import re
from datetime import datetime
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# paths (adjust base if needed)
# ------------------------------------------------------------
BASE = Path("/home/user/Documents/University/Master Thesis")
GLOBAL4_FINAL = BASE / "disrupt-sc/output/Global4/final8weeks"
GLORIA = BASE / "Databases/Gloria"
SECTORS_FILE = GLORIA / "IO-Table/sector_table_g.csv"
TARGET_FILE = "loss_per_region_sector_time.csv"
PARAMS_FILE = "parameters.yaml"
REGION_SECTOR_LIST = BASE / "supply_concentrations/region_sector_pairs_1pct4.csv"

# ------------------------------------------------------------
# 1) load expected runs from CSV file
# ------------------------------------------------------------
def load_expected_runs_from_csv(csv_path: Path):
    """
    Load expected runs from region_sector_pairs_1pct.csv
    Format: one column with header 'region_sector', values like "NOR_Fish products"
    """
    expected_runs = []
    
    # Read CSV WITH header (first row is column name, skip it)
    df = pd.read_csv(csv_path, header=0)
    
    # Get the first column name dynamically
    col_name = df.columns[0]
    
    for idx, row in df.iterrows():
        region_sector_val = str(row[col_name]).strip()
        
        # Skip empty or header-like values
        if not region_sector_val or region_sector_val.lower() == 'region_sector':
            continue
        
        if '_' not in region_sector_val:
            print(f"Warning: Skipping row {idx+2} - no underscore found: {region_sector_val}")
            continue
            
        parts = region_sector_val.split('_', 1)
        region_iso = parts[0].strip()
        sector_name = parts[1].strip()
        
        region_sector_full = f"{region_iso}_{sector_name}"
        
        expected_runs.append((region_iso, sector_name, region_sector_full))
    
    print(f"Loaded {len(expected_runs)} expected runs from {csv_path.name}")
    return expected_runs

# ------------------------------------------------------------
# 2) read parameters.yaml and extract region_sector using regex
# ------------------------------------------------------------
def extract_region_sector_from_params(params_path: Path):
    """
    Read parameters.yaml as text and extract the region_sector from 
    disruptions.filter.region_sector using regex.
    
    Avoids yaml.load() entirely since the file contains !!python/object tags
    that would try to reconstruct Python objects.
    """
    try:
        with open(params_path, 'r') as f:
            content = f.read()
        
        # Look for the region_sector list under disruptions.filter
        # Pattern matches:
        #   region_sector:
        #   - ARE_Basic organic chemicals
        # or:
        #   region_sector: [ARE_Basic organic chemicals]
        
        # First, find the disruptions section and extract region_sector values
        # We look for the pattern under filter: section
        pattern = r'filter:\s*\n\s+region_sector:\s*\n\s+-\s+(.+?)(?:\n\s+-|\n\s+\w|\Z)'
        match = re.search(pattern, content, re.MULTILINE)
        
        if match:
            region_sector = match.group(1).strip()
            return region_sector
        
        # Alternative: try simpler pattern for inline list
        pattern2 = r'region_sector:\s*\n?\s*-\s+(.+?)(?:\n|\Z)'
        match2 = re.search(pattern2, content, re.MULTILINE)
        
        if match2:
            region_sector = match2.group(1).strip()
            return region_sector
        
        return None
    
    except Exception as e:
        print(f"Error reading {params_path}: {e}")
        return None

# ------------------------------------------------------------
# 3) scan folders and build mapping from region_sector to folder info
# ------------------------------------------------------------
def scan_folders_for_params(base_dir: Path, params_filename: str = "parameters.yaml"):
    """
    Scan all folders under base_dir, read parameters.yaml from each,
    and build a dict mapping region_sector -> list of folder info dicts
    """
    folder_map = {}  # region_sector -> list of folder info dicts
    
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        
        params_path = p / params_filename
        if not params_path.exists():
            print(f"Warning: No {params_filename} in {p.name}, skipping")
            continue
        
        region_sector = extract_region_sector_from_params(params_path)
        if region_sector is None:
            print(f"Warning: Could not extract region_sector from {p.name}")
            continue
        
        # Get folder timestamp from name (for display purposes)
        folder_name = p.name
        timestamp = ""
        parts = folder_name.rsplit('_', 2)
        if len(parts) >= 3:
            datepart, timepart = parts[-2], parts[-1]
            try:
                dt = datetime.strptime(f"{datepart}_{timepart}", "%Y%m%d_%H%M%S")
                timestamp = dt.strftime("%Y%m%d_%H%M%S")
            except ValueError:
                pass
        
        folder_info = {
            "folder_name": folder_name,
            "folder_path": p,
            "timestamp": timestamp,
            "has_loss_file": (p / TARGET_FILE).exists()
        }
        
        if region_sector not in folder_map:
            folder_map[region_sector] = []
        folder_map[region_sector].append(folder_info)
    
    print(f"Scanned {len(folder_map)} unique region_sectors from folders")
    return folder_map

# ------------------------------------------------------------
# 4) main: build two CSVs
# ------------------------------------------------------------
def main():
    # Load expected runs from CSV
    expected_runs = load_expected_runs_from_csv(REGION_SECTOR_LIST)
    print(f"Loaded {len(expected_runs)} expected runs from CSV")
    
    # Scan folders and build region_sector -> folder mapping
    folder_map = scan_folders_for_params(GLOBAL4_FINAL)
    print(f"Found folders for {len(folder_map)} unique region_sectors")
    
    rows = []
    missing_loss = []
    
    # Process each expected run
    for idx, (region_iso, sector_name, region_sector_full) in enumerate(expected_runs):
        print(f"Checking {idx+1}/{len(expected_runs)}: {region_sector_full}")
        
        if region_sector_full in folder_map:
            # Found matching folder(s) - use the most recent one
            folders = folder_map[region_sector_full]
            folders_sorted = sorted(folders, key=lambda x: x['timestamp'], reverse=True)
            best_folder = folders_sorted[0]
            
            rows.append({
                "region_sector": region_sector_full,
                "timestamp": best_folder['timestamp'],
                "folder_name": best_folder['folder_name'],
                "exists_in_global4_final": True,
                "has_loss_per_region_sector_time": best_folder['has_loss_file'],
            })
            
            if not best_folder['has_loss_file']:
                missing_loss.append({"region_sector": region_sector_full})
        else:
            # No matching folder found
            rows.append({
                "region_sector": region_sector_full,
                "timestamp": "",
                "folder_name": "",
                "exists_in_global4_final": False,
                "has_loss_per_region_sector_time": False,
            })
            missing_loss.append({"region_sector": region_sector_full})
    
    # Check for extra folders (folders with region_sector not in expected runs)
    expected_region_sectors = set(run[2] for run in expected_runs)
    extra_folders = []
    for region_sector, folders in folder_map.items():
        if region_sector not in expected_region_sectors:
            for folder in folders:
                extra_folders.append({
                    "region_sector": "UNKNOWN (extra folder)",
                    "timestamp": folder['timestamp'],
                    "folder_name": folder['folder_name'],
                    "exists_in_global4_final": True,
                    "has_loss_per_region_sector_time": folder['has_loss_file'],
                })
    
    if extra_folders:
        print(f"\nWarning: Found {len(extra_folders)} folders with unexpected region_sectors")
        rows.extend(extra_folders)
    
    # Create DataFrames (preserve order, no sorting)
    status_df = pd.DataFrame(rows)
    status_df.to_csv("runs_vs_folders_status2.csv", index=False)
    
    missing_df = pd.DataFrame(missing_loss)
    missing_df.to_csv("missing_loss_region_sector2.csv", index=False)
    
    # Calculate statistics
    matched = len([r for r in rows if r['exists_in_global4_final'] and r['region_sector'] != 'UNKNOWN (extra folder)'])
    missing = len([r for r in rows if not r['exists_in_global4_final']])
    
    print(f"\nSummary:")
    print(f"Total expected runs: {len(expected_runs)}")
    print(f"Unique region_sectors in folders: {len(folder_map)}")
    print(f"Matched runs: {matched}")
    print(f"Missing runs (no folder): {missing}")
    print(f"Missing loss files: {len(missing_df)}")
    print(f"Extra folders (unexpected): {len(extra_folders)}")

if __name__ == "__main__":
    main()