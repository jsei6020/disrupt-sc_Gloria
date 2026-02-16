import re
from datetime import datetime
import pandas as pd

filepath = "/home/user/Documents/University/Master Thesis/"
GLORIA = filepath + "Databases/Gloria/"
REGIONS_FILE = GLORIA + "IO-Table/regions_table.csv"
SECTORS_FILE = GLORIA + "IO-Table/sector_table_g.csv"
LOG_FILE = filepath + "disrupt-sc/output/Global4/final/logs_final/master.log"

FOLDERS = [
    "./DEU_Motor ve_20260208_093745",
    "./CHN_Dyes_20260205_043902",
    "./VNM_Growing _20260130_142417",
    "./DEU_Sugar re_20260203_035001",
    "./VNM_Growing _20260130_132535",
    "./USA_Growing _20260130_141440",
    "./IDN_Nickel o_20260129_121442",
    "./COL_Growing _20260130_124756",
    "./THA_Sugar re_20260203_050812",
    "./CAN_Growing _20260130_134152",
    "./ARG_Growing _20260129_132650",
    "./DEU_Dyes_20260205_045434",
    "./NLD_Sugar re_20260203_045144",
    "./USA_Dyes_20260205_053550",
    "./GBR_Sugar re_20260203_042059",
    "./BEL_Sugar re_20260203_031745",
    "./POL_Sugar re_20260203_045437",
    "./FRA_Dyes_20260205_051001",
    "./USA_Motor ve_20260208_102527",
    "./IND_Growing _20260130_135817",
    "./BRA_Growing _20260130_123225",
    "./BRA_Growing _20260129_164522",
    "./ESP_Motor ve_20260208_095324",
    "./CAN_Growing _20260129_132650",
    "./IDN_Nickel o_20260129_121010",
    "./CHN_Motor ve_20260208_090641",
    "./IDN_Growing _20260130_131852",
    "./FRA_Sugar re_20260203_040528",
    "./BRA_Sugar re_20260203_033430",
    "./UKR_Growing _20260128_140042",
    "./CHN_Growing _20260130_135137",
    "./KOR_Motor ve_20260208_100906",
    "./NLD_Dyes_20260205_052526",
    "./MEX_Motor ve_20260208_102109",
    "./CZE_Motor ve_20260208_092214",
    "./RUS_Growing _20260130_140753",
    "./ARE_Growing _20260130_133517",
    "./JPN_Motor ve_20260208_100445",
    "./IND_Sugar re_20260203_043626",
    "./CAN_Motor ve_20260208_085111",
    "./HND_Growing _20260130_130325",
    "./USA_Growing _20260129_164802",
]

def parse_log_to_dt_list(log_path: str):
    """Return list of (datetime, iso_sector) from 'Starting run' lines only."""
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(
                r"Starting run \d+ for ([A-Z]{3}_[^']+) at "
                r"(\w+ \w+  ?\d+ \d\d:\d\d:\d\d UTC \d{4})",
                line,
            )
            if not m:
                continue
            iso_sector, timestr = m.groups()
            dt = datetime.strptime(timestr, "%a %b %d %H:%M:%S UTC %Y")
            entries.append((dt, iso_sector))
    return entries

def find_best_match(ts_dt, entries, max_delta=5):
    """Find iso_sector whose start time is within ±max_delta seconds of ts_dt."""
    best_iso = None
    best_abs = None
    for dt, iso_sector in entries:
        delta = (ts_dt - dt).total_seconds()
        if abs(delta) <= max_delta:
            if best_abs is None or abs(delta) < best_abs:
                best_abs = abs(delta)
                best_iso = iso_sector
    return best_iso

def main():
    entries = parse_log_to_dt_list(LOG_FILE)

    rows = []

    # 1) Match each folder timestamp to closest 'Starting run' within ±5s
    matched_dts = set()
    for folder in FOLDERS:
        parts = folder.split("_")
        datepart, timepart = parts[-2], parts[-1]
        ts_str = f"{datepart}_{timepart}"
        ts_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

        iso_sector = find_best_match(ts_dt, entries, max_delta=5)
        if iso_sector:
            matched_dts.add(ts_dt)
        rows.append(
            {
                "ISO_sector": iso_sector or "",
                "Timestamp": ts_str,
                "Folder": folder,
            }
        )

    # 2) Add all started runs that are not within ±5s of any matched folder
    for dt, iso_sector in entries:
        close_to_any = any(
            abs((dt - mdt).total_seconds()) <= 5 for mdt in matched_dts
        )
        if not close_to_any:
            ts_str = dt.strftime("%Y%m%d_%H%M%S")
            rows.append(
                {
                    "ISO_sector": iso_sector,
                    "Timestamp": ts_str,
                    "Folder": "",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv("iso_sector_mapping_all_fuzzy.csv", index=False)

if __name__ == "__main__":
    main()
