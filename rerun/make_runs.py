"""Build rerun/runs.csv (tag, region_sector, overrides) from the main run list and, optionally, the sensitivity list.
Main runs: tag = ISO3_NNN_Slug (config/sector_slugs.csv), no overrides.
Sensitivity runs are not part of the main launch: their bans are chosen from the top 30 runs by loss once the
full results exist, so runlist_sensitivity.csv must be rebuilt before --with-sensitivity or --only-sensitivity is used.
Usage: python rerun/make_runs.py [--with-sensitivity | --only-sensitivity] [--limit N]"""
import csv, sys
from pathlib import Path
here = Path(__file__).resolve().parent
slugs = {r["sector"]: (int(r["number"]), r["slug"]) for r in csv.DictReader(open(here.parent / "config" / "sector_slugs.csv"))}
rows = []
if "--only-sensitivity" not in sys.argv:
    for r in csv.DictReader(open(here / "runlist_rerun.csv")):
        reg, sec = r["region_sector"].split("_", 1)
        n, s = slugs[sec]
        rows.append({"tag": f"{reg}_{n:03d}_{s}", "region_sector": r["region_sector"], "overrides": "{}"})
for r in (csv.DictReader(open(here / "runlist_sensitivity.csv"))
          if {"--with-sensitivity", "--only-sensitivity"} & set(sys.argv) else []):
    rows.append({"tag": "sens_" + r["tag"], "region_sector": r["region_sector"], "overrides": r["overrides"]})
if "--limit" in sys.argv:
    rows = rows[:int(sys.argv[sys.argv.index("--limit") + 1])]
assert len({r["tag"] for r in rows}) == len(rows), "duplicate tags"
with open(here / "runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "region_sector", "overrides"]); w.writeheader(); w.writerows(rows)
print(len(rows), "runs ->", here / "runs.csv")
