"""Post-process one finished run: python rerun/postprocess.py <output folder> <tag>

Writes small aggregates next to the outputs (agg_ts.csv: loss per time step; agg_rs.csv: loss per
region and supplying region-sector; agg_regts.csv: loss per buying region and time step), compresses
loss_per_region_sector_time.csv to .csv.gz (verified by reading it back), and deletes the transport
outputs (transport_edges_with_flows_*.geojson, transport_network_data.jsonl), which the analysis does
not use. Exits non-zero if anything is missing, so the runner does not mark the run done.
"""
import gzip
import shutil
import sys
from pathlib import Path

import pandas as pd

folder, tag = Path(sys.argv[1]), sys.argv[2]
loss = folder / "loss_per_region_sector_time.csv"
gz = folder / "loss_per_region_sector_time.csv.gz"
if not loss.exists() and not gz.exists():
    sys.exit(f"{tag}: no loss file in {folder}")
src = loss if loss.exists() else gz
df = pd.read_csv(src)
df.groupby("time_step")["loss"].sum().to_csv(folder / "agg_ts.csv")
df.groupby(["region", "sector"])["loss"].sum().to_csv(folder / "agg_rs.csv")
df.groupby(["region", "time_step"])["loss"].sum().to_csv(folder / "agg_regts.csv")
if loss.exists():
    with open(loss, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)
    back = pd.read_csv(gz)
    if len(back) != len(df) or abs(back["loss"].sum() - df["loss"].sum()) > 1e-6 * max(1.0, abs(df["loss"].sum())):
        sys.exit(f"{tag}: compressed loss file does not match")
    loss.unlink()
for p in list(folder.glob("transport_edges_with_flows_*.geojson")) + [folder / "transport_network_data.jsonl"]:
    if p.exists():
        p.unlink()
(folder / "TAG").write_text(tag + "\n")
print(f"{tag}: {len(df)} rows, total loss {df['loss'].sum():.6g}, steps {df['time_step'].max()}")
