"""Queue runner for the rerun (2026-11). Runs on the VM from the repo root:

    python rerun/runner.py rerun/runs.csv [--maxpar 10] [--min-free-gb 20] [--shutdown]

runs.csv columns: tag, region_sector, overrides (JSON; keys: disruption_duration, any top-level parameter,
or a nested one with dots, e.g. "sectors.pichler_grouping": "none" for the upper-bound runs).
Per run: config/parameters/user_defined_Global4_run_r_<tag>.yaml is built from the base config
(user_defined_Global4_rerun.yaml) with the region-sector and the overrides; data/<scope> links to the data.
A run is done when its output folder (found through run_info.json) has loss_summary.csv and the
process exited with 0; then rerun/postprocess.py aggregates it, compresses the loss file and deletes
the transport files. State lives in ~/rerun_state (done/, failed/, logs/, queue.log), so a restart
(after a spot preemption: cron @reboot) resumes: done runs are skipped, interrupted runs start again.
A run that fails twice is not retried. With --shutdown the VM powers off when nothing is left.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
STATE = Path.home() / "rerun_state"
DATA = Path.home() / "disrupt-sc-data" / "Global4"
BASE = REPO / "config" / "parameters" / "user_defined_Global4_rerun.yaml"
PREFIX = "Global4_run_r_"


def log(msg):
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(STATE / "queue.log", "a") as f:
        f.write(line + "\n")


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def running_scopes():
    """Scopes of model processes currently running (also those started by an earlier runner)."""
    out = subprocess.run(["pgrep", "-af", f"src/disruptsc/main.py {PREFIX}"], capture_output=True, text=True).stdout
    scopes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] == PY and parts[2] == "src/disruptsc/main.py":
            scopes[parts[3]] = int(parts[0])
    return scopes


def write_config(tag, region_sector, overrides):
    with open(BASE) as f:
        cfg = yaml.safe_load(f)
    dis = cfg["disruptions"][0]
    dis["filter"]["region_sector"] = [region_sector]
    for key, value in overrides.items():
        if key == "disruption_duration":
            dis["duration"] = int(value)
        elif "." in key:  # nested parameter, e.g. sectors.pichler_grouping
            node = cfg
            *parents, leaf = key.split(".")
            for k in parents:
                node = node[k]
            if leaf not in node:
                raise KeyError(f"{tag}: override {key} does not exist in the base config")
            node[leaf] = value
        elif key not in cfg:
            raise KeyError(f"{tag}: override {key} is not a top-level parameter of the base config")
        else:
            cfg[key] = value
    scope = PREFIX + tag
    with open(REPO / "config" / "parameters" / f"user_defined_{scope}.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    link = REPO / "data" / scope
    if not link.exists():
        link.symlink_to(DATA)
    return scope


def find_output(scope):
    """Newest output folder whose run_info.json names this scope."""
    hits = []
    for info in (REPO / "output" / "Global4").glob("*/run_info.json"):
        try:
            if json.loads(info.read_text()).get("scope") == scope:
                hits.append(info.parent)
        except (OSError, ValueError):
            continue
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--maxpar", type=int, default=10)
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--shutdown", action="store_true")
    a = ap.parse_args()
    for d in ("done", "failed", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(a.runs, newline="") as f:
        runs = [(r["tag"], r["region_sector"], json.loads(r.get("overrides") or "{}")) for r in csv.DictReader(f)]
    log(f"runner start: {len(runs)} runs, commit {commit}, maxpar {a.maxpar}")
    procs = {}  # tag -> Popen (started by this runner)
    adopted = {}  # tag -> pid (still running from an earlier runner)
    for scope, pid in running_scopes().items():
        adopted[scope[len(PREFIX):]] = pid
    pending = [r for r in runs if not (STATE / "done" / r[0]).exists() and r[0] not in adopted]

    def finish(tag, code):
        scope = PREFIX + tag
        out = find_output(scope)
        ok = code == 0 and out is not None and (out / "loss_summary.csv").exists()
        if ok:
            pp = subprocess.run(["nice", "-n", "10", PY, str(REPO / "rerun" / "postprocess.py"), str(out), tag],
                                capture_output=True, text=True)
            ok = pp.returncode == 0
            if not ok:
                log(f"postprocess failed {tag}: {pp.stderr.strip()[-300:]}")
        if ok:
            (STATE / "done" / tag).write_text(str(out) + "\n")
            log(f"done {tag} -> {out.name}")
        else:
            fails = STATE / "failed" / tag
            n = int(fails.read_text()) + 1 if fails.exists() else 1
            fails.write_text(str(n))
            log(f"FAILED {tag} (exit {code}, folder {out.name if out else None}, attempt {n})")
            if n < 2:
                pending.append(next(r for r in runs if r[0] == tag))

    while pending or procs or adopted:
        for tag, p in list(procs.items()):
            if p.poll() is not None:
                del procs[tag]
                finish(tag, p.returncode)
        alive = running_scopes()
        for tag, pid in list(adopted.items()):
            if PREFIX + tag not in alive:
                del adopted[tag]
                out = find_output(PREFIX + tag)
                finish(tag, 0 if out is not None and (out / "loss_summary.csv").exists() else 1)
        if pending and len(alive) < a.maxpar and mem_available_gb() >= a.min_free_gb:
            tag, rs, ov = pending.pop(0)
            if (STATE / "failed" / tag).exists() and int((STATE / "failed" / tag).read_text()) >= 2:
                continue
            scope = write_config(tag, rs, ov)
            logf = open(STATE / "logs" / f"{tag}.log", "w")
            env = dict(os.environ, DSC_COMMIT=commit)
            procs[tag] = subprocess.Popen(["/usr/bin/time", "-v", PY, "src/disruptsc/main.py", scope, "--cache_isolation"],
                                          stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=env)
            log(f"start {tag} ({rs}, {ov or 'baseline'}); running {len(alive) + 1}, free {mem_available_gb():.0f} GB")
            time.sleep(45)  # stagger start-up peaks
            continue
        time.sleep(30)
    left = [r[0] for r in runs if not (STATE / "done" / r[0]).exists()]
    log(f"queue finished; not done: {len(left)} {left[:20]}")
    if a.shutdown:
        log("shutdown in 10 minutes")
        subprocess.run(["sudo", "shutdown", "-h", "+10", "rerun queue finished"])


if __name__ == "__main__":
    main()
