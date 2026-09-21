from collections import defaultdict
from pathlib import Path
import csv
import math


def _recover_loss_per_time_step(folder: Path):
    """
    Recover aggregate losses by time step from a mostly-valid but potentially
    malformed streamed CSV.

    Expected valid format:
        region,sector,time_step,loss

    Normal 4-field rows are aggregated. Malformed rows are written to a
    separate audit file and are not silently counted.
    """
    path = folder / "loss_stream_unsorted.csv"

    if not path.exists():
        print(f"Input file not found: {path}")
        return

    output_path = folder / "loss_per_time_step_recovered.csv"
    malformed_path = folder / "loss_stream_malformed_lines.csv"

    losses_by_time = defaultdict(float)

    valid_rows = 0
    malformed_rows = 0
    skipped_header_rows = 0
    malformed_records = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue

            # Skip a header, if the file contains one.
            if line_number == 1 and [value.strip() for value in row] == [
                "region", "sector", "time_step", "loss"
            ]:
                skipped_header_rows += 1
                continue

            if len(row) != 4:
                malformed_rows += 1
                malformed_records.append(
                    {
                        "line_number": line_number,
                        "field_count": len(row),
                        "raw_line": ",".join(row),
                    }
                )
                continue

            region, sector, time_step_raw, loss_raw = (
                value.strip() for value in row
            )

            try:
                time_step = int(time_step_raw)
                loss = float(loss_raw)
            except ValueError:
                malformed_rows += 1
                malformed_records.append(
                    {
                        "line_number": line_number,
                        "field_count": len(row),
                        "raw_line": ",".join(row),
                    }
                )
                continue

            if not math.isfinite(loss):
                malformed_rows += 1
                malformed_records.append(
                    {
                        "line_number": line_number,
                        "field_count": len(row),
                        "raw_line": ",".join(row),
                    }
                )
                continue

            losses_by_time[time_step] += loss
            valid_rows += 1

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_step", "loss"])

        for time_step in sorted(losses_by_time):
            writer.writerow([time_step, losses_by_time[time_step]])

    with malformed_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["line_number", "field_count", "raw_line"],
        )
        writer.writeheader()
        writer.writerows(malformed_records)

    print(f"Read: {path}")
    print(f"Valid records aggregated: {valid_rows}")
    print(f"Skipped header rows: {skipped_header_rows}")
    print(f"Malformed physical lines excluded: {malformed_rows}")
    print(f"Recovered time-step totals: {output_path}")
    print(f"Malformed-line audit file: {malformed_path}")

    if malformed_rows:
        print(
            "\nImportant: malformed rows were excluded rather than guessed. "
            "Inspect the audit file before treating totals as final."
        )


def main():
    folder = Path(
        "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/restoration/USA_Growing _20260918_070611/"
    )
    _recover_loss_per_time_step(folder)


if __name__ == "__main__":
    main()