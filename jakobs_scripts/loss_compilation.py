from pathlib import Path
import pandas as pd



def _finalize_loss_per_region_sector_time(folder):
        """
        Build the final, ordered loss_per_region_sector_time.csv from the
        streamed, unsorted per-step aggregation file.
        """
        path = folder / 'loss_stream_unsorted.csv'
        if not path.exists():
            return

        df = pd.read_csv(path)

        if df.empty:
            # nothing to do; optional: still write an empty file with header
            df.to_csv(folder / 'loss_per_region_sector_time.csv', index=False)
            return

        # optional extra safety: re-aggregate in case some (region, sector, time_step)
        # combos appeared in multiple chunks (shouldn't if you only ever add once per step)
        grouped = (
            df.groupby(['region', 'sector', 'time_step'], as_index=False)['loss']
            .sum()
        )

        # sort as requested: first by region, then sector, then time_step
        grouped = grouped.sort_values(['region', 'sector', 'time_step'])

        # write main output
        grouped.to_csv(folder / 'loss_per_region_sector_time.csv', index=False)

        #delete unsorted file for storage efficiency
        #path.unlink()

def main():
    #folder1 = Path("/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks/AFG_Growing _20260427_234341")
    #_finalize_loss_per_region_sector_time(folder1)
    folder2 = Path("/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/restoration/USA_Growing _20260918_060920/")
    _finalize_loss_per_region_sector_time(folder2)

if __name__ == "__main__":
    main()