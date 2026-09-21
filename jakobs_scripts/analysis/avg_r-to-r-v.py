import csv
from collections import defaultdict


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
VULN_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/region_vulnerability_matrix_top3_original_ban.csv"
CONTINENT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/continent_groups.csv"
OUT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/top10_banning_countries_by_affected_continent_transposed.csv"


# -------------------------------------------------------------------
# 1. Read continent groups and build region -> continent mapping
# -------------------------------------------------------------------
continent_by_region = {}

with open(CONTINENT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("North America Acronym"):
            continent_by_region[row["North America Acronym"].strip()] = "North America"
        if row.get("Europe Acronym"):
            continent_by_region[row["Europe Acronym"].strip()] = "Europe"
        if row.get("Asia Acronym"):
            continent_by_region[row["Asia Acronym"].strip()] = "Asia"
        if row.get("Oceania Acronym"):
            continent_by_region[row["Oceania Acronym"].strip()] = "Oceania"
        if row.get("Latin America Acronym"):
            continent_by_region[row["Latin America Acronym"].strip()] = "Latin America"
        if row.get("Africa Acronym"):
            continent_by_region[row["Africa Acronym"].strip()] = "Africa"


# -------------------------------------------------------------------
# 2. Parse header to recover affected-region column blocks
#    Format assumed:
#    header row = AFG,AFG,|,AGO,AGO,|,...
#    each affected region occupies 3 columns:
#      col j   = banning region code
#      col j+1 = vulnerability value
#      col j+2 = "|"
# -------------------------------------------------------------------
with open(VULN_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)

    try:
        header_row = next(reader)
    except StopIteration:
        raise RuntimeError("Vulnerability matrix file appears to be empty.")

    second_row = next(reader, None)
    if second_row is not None and any(cell.strip() for cell in second_row):
        data_rows = [second_row]
    else:
        data_rows = []

    data_rows.extend(list(reader))


# Build list of affected-region blocks from header
affected_blocks = []
j = 0
while j + 1 < len(header_row):
    affected_region = header_row[j].strip()
    if affected_region and affected_region != "|":
        affected_blocks.append((affected_region, j, j + 1))
    j += 3


# -------------------------------------------------------------------
# 3. Aggregate vulnerability by affected continent and banning country
# -------------------------------------------------------------------
# continent_country_sums[affected_continent][banning_country] = total vulnerability
continent_country_sums = defaultdict(lambda: defaultdict(float))

for row in data_rows:
    if not any(cell.strip() for cell in row):
        continue

    for affected_region, code_col, value_col in affected_blocks:
        affected_continent = continent_by_region.get(affected_region)
        if affected_continent is None:
            continue

        banning_country = row[code_col].strip() if code_col < len(row) else ""
        value_str = row[value_col].strip() if value_col < len(row) else ""

        if not banning_country or banning_country == "|":
            continue

        try:
            value = float(value_str)
        except ValueError:
            value = 0.0

        continent_country_sums[affected_continent][banning_country] += value


# -------------------------------------------------------------------
# 4. Rank top 10 banning countries for each affected continent
# -------------------------------------------------------------------
continents_order = [
    "North America",
    "Europe",
    "Asia",
    "Oceania",
    "Latin America",
    "Africa"
]

ranked_by_continent = {}
for continent in continents_order:
    country_totals = continent_country_sums.get(continent, {})
    ranked = sorted(country_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    ranked_by_continent[continent] = ranked


# -------------------------------------------------------------------
# 5. Export TRANSPOSED output:
#    columns = continent blocks
#    rows = ranks 1..10
# -------------------------------------------------------------------
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Header row
    header = []
    for continent in continents_order:
        header.extend([continent, continent, "|"])
    writer.writerow(header)

    # Sub-header row
    subheader = []
    for _ in continents_order:
        subheader.extend(["Country", "TotalVulnerability", "|"])
    writer.writerow(subheader)

    # Rank rows
    for rank_idx in range(20):
        row = []
        for continent in continents_order:
            ranked = ranked_by_continent.get(continent, [])
            if rank_idx < len(ranked):
                country, total_val = ranked[rank_idx]
                row.extend([country, total_val, "|"])
            else:
                row.extend(["", "", "|"])
        writer.writerow(row)

print(f"Wrote transposed top-10 banning countries by affected continent to {OUT_FILE}")