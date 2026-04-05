"""
Extract a subset MRIO table from GLORIA, with region/sector aggregation.
Output: double-indexed CSV (region × sector).

Values are in millions USD. Rounded to 2 decimal places (~10k USD precision).
"""

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE = "../GLORIA/mrio_va_fd.pkl"
OUTPUT = "output/mrio_subset.csv"

# Regions kept at full detail in the IO core
INTERNAL_REGIONS = ["ARE", "QAT", "KWT", "BHR", "OMN", "SAU", "IRQ", "IRN"]

# External region groups: label -> list of ISO3 codes
# Single-country groups are fine (e.g. "CHN": ["CHN"])
# All regions not in INTERNAL_REGIONS or here go into "RoW" catch-all
EXTERNAL_REGIONS = {
    "CHN": ["CHN"],
    "IND": ["IND"],
    "PAK": ["PAK"],
    "TUR": ["TUR"],
    "AFR": [
        "XAF",  # Rest of Africa (GLORIA aggregate)
        "AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR",
        "COD", "COG", "DJI", "DZA", "EGY", "ERI", "ETH", "GAB",
        "GHA", "GIN", "GMB", "GNQ", "HTI", "KEN", "LBR", "LBY",
        "MAR", "MDG", "MLI", "MOZ", "MRT", "MWI", "NAM", "NER",
        "NGA", "RWA", "SDS", "SEN", "SLE", "SOM", "SDN", "TCD",
        "TGO", "TUN", "TZA", "UGA", "ZAF", "ZMB", "ZWE",
    ],
    "EAS": [
        "XAS",  # Rest of Asia-Pacific (GLORIA aggregate)
        "JPN", "KOR", "PRK", "MNG",  # East Asia
        "HKG", "SGP", "MYS", "IDN", "THA",  # SE Asia
        "PHL", "VNM", "KHM", "LAO", "MMR",  # SE Asia cont.
        "BRN",  # Brunei
    ],
}

# Sector aggregation: new_name -> list of original sector names to merge
# Sectors not listed here stay at original granularity
# Set to None or {} for no aggregation
SECTOR_AGGREGATION = {
    "Agriculture": [
        "Growing wheat", "Growing maize", "Growing cereals n.e.c.",
        "Growing leguminous crops and oil seeds", "Growing rice",
        "Growing vegetables, roots, tubers", "Growing sugar beet and cane",
        "Growing tobacco", "Growing fibre crops", "Growing crops n.e.c.",
        "Growing grapes", "Growing fruits and nuts",
        "Growing beverage crops (coffee, tea etc)",
        "Growing spices, aromatic, drug and pharmaceutical crops",
        "Seeds and plant propagation",
        "Raising of cattle", "Raising of sheep and goats",
        "Raising of swine/pigs", "Raising of poultry",
        "Raising of animals n.e.c.; services to agriculture",
    ],
    "Forestry": ["Forestry and logging"],
    "Fishing": ["Fishing", "Crustaceans and molluscs"],
    "Coal": ["Hard coal", "Lignite and peat"],
    "Oil": ["Petroleum extraction"],
    "Gas": ["Gas extraction"],
    "Mining": [
        "Iron ores", "Uranium ores", "Aluminium ore", "Copper ores",
        "Gold ores", "Lead/zinc/silver ores", "Nickel ores", "Tin ores",
        "Other non-ferrous ores", "Quarrying of stone, sand and clay",
        "Chemical and fertilizer minerals", "Extraction of salt",
        "Mining and quarrying n.e.c.; services to mining",
    ],
    "Food manufacturing": [
        "Beef meat", "Sheep meat", "Pork", "Poultry meat",
        "Other meat products", "Fish products", "Cereal products",
        "Vegetable products", "Fruit products",
        "Food products and feeds n.e.c.",
        "Sugar refining; cocoa, chocolate and confectionery",
        "Animal oils and fats", "Vegetable oils and fats", "Dairy products",
        "Alcoholic and other beverages", "Tobacco products",
    ],
    "Fertilizer": [
        "Nitrogenous fertilizers",
        "Non-nitrogenous and mixed fertilizers",
    ],
    "Refining": [
        "Coke oven products", "Refined petroleum products",
    ],
    "Other manufacturing": [
        "Textiles and clothing", "Leather and footwear",
        "Sawmill products", "Pulp and paper", "Printing",
        "Basic petrochemical products", "Basic inorganic chemicals",
        "Basic organic chemicals", "Pharmaceuticals and medicinal products",
        "Dyes, paints, glues, detergents and other chemical products",
        "Rubber products", "Plastic products",
        "Clay building materials", "Glass and other ceramics n.e.c.",
        "Cement, lime and plaster products",
        "Other non-metallic mineral products n.e.c.",
        "Basic iron and steel", "Basic aluminium", "Basic copper",
        "Basic gold", "Basic lead/zinc/silver", "Basic nickel",
        "Basic tin", "Basic non-ferrous metals n.e.c.",
        "Fabricated metal products", "Machinery and equipment",
        "Motor vehicles, trailers and semi-trailers",
        "Other transport equipment",
        "Repair and installation of machinery and equipment (service)",
        "Computers; electronic products; optical and precision instruments",
        "Electrical equipment", "Furniture and other manufacturing n.e.c.",
    ],
    "Electricity": [
        "Electric power generation, transmission and distribution",
    ],
    "Water": [
        "Water collection, treatment and supply; sewerage",
    ],
    "Other utility": [
        "Distribution of gaseous fuels through mains",
        "Waste collection, treatment, and disposal",
        "Materials recovery",
    ],
    "Construction": [
        "Building construction", "Civil engineering construction",
    ],
    "Trade": [
        "Wholesale and retail trade; repair of motor vehicles and motorcycles",
    ],
    "Transport": [
        "Road transport", "Rail transport", "Transport via pipeline",
        "Water transport", "Air transport", "Services to transport",
    ],
    "Services": [
        "Postal and courier services", "Hospitality", "Publishing",
        "Telecommunications", "Information services",
        "Finance and insurance", "Property and real estate",
        "Professional, scientific and technical services",
        "Administrative services",
        "Arts, entertainment and recreation", "Other services",
    ],
    "Public administration": [
        "Government; social security; defence; public order",
        "Education", "Human health and social work activities",
    ],
}

# Value-added: True = aggregate into single "Value added" row per region
#              False = keep 6 separate components
AGGREGATE_VALUE_ADDED = True

# Final demand: True = aggregate into single "Final demand" col per region
#               False = keep 6 separate components
AGGREGATE_FINAL_DEMAND = True

# Decimal places in output (2 ≈ 10k USD precision for M USD values)
DECIMALS = 2

# ============================================================================
# KNOWN CATEGORIES
# ============================================================================

VALUE_ADDED_SECTORS = [
    "Compensation of employees D.1",
    "Taxes on production D.29",
    "Subsidies on production D.39",
    "Net operating surplus B.2n",
    "Net mixed income B.3n",
    "Consumption of fixed capital K.1",
]

FINAL_DEMAND_SECTORS = [
    "Household final consumption P.3h",
    "Non-profit institutions serving households P.3n",
    "Government final consumption P.3g",
    "Gross fixed capital formation P.51",
    "Changes in inventories P.52",
    "Acquisitions less disposals of valuables P.53",
]


# ============================================================================
# PROCESSING
# ============================================================================

def load_and_extract():
    print("Loading GLORIA pickle...")
    df = pd.read_pickle(SOURCE)
    print(f"  Loaded: {df.shape[0]} x {df.shape[1]}")

    all_regions = df.index.get_level_values(0).unique().tolist()

    # --- Build region mapping: original_code -> group_label ---
    region_map = {}

    for r in INTERNAL_REGIONS:
        if r not in all_regions:
            print(f"  WARNING: internal region '{r}' not found in data, skipping")
            continue
        region_map[r] = r

    claimed = set(INTERNAL_REGIONS)
    for label, codes in EXTERNAL_REGIONS.items():
        for c in codes:
            if c not in all_regions:
                print(f"  WARNING: external region code '{c}' (group '{label}') not found, skipping")
                continue
            if c in claimed:
                print(f"  WARNING: '{c}' already assigned, skipping duplicate in '{label}'")
                continue
            region_map[c] = label
            claimed.add(c)

    # Everything else -> RoW
    for r in all_regions:
        if r not in claimed:
            region_map[r] = "RoW"

    group_labels = list(dict.fromkeys(region_map.values()))  # preserve order
    print(f"  Region groups: {group_labels}")
    print(f"  Regions mapped: {len(region_map)} / {len(all_regions)}")

    # --- Build sector mapping ---
    all_row_sectors = [s for s in df.index.get_level_values(1).unique()
                       if s not in VALUE_ADDED_SECTORS]
    all_col_sectors = [s for s in df.columns.get_level_values(1).unique()
                       if s not in FINAL_DEMAND_SECTORS]

    sector_map = {}
    claimed_sectors = set()
    if SECTOR_AGGREGATION:
        for label, members in SECTOR_AGGREGATION.items():
            for s in members:
                if s in claimed_sectors:
                    print(f"  WARNING: sector '{s}' already assigned, skipping in '{label}'")
                    continue
                sector_map[s] = label
                claimed_sectors.add(s)

    # Unmapped industry sectors keep their name
    for s in all_row_sectors:
        if s not in sector_map:
            sector_map[s] = s
    for s in all_col_sectors:
        if s not in sector_map:
            sector_map[s] = s

    industry_labels = list(dict.fromkeys(
        sector_map[s] for s in all_row_sectors if s in sector_map
    ))
    print(f"  Industry sectors in output: {len(industry_labels)}")

    # --- VA sector mapping ---
    if AGGREGATE_VALUE_ADDED:
        va_map = {s: "Value added" for s in VALUE_ADDED_SECTORS}
        va_labels = ["Value added"]
    else:
        va_map = {s: s for s in VALUE_ADDED_SECTORS}
        va_labels = VALUE_ADDED_SECTORS

    # --- FD sector mapping ---
    if AGGREGATE_FINAL_DEMAND:
        fd_map = {s: "Final demand" for s in FINAL_DEMAND_SECTORS}
        fd_labels = ["Final demand"]
    else:
        fd_map = {s: s for s in FINAL_DEMAND_SECTORS}
        fd_labels = FINAL_DEMAND_SECTORS

    # --- Build output index/columns ---
    row_idx = (
            [(r, s) for r in group_labels for s in industry_labels] +
            [(r, v) for r in group_labels for v in va_labels]
    )
    col_idx = (
            [(r, s) for r in group_labels for s in industry_labels] +
            [(r, f) for r in group_labels for f in fd_labels]
    )

    # --- Aggregate using groupby ---
    print("Aggregating...")

    # Remap the index and columns of the full dataframe
    row_region_map = df.index.get_level_values(0).map(region_map)
    row_sector_map = df.index.get_level_values(1).map(
        lambda s: va_map.get(s, sector_map.get(s, s))
    )
    col_region_map = df.columns.get_level_values(0).map(region_map)
    col_sector_map = df.columns.get_level_values(1).map(
        lambda s: fd_map.get(s, sector_map.get(s, s))
    )

    # Set new index/columns for groupby
    df.index = pd.MultiIndex.from_arrays([row_region_map, row_sector_map])
    df.columns = pd.MultiIndex.from_arrays([col_region_map, col_sector_map])

    # Group rows
    print("  Grouping rows...")
    df = df.groupby(level=[0, 1], sort=False).sum()

    # Group columns
    print("  Grouping columns...")
    df = df.T.groupby(level=[0, 1], sort=False).sum().T

    # Reindex to desired order
    print("  Reindexing...")
    row_mi = pd.MultiIndex.from_tuples(row_idx)
    col_mi = pd.MultiIndex.from_tuples(col_idx)
    df = df.reindex(index=row_mi, columns=col_mi, fill_value=0.0)

    # Round
    df = df.round(DECIMALS)

    return df


def save_csv(df):
    import os
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    df.to_csv(OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"Saved to {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    mrio = load_and_extract()
    print(f"Output shape: {mrio.shape}")
    save_csv(mrio)
    print("Done.")
