import math
from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

BASE_PATH = Path("/home/user/Documents/University/Master Thesis")

VULNERABILITY_PATH = BASE_PATH / "vulnerability_analysis"
THREAT_PATH = BASE_PATH / "threat_analysis"
TABLES_PATH = BASE_PATH / "tables"

REGIONS_TABLE_FILE = (
    BASE_PATH
    / "Databases"
    / "Gloria"
    / "IO-Table"
    / "regions_table.csv"
)


# ============================================================================
# VARIABLES — metric files, ISO columns, and metric columns
# ============================================================================

VARIABLES = [
    # ========================================================================
    # DEFENSIVE VULNERABILITY METRICS
    # ========================================================================

    {
        "label": "FDID-V",
        "file": VULNERABILITY_PATH / "v-fdid.csv",
        "id_col": "iso",
        "metric_col": "final_demand_import_dependency",
    },

    {
        "label": "IS-V",
        "file": VULNERABILITY_PATH / "v-is_fd.csv",
        "id_col": "iso",
        "metric_col": "intensity_score",
    },

    {
        "label": "IDLR-V",
        "file": VULNERABILITY_PATH / "v-idlr.csv",
        "id_col": "iso",
        "metric_col": "indirect_direct_ratio",
    },

    {
        "label": "RT-V",
        "file": VULNERABILITY_PATH / "v-vs_rt.csv",
        "id_col": "iso_3",
        "metric_col": "metric_time_90",
    },

    # ========================================================================
    # OFFENSIVE THREAT METRICS
    # ========================================================================

    {
        "label": "ATL-T",
        "file": THREAT_PATH / "t-atl3.csv",
        "id_col": "region_iso",
        "metric_col": "region_avg_top3_loss_trillions",
    },

    {
        "label": "RT-T",
        "file": THREAT_PATH / "t-rt3.csv",
        "id_col": "region_iso",
        "metric_col": "region_avg_top3_time_to_90pct",
    },

    {
        "label": "IDLR-T",
        "file": THREAT_PATH / "t-idlr3.csv",
        "id_col": "region_iso",
        "metric_col": "region_avg_top3_ratio",
    },
]


# ============================================================================
# TABLE LAYOUT SETTINGS
# ============================================================================

TABLES_PATH.mkdir(parents=True, exist_ok=True)

# Each LaTeX appendix table contains three repeated blocks:
# Rank | Country | Metric
ROWS_PER_BLOCK = 55
NUMBER_OF_BLOCKS = 3
MAX_ROWS_ON_PAGE = ROWS_PER_BLOCK * NUMBER_OF_BLOCKS

# Extend the table equally into both page margins.
# Requires \usepackage{changepage} in the LaTeX preamble.
TABLE_MARGIN_EXTENSION = "2.5cm"

# Width of the country/region-name column within each ranking block.
# Must use \linewidth, because it adapts to the width inside adjustwidth.
COUNTRY_COLUMN_WIDTH = "0.17\\linewidth"

# Visible blank space between the three ranking blocks.
BLOCK_GAP = r"\hspace{1.0cm}"


# ============================================================================
# NUMBER FORMATTING SETTINGS
# ============================================================================

# For values with an absolute magnitude below 1, display this many
# significant figures. For example:
# 0.014913824 -> 0.01491
# 0.000123456 -> 0.0001235
SMALL_VALUE_SIGNIFICANT_FIGURES = 4


# Values >= 100 retain all digits before the decimal separator and are rounded
# to this many decimal places.
#
# Examples with 4:
# 4820477.712345 -> 4820477.7123
# 6.820846       -> 6.8208
LARGE_VALUE_DECIMAL_PLACES = 0


# Always display a decimal place, even for an integer-valued metric.
MINIMUM_DECIMAL_PLACES = 0


# ============================================================================
# LOAD ISO / REGION NAME MAPPING
# ============================================================================

def load_region_name_map():
    """
    Load ISO/region-code to country/region-name mappings.

    regions_table.csv must include:
    - Region_acronyms
    - Region_names
    """
    print("\n[1/2] Loading region-name mapping...")

    if not REGIONS_TABLE_FILE.is_file():
        raise FileNotFoundError(
            f"Region table not found:\n{REGIONS_TABLE_FILE}"
        )

    regions_df = pd.read_csv(REGIONS_TABLE_FILE)

    required_columns = ["Region_acronyms", "Region_names"]
    missing_columns = [
        column
        for column in required_columns
        if column not in regions_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns {missing_columns} in "
            f"{REGIONS_TABLE_FILE.name}.\n"
            f"Available columns: {list(regions_df.columns)}"
        )

    regions_df = regions_df[required_columns].copy()

    regions_df["Region_acronyms"] = (
        regions_df["Region_acronyms"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    regions_df["Region_names"] = (
        regions_df["Region_names"]
        .astype("string")
        .str.strip()
    )

    regions_df = regions_df.dropna(
        subset=["Region_acronyms", "Region_names"]
    )

    duplicate_codes = regions_df["Region_acronyms"].duplicated(keep=False)

    if duplicate_codes.any():
        duplicate_examples = (
            regions_df.loc[duplicate_codes, "Region_acronyms"]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Duplicate Region_acronyms found in regions_table.csv.\n"
            "Each ISO/region code must map to exactly one country/region "
            "name.\n"
            f"Examples: {duplicate_examples}"
        )

    region_name_map = dict(
        zip(
            regions_df["Region_acronyms"],
            regions_df["Region_names"],
        )
    )

    print(
        f"  Loaded names for {len(region_name_map)} ISO/region codes."
    )

    return region_name_map


# ============================================================================
# LOAD AND PREPARE ONE RANKING METRIC
# ============================================================================

def load_metric_for_table(var, region_name_map):
    """
    Load one metric file and produce:

        Rank | Country | Value

    The existing row order in the source CSV is preserved deliberately.
    No ISO, country-name, or metric sorting happens in this function.

    Thus, after invalid observations are removed:
    - first valid source row = rank 1,
    - second valid source row = rank 2,
    - etc.
    """
    file_path = Path(var["file"]).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Metric file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [var["id_col"], var["metric_col"]]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns {missing_columns} in {file_path.name}.\n"
            f"Available columns: {list(df.columns)}"
        )

    table = df[[var["id_col"], var["metric_col"]]].copy()
    table.columns = ["ISO", "Value"]

    table["ISO"] = (
        table["ISO"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    table["Value"] = pd.to_numeric(
        table["Value"],
        errors="coerce",
    )

    table = table.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    table = table.dropna(subset=["ISO", "Value"]).copy()

    duplicate_codes = table["ISO"].duplicated(keep=False)

    if duplicate_codes.any():
        duplicate_examples = (
            table.loc[duplicate_codes, "ISO"]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"{file_path.name} contains duplicate ISO/region identifiers.\n"
            f"Examples: {duplicate_examples}\n"
            "Each appendix ranking table requires exactly one metric value "
            "for each ISO/region code."
        )

    table["Country"] = table["ISO"].map(region_name_map)

    unmapped_codes = (
        table.loc[table["Country"].isna(), "ISO"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if unmapped_codes:
        raise ValueError(
            f"Could not find country/region names for "
            f"{len(unmapped_codes)} code(s) in {file_path.name}.\n"
            f"Unmapped codes: {unmapped_codes[:30]}\n"
            "Ensure these identifiers match the Region_acronyms column "
            "in regions_table.csv."
        )

    # Deliberately retain existing source CSV order.
    table = table[["Country", "Value"]].copy().reset_index(drop=True)

    # Add a ranking index before the country column.
    table.insert(0, "Rank", table.index + 1)

    return table


# ============================================================================
# LATEX AND NUMBER FORMATTING
# ============================================================================

def latex_escape(text):
    """Escape text that has special meanings in LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    return "".join(
        replacements.get(char, char)
        for char in str(text)
    )


def format_value(value):
    """
    Format values for LaTeX ranking tables.

    Rules:
    - abs(value) < 1: round to four significant figures.
    - abs(value) >= 1: preserve all digits before the decimal separator,
      then round the fractional portion to LARGE_VALUE_DECIMAL_PLACES.
    - zero is shown as 0.0.

    Examples:
    - 4820477.712345 -> 4820477.7123
    - 6.820846       -> 6.8208
    - 0.014913824    -> 0.01491
    - 0.000123456    -> 0.0001235
    """
    if pd.isna(value):
        return r"\textemdash"

    value = float(value)

    if value == 0:
        return f"{0:.{MINIMUM_DECIMAL_PLACES}f}"

    absolute_value = abs(value)

    if absolute_value >= 100:
        # Decimal-place rounding cannot change the integer part.
        rounded_value = round(value, LARGE_VALUE_DECIMAL_PLACES)

        return f"{rounded_value:.{LARGE_VALUE_DECIMAL_PLACES}f}"

    # For values between -1 and 1, calculate the decimal places needed
    # to retain exactly SMALL_VALUE_SIGNIFICANT_FIGURES significant figures.
    exponent = math.floor(math.log10(absolute_value))
    decimal_places = (
        SMALL_VALUE_SIGNIFICANT_FIGURES
        - 1
        - exponent
    )

    rounded_value = round(value, decimal_places)

    return f"{rounded_value:.{decimal_places}f}"

def make_latex_label(variable_label):
    """Create a safe LaTeX label from a metric label."""
    safe = (
        variable_label.lower()
        .replace(" ", "-")
        .replace("_", "-")
        .replace("/", "-")
    )

    safe = "".join(
        char
        for char in safe
        if char.isalnum() or char == "-"
    )

    return f"tab:appendix:{safe}"


# ============================================================================
# BUILD THREE-BLOCK RANKING TABLE
# ============================================================================

def split_into_blocks(table):
    """
    Split the rankings into vertical blocks.

    With 60 rows per block:
    - ranks 1–60 occupy the first block,
    - ranks 61–120 occupy the second block,
    - ranks 121–180 occupy the third block.
    """
    blocks = []

    for start in range(0, len(table), ROWS_PER_BLOCK):
        block = table.iloc[start:start + ROWS_PER_BLOCK].copy()
        blocks.append(block.reset_index(drop=True))

    return blocks


def make_table_body(blocks):
    """
    Create LaTeX table rows with three repeated blocks:

    Rank | Country | Value    Rank | Country | Value    Rank | Country | Value
    """
    max_rows = max(len(block) for block in blocks)
    latex_rows = []

    for row_number in range(max_rows):
        cells = []

        for block in blocks:
            if row_number < len(block):
                rank = str(block.loc[row_number, "Rank"])
                country = latex_escape(
                    block.loc[row_number, "Country"]
                )
                value = format_value(
                    block.loc[row_number, "Value"]
                )

                cells.extend([rank, country, value])
            else:
                cells.extend(["", "", ""])

        latex_rows.append(" & ".join(cells) + r" \\")

    return "\n".join(latex_rows)


# ============================================================================
# GENERATE ONE LATEX TABLE FILE
# ============================================================================

def write_table(var, region_name_map):
    """Generate one widened and centred LaTeX appendix ranking table."""
    table = load_metric_for_table(var, region_name_map)

    if len(table) > MAX_ROWS_ON_PAGE:
        raise ValueError(
            f"{var['label']} has {len(table)} observations, but the current "
            f"layout supports at most {MAX_ROWS_ON_PAGE}.\n"
            "Increase ROWS_PER_BLOCK or NUMBER_OF_BLOCKS, or split the "
            "table over multiple pages."
        )

    blocks = split_into_blocks(table)

    # Always output exactly three display blocks.
    while len(blocks) < NUMBER_OF_BLOCKS:
        blocks.append(
            pd.DataFrame(columns=["Rank", "Country", "Value"])
        )

    if len(blocks) > NUMBER_OF_BLOCKS:
        raise ValueError(
            f"{var['label']} requires {len(blocks)} blocks, but "
            f"NUMBER_OF_BLOCKS is set to {NUMBER_OF_BLOCKS}."
        )

    header = " & ".join(
        ["Rank", "Country", latex_escape(var["label"])]
        * NUMBER_OF_BLOCKS
    ) + r" \\"

    body = make_table_body(blocks)

    # Structure of each repeated block:
    # r = rank
    # p{...} = fixed-width/wrapping country name
    # r = numerical metric value
    #
    # @{\hspace{...}} creates visible whitespace between ranking blocks.
    column_format = (
        f"@{{}}r@{{\\,}}p{{{COUNTRY_COLUMN_WIDTH}}}"
        f"r@{{{BLOCK_GAP}}}"
        f"r@{{\\,}}p{{{COUNTRY_COLUMN_WIDTH}}}"
        f"r@{{{BLOCK_GAP}}}"
        f"r@{{\\,}}p{{{COUNTRY_COLUMN_WIDTH}}}r@{{}}"
    )

    latex_label = make_latex_label(var["label"])
    filename_label = latex_label.replace("tab:appendix:", "")

    latex = rf"""% Auto-generated file: do not edit manually.
% Source: {Path(var["file"]).name}
% Metric: {var["metric_col"]}
% Valid country/region observations: {len(table)}
% Ordering: preserved from source CSV; rank follows that ordering.
%
% Required LaTeX preamble packages:
% \usepackage{{booktabs}}
% \usepackage{{changepage}}

\begin{{table}}[p]
\vspace*{{-3cm}}
\begin{{adjustwidth}}{{-{TABLE_MARGIN_EXTENSION}}}{{-{TABLE_MARGIN_EXTENSION}}}
\centering

\caption{{Country-level ranking for {latex_escape(var["label"])}.}}
\label{{{latex_label}}}

\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\renewcommand{{\arraystretch}}{{0.93}}

% The makebox centres tables consistently within the widened local area.
\noindent\makebox[\linewidth][c]{{%
\begin{{tabular}}{{{column_format}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}

\end{{adjustwidth}}
\end{{table}}
"""

    output_file = TABLES_PATH / f"appendix_{filename_label}.tex"
    output_file.write_text(latex, encoding="utf-8")

    print(
        f"  Created {output_file.name}: "
        f"{len(table)} observations, "
        f"{len(blocks)} display blocks."
    )


# ============================================================================
# RUN
# ============================================================================

def main():
    region_name_map = load_region_name_map()

    print("\n[2/2] Generating appendix ranking tables...")

    for var in VARIABLES:
        write_table(var, region_name_map)

    print("\nFinished. Generated LaTeX table files are located in:")
    print(TABLES_PATH)


if __name__ == "__main__":
    main()