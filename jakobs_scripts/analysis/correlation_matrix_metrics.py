import pandas as pd
from pathlib import Path
from scipy import stats


# ============================================================================
# CONFIGURATION — fill in filenames and column names
# ============================================================================

# Main project directory
BASE_PATH = Path("/home/user/Documents/University/Master Thesis")

# Analysis-specific directories
VULNERABILITY_PATH = BASE_PATH / "vulnerability_analysis"
THREAT_PATH = BASE_PATH / "threat_analysis"

# Output CSV location
OUTPUT_FILE = BASE_PATH / "vulnerability_analysis/correlation_matrix_all_metrics.csv"

# "pairwise": Each correlation uses all regions with valid values for that
# particular pair of metrics.
# "complete_case": Every correlation uses the same regions: only regions
# with valid values for every configured metric are retained.
MISSING_DATA_POLICY = "pairwise"

# Number of decimal places for correlation coefficients in the output.
DECIMALS = 4


# ============================================================================
# VARIABLES — fill in CSV filenames and metric-column names
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
        "label": "Direct IS-V",
        "file": VULNERABILITY_PATH / "extended_vulnerability_analysis.csv",
        "id_col": "iso2",
        "metric_col": "direct_loss_to_output_ratio",
    },

    {
        "label": "Indirect IS-V",
        "file": VULNERABILITY_PATH / "extended_vulnerability_analysis.csv",
        "id_col": "iso",
        "metric_col": "indirect_loss_to_output_ratio",
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

    #{
    #    "label": "VS-V",
    #    "file": VULNERABILITY_PATH / "",
    #    "id_col": "",
    #    "metric_col": "",
    #},

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

    # ========================================================================
    # COUNTRY-LEVEL EXPLANATORY VARIABLES
    # Place these in the most appropriate directory, or use BASE_PATH directly.
    # ========================================================================

    {
        "label": "Total Output",
        "file": VULNERABILITY_PATH / "total_region_output.csv",
        "id_col": "iso",
        "metric_col": "total_output",
    },

    {
        "label": "Domestic Sector Linkages",
        "file": BASE_PATH / "supply_concentrations/domestic_linkages_by_country_gt_0p001.csv",
        "id_col": "country_iso",
        "metric_col": "avg_count_per_supplier_sector",
    },

    {
        "label": "GDP",
        "file": VULNERABILITY_PATH / "total_region_final_demand.csv",
        "id_col": "iso",
        "metric_col": "total_final_demand",
    },

    #{
    #    "label": "Manufacturing Export Share",
    #    "file": BASE_PATH / "",
    #    "id_col": "",
    #   "metric_col": "",
    #},

    #{
    #    "label": "Manufacturing Final Demand Share",
    #    "file": BASE_PATH / "",
    #    "id_col": "",
    #    "metric_col": "",
    #},

    #{
    #    "label": "Construction Final Demand Share",
    #    "file": BASE_PATH / "",
    #    "id_col": "",
    #    "metric_col": "",
    #},

    #{
    #    "label": "Service Final Demand Share",
    #    "file": BASE_PATH / "",
    #    "id_col": "",
    #    "metric_col": "",
    #},
]

# ============================================================================
# MAIN SCRIPT
# ============================================================================


def validate_config(variables, output_file):
    if not output_file:
        raise ValueError("Set OUTPUT_FILE to the path of the output CSV.")

    if len(variables) < 2:
        raise ValueError("Add at least two variables to VARIABLES.")

    labels = []
    for i, var in enumerate(variables, start=1):
        required = ["label", "file", "id_col", "metric_col"]
        missing_keys = [key for key in required if key not in var]
        if missing_keys:
            raise ValueError(f"Variable {i} is missing configuration keys: {missing_keys}")

        blank_fields = [key for key in required if not str(var[key]).strip()]
        if blank_fields:
            raise ValueError(f"Fill in variable {i}: {', '.join(blank_fields)}")

        labels.append(var["label"])

    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Variable labels must be unique. Duplicates: {duplicates}")


def load_metric(var):
    file_path = Path(var["file"]).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    required_cols = [var["id_col"], var["metric_col"]]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(
            f"Missing column(s) {missing_cols} in {file_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[var["id_col"], var["metric_col"]]].copy()
    out.columns = ["id", var["label"]]
    out["id"] = out["id"].astype(str).str.strip()
    out[var["label"]] = pd.to_numeric(out[var["label"]], errors="coerce")
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["id", var["label"]])

    duplicate_ids = out["id"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = out.loc[duplicate_ids, "id"].head(10).tolist()
        raise ValueError(
            f"{file_path.name} has duplicate IDs after cleaning, e.g. {examples}. "
            "Aggregate or otherwise make IDs unique before running correlations."
        )

    return out


def pearson_pair(x, y):
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)

    if n < 3:
        return float("nan"), float("nan"), n

    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan"), float("nan"), n

    r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
    return r, p, n


def format_r_p(r, p, decimals):
    if pd.isna(r):
        return "NA"

    if p < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p:.3f}"

    return f"(Pearson's $r$ = {r:.{decimals}f}, {p_text})"


def main():
    try:
        validate_config(VARIABLES, OUTPUT_FILE)

        if MISSING_DATA_POLICY not in {"pairwise", "complete_case"}:
            raise ValueError("MISSING_DATA_POLICY must be 'pairwise' or 'complete_case'.")

        merged = None
        for var in VARIABLES:
            metric_df = load_metric(var)
            print(
                f"Loaded {len(metric_df)} valid observations for {var['label']} "
                f"from {Path(var['file']).name}"
            )

            if merged is None:
                merged = metric_df
            else:
                merged = merged.merge(metric_df, on="id", how="outer", validate="one_to_one")

        labels = [var["label"] for var in VARIABLES]

        if MISSING_DATA_POLICY == "complete_case":
            analysis_df = merged.dropna(subset=labels).copy()
            if len(analysis_df) < 3:
                raise ValueError(
                    "Fewer than three complete observations remain after combining all variables."
                )
        else:
            analysis_df = merged.copy()

        correlation = pd.DataFrame(index=labels, columns=labels, dtype=float)
        p_values = pd.DataFrame(index=labels, columns=labels, dtype=float)
        sample_sizes = pd.DataFrame(index=labels, columns=labels, dtype="Int64")
        formatted = pd.DataFrame(index=labels, columns=labels, dtype=object)

        for row_label in labels:
            for col_label in labels:
                if row_label == col_label:
                    n = analysis_df[row_label].notna().sum()
                    correlation.loc[row_label, col_label] = 1.0
                    p_values.loc[row_label, col_label] = 0.0
                    sample_sizes.loc[row_label, col_label] = n
                    formatted.loc[row_label, col_label] = "(Pearson's $r$ = 1.0000, p = 0.000)"
                else:
                    r, p, n = pearson_pair(
                        analysis_df[row_label],
                        analysis_df[col_label]
                    )
                    correlation.loc[row_label, col_label] = r
                    p_values.loc[row_label, col_label] = p
                    sample_sizes.loc[row_label, col_label] = n
                    formatted.loc[row_label, col_label] = format_r_p(r, p, DECIMALS)

        output_path = Path(OUTPUT_FILE).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # A single CSV with clearly separated matrices. The final block is
        # ready to copy from when writing results paragraphs.
        blocks = [
            "PEARSON_CORRELATION_COEFFICIENTS",
            correlation.round(DECIMALS).to_csv(),
            "PEARSON_P_VALUES",
            p_values.round(DECIMALS).to_csv(),
            "PAIRWISE_SAMPLE_SIZES",
            sample_sizes.to_csv(),
            "THESIS_READY_CORRELATIONS",
            formatted.to_csv(),
        ]
        output_path.write_text("\n".join(blocks), encoding="utf-8")

        print("\n" + "=" * 72)
        print(f"Pearson correlation matrix ({MISSING_DATA_POLICY} missing-data policy)")
        print("=" * 72)
        print(correlation.round(DECIMALS).to_string())

        print("\nThesis-ready correlation matrix:")
        print(formatted.to_string())

        print("\n" + "=" * 72)
        print(f"Saved CSV: {output_path.resolve()}")
        print("The CSV contains coefficients, p-values, sample sizes, and")
        print("a matrix formatted as: (Pearson's $r$ = ..., p = ...).")
        print("=" * 72)

    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
