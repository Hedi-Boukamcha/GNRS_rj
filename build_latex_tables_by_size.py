# build_latex_tables_by_size.py
import pandas as pd
from pathlib import Path

# ##########################################################
# =*= ONE SUMMARY TABLE PER SIZE: SCENARIO x TIER ROWS,    =*=
# =*= TOLERANCE x METRIC x {MIN, MAX, AVG} COLUMNS         =*=
# ##########################################################
# Requires the LaTeX packages: booktabs, graphicx
__author__  = "Hedi Boukamcha"
__version__ = "1.0.0"
__license__ = "MIT"

REPO_ROOT   = Path(__file__).resolve().parent
OUTPUT_DIR  = REPO_ROOT / "analysis" / "results_by_size"
RESULTS_DIR = OUTPUT_DIR / "csv"
LATEX_DIR   = OUTPUT_DIR / "latex_by_size"

TABLE_FONT_SIZE = r"\large"

SIZE_LABEL = {"s": "Small", "m": "Medium", "l": "Large", "xl": "X-Large"}

CUT_TIME_ORDER = ["early", "middle", "late"]
CUT_TIME_LABEL = {"early": "Early", "middle": "Middle", "late": "Late"}

SCENARIO_ORDER = ["same_costs", "portion_of_3_7", "portion_of_7_3"]
SCENARIO_LABEL = {
    "same_costs":      "Same costs",
    "portion_of_3_7":  "Cost ratio 3/7",
    "portion_of_7_3":  "Cost ratio 7/3",
}

# Metrics kept in the summary table: (csv column, header, formatter)
# "accepted_over_new" is dropped here (it is a count ratio, not a
# continuous quantity) in favor of "acceptance_ratio" which already
# expresses the same information as a percentage.
METRICS = [
    ("delta_tjE",        r"$\Delta wT_j^E$", lambda x: f"{x:.1f}"),
    ("acceptance_ratio", r"Acc.\ (\%)",       lambda x: f"{100 * x:.0f}"),
    ("computing_time",   r"Time (s)",         lambda x: f"{x:.2f}"),
]

STATS = [
    ("min",  "Min", lambda s: s.min()),
    ("mean", "Avg", lambda s: s.mean()),
    ("max",  "Max", lambda s: s.max()),
]


def load_size_df(size: str) -> pd.DataFrame:
    frames = []
    for csv_file in sorted(RESULTS_DIR.glob(f"results_{size}_*.csv")):
        _, _, *scenario_parts = csv_file.stem.split("_")
        scenario = "_".join(scenario_parts)
        df = pd.read_csv(csv_file)
        df["scenario"] = scenario
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_latex(df: pd.DataFrame, size: str) -> str:
    deltas = sorted(df["delta_ratio"].unique())
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]

    n_stats = len(STATS)
    n_metrics = len(METRICS)
    n_metric_cols = n_metrics * n_stats
    n_delta_cols = n_metric_cols

    # --- column format: 2 label columns + one block of columns per delta ---
    col_format = "ll|" + "|".join("c" * n_delta_cols for _ in deltas)

    # --- header row 1: tolerance level ($\Delta = ...$) ---
    header_delta = " & ".join(
        rf"\multicolumn{{{n_delta_cols}}}{{c}}{{$\Delta={d:g}$}}" for d in deltas
    )
    cmidrules_delta = " ".join(
        rf"\cmidrule(lr){{{3 + i * n_delta_cols}-{2 + (i + 1) * n_delta_cols}}}"
        for i in range(len(deltas))
    )

    # --- header row 2: metric name, repeated under each delta block ---
    header_metric = " & ".join(
        rf"\multicolumn{{{n_stats}}}{{c}}{{{label}}}"
        for _ in deltas for _, label, _ in METRICS
    )
    cmidrules_metric = " ".join(
        rf"\cmidrule(lr){{{3 + k * n_stats}-{2 + (k + 1) * n_stats}}}"
        for k in range(len(deltas) * n_metrics)
    )

    # --- header row 3: Min / Max / Avg ---
    header_stats = " & ".join(
        stat_label for _ in deltas for _ in METRICS for _, stat_label, _ in STATS
    )

    rows = []
    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario]
        for i, tier in enumerate(CUT_TIME_ORDER):
            tier_df = scenario_df[scenario_df["cut_time_pos"] == tier]
            cells = [SCENARIO_LABEL.get(scenario, scenario) if i == 0 else ""]
            cells.append(CUT_TIME_LABEL[tier])
            for delta in deltas:
                cell_df = tier_df[tier_df["delta_ratio"] == delta]
                for col, _, fmt in METRICS:
                    for _, _, agg in STATS:
                        if cell_df.empty:
                            cells.append("--")
                        else:
                            cells.append(fmt(agg(cell_df[col])))
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    size_label = SIZE_LABEL.get(size, size)
    caption = (
        rf"Summary results for {size_label} instances: "
        rf"minimum, maximum and average per scenario and cut-time tier, "
        rf"by acceptance-tolerance level $\Delta$."
    )
    label = f"tab:results_summary_{size}"

    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        TABLE_FONT_SIZE,
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        rf" & & {header_delta} \\",
        cmidrules_delta,
        rf" & & {header_metric} \\",
        cmidrules_metric,
        rf"Scenario & Tier & {header_stats} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    LATEX_DIR.mkdir(exist_ok=True)
    sizes = sorted({f.stem.split("_")[1] for f in RESULTS_DIR.glob("results_*.csv") if f.stem != "results_all_sizes"})

    if not sizes:
        raise RuntimeError(f"No results_<size>_<scenario>.csv files found in {RESULTS_DIR}")

    for size in sizes:
        df = load_size_df(size)
        if df.empty:
            continue
        df["cut_time_pos"] = pd.Categorical(df["cut_time_pos"], categories=CUT_TIME_ORDER, ordered=True)
        tex = build_latex(df, size)
        out_path = LATEX_DIR / f"results_summary_{size}.tex"
        out_path.write_text(tex)
        print(f"[INFO] {out_path}")


if __name__ == "__main__":
    main()
