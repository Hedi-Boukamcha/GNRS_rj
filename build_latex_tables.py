# build_latex_tables.py
import pandas as pd
from pathlib import Path

# ##########################################################
# =*= TURN THE PER-SIZE/SCENARIO RESULT TABLES INTO LATEX =*=
# ##########################################################
__author__  = "Hedi Boukamcha"
__version__ = "2.0.0"
__license__ = "MIT"

REPO_ROOT   = Path(__file__).resolve().parent
OUTPUT_DIR  = REPO_ROOT / "analysis" / "results_by_size"
RESULTS_DIR = OUTPUT_DIR / "csv"
LATEX_DIR   = OUTPUT_DIR / "latex"

CUT_TIME_ORDER = ["early", "middle", "late"]
CUT_TIME_LABEL = {"early": "Early", "middle": "Middle", "late": "Late"}

TABLE_FONT_SIZE = r"\normalsize"

SIZE_LABEL = {"s": "Small", "m": "Medium", "l": "Large", "xl": "X-Large"}
SCENARIO_LABEL = {
    "same_costs":      "same costs",
    "portion_of_3_7":  "cost ratio 3/7",
    "portion_of_7_3":  "cost ratio 7/3",
}

# Metrics kept in the paper table: (csv column, header, formatter)
METRICS = [
    ("delta_tjE",             r"$\Delta wT_j^E$",  lambda x: f"{x:.1f}"),
    ("accepted_over_new",     r"Nb. acc.",         lambda x: str(x)),
    ("acceptance_ratio",      r"Acc.\ (\%)",       lambda x: f"{100 * x:.0f}"),
    ("computing_time",        r"Time (s)",         lambda x: f"{x:.2f}"),
]


def build_latex(df_delta: pd.DataFrame, size: str, scenario: str, delta: float) -> str:
    instances = sorted(df_delta["instance"].unique())
    n_metrics = len(METRICS)
    n_zones = len(CUT_TIME_ORDER)

    col_format = "l|" + "|".join("c" * n_metrics for _ in range(n_zones))

    header_zones = " & ".join(
        rf"\multicolumn{{{n_metrics}}}{{{'c|' if i < n_zones - 1 else 'c'}}}{{{CUT_TIME_LABEL[z]}}}"
        for i, z in enumerate(CUT_TIME_ORDER)
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i * n_metrics}-{1 + (i + 1) * n_metrics}}}" for i in range(n_zones)
    )
    header_metrics = " & ".join(m[1] for _ in CUT_TIME_ORDER for m in METRICS)

    rows = []
    for inst in instances:
        cells = [str(inst)]
        for zone in CUT_TIME_ORDER:
            match = df_delta[(df_delta["instance"] == inst) & (df_delta["cut_time_pos"] == zone)]
            for col, _, fmt in METRICS:
                cells.append(fmt(match[col].iloc[0]) if not match.empty else "--")
        rows.append(" & ".join(cells) + r" \\")

    total_cells = ["Avg"]
    for zone in CUT_TIME_ORDER:
        zone_df = df_delta[df_delta["cut_time_pos"] == zone]
        avg_acceptance_pct = rf"{100 * zone_df['acceptance_ratio'].mean():.0f}\%"
        for col, _, fmt in METRICS:
            if col == "accepted_over_new":
                total_cells.append(f"{zone_df['nb_accepted_new_jobs'].sum():.0f}/{zone_df['nb_new_jobs'].sum():.0f}")
            elif col == "acceptance_ratio":
                total_cells.append(avg_acceptance_pct)
            elif col == "delta_tjE":
                total_cells.append(fmt(zone_df["delta_tjE"].mean()))
            elif col == "computing_time":
                total_cells.append(fmt(zone_df["computing_time"].mean()))
            else:
                total_cells.append("--")
    total_row = " & ".join(total_cells) + r" \\"

    size_label = SIZE_LABEL.get(size, size)
    scenario_label = SCENARIO_LABEL.get(scenario, scenario.replace("_", " "))
    caption = (
        rf"Results per instance "
        rf"({size_label} instances, {scenario_label} scenario, $\Delta={delta:g}$) "
        rf"by cut-time position."
    )
    delta_tag = f"{delta:g}".replace(".", "_")
    label = f"tab:results_{size}_{scenario}_delta_{delta_tag}"

    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        TABLE_FONT_SIZE,
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        rf" & {header_zones} \\",
        cmidrules,
        rf"Inst. & {header_metrics} \\",
        r"\midrule",
        *rows,
        r"\midrule",
        total_row,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    LATEX_DIR.mkdir(exist_ok=True)
    csv_files = sorted(RESULTS_DIR.glob("results_*.csv"))
    csv_files = [f for f in csv_files if f.stem != "results_all_sizes"]

    if not csv_files:
        raise RuntimeError(f"No results_<size>_<scenario>.csv files found in {RESULTS_DIR}")

    for csv_file in csv_files:
        _, size, *scenario_parts = csv_file.stem.split("_")
        scenario = "_".join(scenario_parts)
        df = pd.read_csv(csv_file)
        df["cut_time_pos"] = pd.Categorical(df["cut_time_pos"], categories=CUT_TIME_ORDER, ordered=True)

        for delta in sorted(df["delta_ratio"].unique()):
            df_delta = df[df["delta_ratio"] == delta]
            tex = build_latex(df_delta, size, scenario, delta)
            delta_tag = f"{delta:g}".replace(".", "_")
            out_path = LATEX_DIR / f"{csv_file.stem}_delta_{delta_tag}.tex"
            out_path.write_text(tex)
            print(f"[INFO] {out_path}")


if __name__ == "__main__":
    main()
