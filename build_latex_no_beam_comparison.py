# build_latex_no_beam_comparison.py
# Table LaTeX de comparaison AVEC / SANS beam pour les instances relancées par run_no_beam_negative_diff.py
# (celles où l'agent complet avec beam améliorait le retard des jobs existants après le cut, delta_tjE < 0).
# Une table par taille ; une ligne par (scénario, instance, tier, Delta) ; pour chaque métrique,
# une colonne "Beam" et une colonne "No beam".
#
# Entrées :
#   - avec beam : <beam_root>/results_by_size/csv/results_all_sizes.csv   (build_analysis_tables.py)
#   - sans beam : <no_beam_root>/<size>/delta_X/<scenario>/<instance>/summary.csv + order_2_acceptance_analysis.csv
# Sorties :
#   - <no_beam_root>/results_by_size/latex/no_beam_comparison_<size>.tex
#   - <no_beam_root>/results_by_size/csv/no_beam_comparison.csv
#
# RUN : python3 build_latex_no_beam_comparison.py
# Packages LaTeX requis : booktabs, graphicx
import argparse
from pathlib import Path

import pandas as pd

from build_analysis_tables import build_master_table  # même calcul de delta_tjE que les autres tables

REPO_ROOT = Path(__file__).resolve().parent

TABLE_FONT_SIZE = r"\normalsize"
SIZE_LABEL = {"s": "Small", "m": "Medium", "l": "Large", "xl": "X-Large"}
CUT_TIME_ORDER = ["early", "middle", "late"]
CUT_TIME_LABEL = {"early": "Early", "middle": "Middle", "late": "Late"}
SCENARIO_ORDER = ["same_costs", "portion_of_3_7", "portion_of_7_3"]
SCENARIO_LABEL = {"same_costs": "Same costs", "portion_of_3_7": "Cost ratio 3/7", "portion_of_7_3": "Cost ratio 7/3"}

KEYS = ["size", "cost_variant", "delta_ratio", "instance", "cut_time_pos"]

# (colonne, en-tête, format)
METRICS = [
    ("delta_tjE",          r"$\Delta wT_j^E$", lambda x: f"{x:.1f}"),
    ("accepted_over_new",  r"Nb.\ acc.",       lambda x: str(x)),
    ("weighted_tardiness", r"$wT$ final",      lambda x: f"{x:.1f}"),
    ("computing_time",     r"Time (s)",        lambda x: f"{x:.2f}"),
]


def load(beam_root: Path, no_beam_root: Path) -> pd.DataFrame:
    beam = pd.read_csv(beam_root / "results_by_size" / "csv" / "results_all_sizes.csv")
    no_beam = build_master_table(no_beam_root)
    cols = KEYS + [m for m, _, _ in METRICS]
    beam = beam[cols].copy()
    no_beam = no_beam[cols].copy()
    for df in (beam, no_beam):
        df["cut_time_pos"] = df["cut_time_pos"].astype(str)
        df["instance"] = df["instance"].astype(int)
    merged = no_beam.merge(beam, on=KEYS, suffixes=("_nobeam", "_beam"), how="left")
    # On ne garde que les cas sélectionnés par run_no_beam_negative_diff.py (delta_tjE < 0 avec beam).
    # Le dossier sans beam peut contenir d'autres tests manuels (ex. lancés avec d'autres poids).
    selected = pd.to_numeric(merged["delta_tjE_beam"], errors="coerce") < 0
    for _, r in merged[~selected].iterrows():
        print(f"[WARN] ignoré (pas dans la sélection delta_tjE < 0 avec beam) : "
              f"{r['size']}/{r['cost_variant']}/instance_{int(r['instance'])}_{r['cut_time_pos']} delta={r['delta_ratio']:g}")
    merged = merged[selected].copy()
    merged["scenario_rank"] = merged["cost_variant"].map({s: i for i, s in enumerate(SCENARIO_ORDER)})
    merged["tier_rank"] = merged["cut_time_pos"].map({t: i for i, t in enumerate(CUT_TIME_ORDER)})
    merged = merged.sort_values(["size", "scenario_rank", "instance", "tier_rank", "delta_ratio"])
    return merged.drop(columns=["scenario_rank", "tier_rank"]).reset_index(drop=True)


def cell(row, col, suffix, fmt):
    v = row.get(f"{col}_{suffix}")
    return "--" if pd.isna(v) else fmt(v)


def build_latex(df: pd.DataFrame, size: str, beam_title: str) -> str:
    n_m = len(METRICS)
    col_format = "lllc|" + "|".join("cc" for _ in range(n_m))
    header_metric = " & ".join(rf"\multicolumn{{2}}{{c}}{{{label}}}" for _, label, _ in METRICS)
    cmidrules = " ".join(rf"\cmidrule(lr){{{5 + 2 * k}-{6 + 2 * k}}}" for k in range(n_m))
    header_sub = " & ".join("Beam & No beam" for _ in METRICS)

    rows, prev_scenario = [], None
    for _, r in df.iterrows():
        scenario = r["cost_variant"]
        if prev_scenario is not None and scenario != prev_scenario:
            rows.append(r"\midrule")
        cells = [
            SCENARIO_LABEL.get(scenario, scenario) if scenario != prev_scenario else "",
            str(int(r["instance"])),
            CUT_TIME_LABEL.get(r["cut_time_pos"], r["cut_time_pos"]),
            f"{r['delta_ratio']:g}",
        ]
        for col, _, fmt in METRICS:
            cells += [cell(r, col, "beam", fmt), cell(r, col, "nobeam", fmt)]
        rows.append(" & ".join(cells) + r" \\")
        prev_scenario = scenario

    # ligne de moyenne pour les métriques numériques
    avg = [r"\midrule" + "\n" + "Avg", "", "", ""]
    for col, _, fmt in METRICS:
        for suffix in ("beam", "nobeam"):
            vals = pd.to_numeric(df[f"{col}_{suffix}"], errors="coerce")
            avg.append("--" if col == "accepted_over_new" or vals.isna().all() else fmt(vals.mean()))
    rows.append(" & ".join(avg) + r" \\")

    size_label = SIZE_LABEL.get(size, size)
    caption = (
        rf"{beam_title}: with vs without beam search, {size_label} instances. "
        rf"Only instances where the beam version improved the existing jobs after the cut "
        rf"($\Delta wT_j^E < 0$) were re-run without beam (local search kept)."
    )
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        TABLE_FONT_SIZE,
        rf"\caption{{{caption}}}",
        rf"\label{{tab:no_beam_comparison_{size}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_format}}}",
        r"\toprule",
        rf" & & & & {header_metric} \\",
        cmidrules,
        rf"Scenario & Inst. & Tier & $\Delta$ & {header_sub} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_root", type=str, default="analysis_complete_agent_no_cmax", help="Analyses de l'agent complet avec beam")
    parser.add_argument("--no_beam_root", type=str, default="analysis_test_no_beam/greedy_ls", help="Analyses des relances sans beam")
    parser.add_argument("--beam_title", type=str, default=r"Complete agent (reward $wT$ only)", help="Nom de l'agent affiché dans le titre")
    args = parser.parse_args()

    beam_root, no_beam_root = REPO_ROOT / args.beam_root, REPO_ROOT / args.no_beam_root
    df = load(beam_root, no_beam_root)

    csv_dir = no_beam_root / "results_by_size" / "csv"
    latex_dir = no_beam_root / "results_by_size" / "latex"
    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_dir / "no_beam_comparison.csv", index=False)
    print(f"[INFO] {(csv_dir / 'no_beam_comparison.csv').relative_to(REPO_ROOT)} -> {len(df)} lignes")

    for size in sorted(df["size"].unique()):
        out = latex_dir / f"no_beam_comparison_{size}.tex"
        out.write_text(build_latex(df[df["size"] == size], size, args.beam_title))
        print(f"[INFO] {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
