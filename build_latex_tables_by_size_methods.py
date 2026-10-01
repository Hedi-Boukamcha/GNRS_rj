# build_latex_tables_by_size_methods.py
# Même table que build_latex_tables_by_size.py (une table par taille : lignes scénario x tier,
# colonnes Delta x métrique x {Min, Avg, Max}), mais pour PLUSIEURS versions d'agent.
# Chaque table porte le nom de la version dans son titre (caption), son label LaTeX et son nom de fichier,
# pour pouvoir les mettre côte à côte dans un document sans ambiguïté.
#
# Entrée  : <analysis_root>/results_by_size/csv/results_<size>_<scenario>.csv  (produits par build_analysis_tables.py)
# Sortie  : <analysis_root>/results_by_size/latex_by_size/results_summary_<method>_<size>.tex
#
# Exemples :
#   python3 build_latex_tables_by_size_methods.py                       # toutes les versions de METHODS trouvées
#   python3 build_latex_tables_by_size_methods.py --methods complete_agent_no_cmax
#   python3 build_latex_tables_by_size_methods.py --root analysis_test_no_beam/greedy_ls --name greedy_ls --title "Complete agent without beam"
# Packages LaTeX requis : booktabs, graphicx
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

# nom court (fichiers / labels) -> (dossier d'analyse, titre affiché dans la caption)
METHODS = {
    "greedy_gnn":             ("analysis",                        "Greedy GNN"),
    "complete_agent":         ("analysis_complete_agent",         r"Complete agent (reward $wT + C_{\max}$)"),
    "complete_agent_no_cmax": ("analysis_complete_agent_no_cmax", r"Complete agent (reward $wT$ only)"),
}

TABLE_FONT_SIZE = r"\large"

SIZE_LABEL = {"s": "Small", "m": "Medium", "l": "Large", "xl": "X-Large"}

CUT_TIME_ORDER = ["early", "middle", "late"]
CUT_TIME_LABEL = {"early": "Early", "middle": "Middle", "late": "Late"}

SCENARIO_ORDER = ["same_costs", "portion_of_3_7", "portion_of_7_3"]
SCENARIO_LABEL = {
    "same_costs":     "Same costs",
    "portion_of_3_7": "Cost ratio 3/7",
    "portion_of_7_3": "Cost ratio 7/3",
}

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


def load_size_df(csv_dir: Path, size: str) -> pd.DataFrame:
    frames = []
    for csv_file in sorted(csv_dir.glob(f"results_{size}_*.csv")):
        _, _, *scenario_parts = csv_file.stem.split("_")
        df = pd.read_csv(csv_file)
        df["scenario"] = "_".join(scenario_parts)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_latex(df: pd.DataFrame, size: str, method_name: str, method_title: str) -> str:
    deltas = sorted(df["delta_ratio"].unique())
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]

    n_stats = len(STATS)
    n_delta_cols = len(METRICS) * n_stats
    col_format = "ll|" + "|".join("c" * n_delta_cols for _ in deltas)

    header_delta = " & ".join(rf"\multicolumn{{{n_delta_cols}}}{{c}}{{$\Delta={d:g}$}}" for d in deltas)
    cmidrules_delta = " ".join(
        rf"\cmidrule(lr){{{3 + i * n_delta_cols}-{2 + (i + 1) * n_delta_cols}}}" for i in range(len(deltas)))
    header_metric = " & ".join(
        rf"\multicolumn{{{n_stats}}}{{c}}{{{label}}}" for _ in deltas for _, label, _ in METRICS)
    cmidrules_metric = " ".join(
        rf"\cmidrule(lr){{{3 + k * n_stats}-{2 + (k + 1) * n_stats}}}" for k in range(len(deltas) * len(METRICS)))
    header_stats = " & ".join(stat for _ in deltas for _ in METRICS for _, stat, _ in STATS)

    rows = []
    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario]
        for i, tier in enumerate(CUT_TIME_ORDER):
            tier_df = scenario_df[scenario_df["cut_time_pos"] == tier]
            cells = [SCENARIO_LABEL.get(scenario, scenario) if i == 0 else "", CUT_TIME_LABEL[tier]]
            for delta in deltas:
                cell_df = tier_df[tier_df["delta_ratio"] == delta]
                for col, _, fmt in METRICS:
                    for _, _, agg in STATS:
                        cells.append("--" if cell_df.empty else fmt(agg(cell_df[col])))
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()

    size_label = SIZE_LABEL.get(size, size)
    caption = (
        rf"{method_title} -- summary results for {size_label} instances: "
        rf"minimum, average and maximum per scenario and cut-time tier, "
        rf"by acceptance-tolerance level $\Delta$."
    )
    label = f"tab:results_summary_{method_name}_{size}"

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


def build_for_method(method_name: str, analysis_root: str, method_title: str) -> int:
    csv_dir = REPO_ROOT / analysis_root / "results_by_size" / "csv"
    if not csv_dir.is_dir():
        print(f"[SKIP] {method_name} : {csv_dir} introuvable (lancer build_analysis_tables.py d'abord)")
        return 0
    latex_dir = REPO_ROOT / analysis_root / "results_by_size" / "latex_by_size"
    latex_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted({f.stem.split("_")[1] for f in csv_dir.glob("results_*.csv") if f.stem != "results_all_sizes"})
    n = 0
    for size in sizes:
        df = load_size_df(csv_dir, size)
        if df.empty:
            continue
        df["cut_time_pos"] = pd.Categorical(df["cut_time_pos"], categories=CUT_TIME_ORDER, ordered=True)
        out_path = latex_dir / f"results_summary_{method_name}_{size}.tex"
        out_path.write_text(build_latex(df, size, method_name, method_title))
        print(f"[INFO] {out_path.relative_to(REPO_ROOT)}")
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None, help=f"Versions à traiter parmi {list(METHODS)} (défaut : toutes)")
    parser.add_argument("--root", type=str, default=None, help="Autre dossier d'analyse à traiter (avec --name et --title)")
    parser.add_argument("--name", type=str, default=None, help="Nom court utilisé dans le fichier et le label")
    parser.add_argument("--title", type=str, default=None, help="Titre affiché dans la caption")
    args = parser.parse_args()

    if args.root:
        if not (args.name and args.title):
            raise ValueError("--root demande aussi --name et --title")
        jobs = [(args.name, args.root, args.title)]
    else:
        names = args.methods or list(METHODS)
        unknown = [m for m in names if m not in METHODS]
        if unknown:
            raise ValueError(f"Versions inconnues : {unknown}. Choix : {list(METHODS)}")
        jobs = [(m, METHODS[m][0], METHODS[m][1]) for m in names]

    total = sum(build_for_method(*job) for job in jobs)
    print(f"[INFO] {total} tables générées")


if __name__ == "__main__":
    main()
