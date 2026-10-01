# run_no_beam_negative_diff.py
# Relance SANS beam (agent complet avec --beam false) toutes les instances où l'agent complet
# (par défaut : version sans Cmax dans la reward) a AMÉLIORÉ le retard des jobs existants après le cut,
# c.-à-d. delta_tjE < 0 dans le CSV consolidé (somme pondérée des Diff_Tj des jobs existants).
# But : voir si ces améliorations viennent de l'élagage du beam pendant la cédule initiale.
#
# Les résultats sont écrits dans les mêmes dossiers que les tests "no beam" manuels :
#   Gantt    : results/test_no_beam/greedy_ls/test/<size>/delta_X/<cost_variant>/<instance>/
#   Analyses : analysis_test_no_beam/greedy_ls/<size>/delta_X/<cost_variant>/<instance>/
#
# Exemple :
#   python3 run_no_beam_negative_diff.py --dry_run          # liste les instances sans rien lancer
#   python3 run_no_beam_negative_diff.py                    # lance tout
#
# epsilon vaut 0 par défaut : les résultats de référence (analysis_complete_agent_no_cmax, 30 sept.)
# ont été produits avec l'ancien critère d'acceptation (aucune tolérance quand le retard initial est nul).
import argparse
import os
import time

import pandas as pd

from acceptation_solver_complete_agent import run_one_controlled_instance


def instance_path(data_root: str, row) -> str:
    return os.path.join(
        data_root,
        str(row["size"]),
        str(row["cost_variant"]),
        f"instance_{int(row['instance'])}_{row['cut_time_pos']}.json",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, default="analysis_complete_agent_no_cmax/results_by_size/csv/results_all_sizes.csv", help="CSV consolidé de l'agent complet (doit contenir delta_tjE)")
    parser.add_argument("--data_root", type=str, default="data/controlled_orders_ub/test", help="Dossier des instances")
    parser.add_argument("--agent_path", type=str, default="data/training_ub_no_cmax/", help="Poids du GNN (mêmes que ceux du CSV de référence)")
    parser.add_argument("--output_root", type=str, default="results/test_no_beam/greedy_ls/test", help="Dossier des Gantt / résultats")
    parser.add_argument("--analysis_root", type=str, default="analysis_test_no_beam/greedy_ls", help="Dossier des analyses")
    parser.add_argument("--improve", type=str, choices=["true", "false"], default="true", help="Garder la recherche locale (true) ou GNN glouton seul (false)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Tolérance sur la fenêtre promise (0 = ancien critère, comme la référence)")
    parser.add_argument("--generate_gantts", type=str, choices=["true", "false"], default="true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sizes", type=str, nargs="+", default=None, help="Limiter à certaines tailles, ex: --sizes s")
    parser.add_argument("--skip_existing", action="store_true", help="Ne pas relancer une instance dont le summary.csv existe déjà")
    parser.add_argument("--dry_run", action="store_true", help="Afficher la liste sans lancer")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    if "delta_tjE" not in df.columns:
        raise ValueError(f"Colonne delta_tjE absente de {args.results_csv}")
    todo = df[pd.to_numeric(df["delta_tjE"], errors="coerce") < 0].copy()
    if args.sizes:
        todo = todo[todo["size"].isin(args.sizes)]
    todo = todo.sort_values(["size", "cost_variant", "delta_ratio", "instance", "cut_time_pos"])

    print(f"{len(todo)} instances avec une différence de retard négative sur les jobs existants")
    print(f"poids={args.agent_path} | beam=false | improve={args.improve} | epsilon={args.epsilon}")
    print(f"Gantt -> {args.output_root} | analyses -> {args.analysis_root}\n")

    done, failed = 0, []
    for _, row in todo.iterrows():
        path = instance_path(args.data_root, row)
        size, variant, delta = str(row["size"]), str(row["cost_variant"]), float(row["delta_ratio"])
        name = os.path.splitext(os.path.basename(path))[0]
        label = f"{size}/{variant}/{name} delta={delta} (delta_tjE avec beam = {row['delta_tjE']})"
        if args.dry_run:
            print("  ", label)
            continue
        delta_name = f"delta_{str(delta).replace('.', '_')}"
        summary = os.path.join(args.analysis_root, size, delta_name, variant, name, "summary.csv")
        if args.skip_existing and os.path.exists(summary):
            print(f"  déjà fait : {label}")
            continue
        if not os.path.exists(path):
            print(f"  ❌ instance introuvable : {path}")
            failed.append(label)
            continue
        print(f"  ▶ {label}")
        t = time.perf_counter()
        try:
            run_one_controlled_instance(
                input_path=path,
                output_root=args.output_root,
                scenario=size,
                inst=variant,
                delta_ratio=delta,
                device=args.device,
                agent_path=args.agent_path,
                generate_gantts=args.generate_gantts == "true",
                use_beam=False,
                improve=args.improve == "true",
                analysis_root=args.analysis_root,
                epsilon_window=args.epsilon,
            )
            done += 1
            print(f"    ✓ {time.perf_counter() - t:.1f} s")
        except Exception as e:
            failed.append(label)
            print(f"    ❌ {type(e).__name__}: {e}")

    if not args.dry_run:
        print(f"\nTerminé : {done} instances relancées, {len(failed)} échecs")
        for f in failed:
            print("   échec :", f)


if __name__ == "__main__":
    main()
