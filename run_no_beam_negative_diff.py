# run_no_beam_negative_diff.py
# Relance SANS beam (agent complet avec --beam false) toutes les instances où l'agent complet
# (version sans Cmax dans la reward) a AMÉLIORÉ le retard des jobs existants après le cut,
# c.-à-d. delta_tjE < 0 dans le CSV consolidé (somme pondérée des Diff_Tj des jobs existants).
# But : voir si ces améliorations viennent de l'élagage du beam pendant la cédule initiale.
#
# Quatre expériences, lancées toutes par défaut (voir EXPERIMENTS). Agent sans Cmax :
#   - old    : instances data/controlled_orders_ub (same_costs, portion_of_3_7, portion_of_7_3)
#              Gantt    : results/test_no_beam/greedy_ls/test/<size>/delta_X/<cost_variant>/<instance>/
#              Analyses : analysis_test_no_beam/greedy_ls/<size>/delta_X/<cost_variant>/<instance>/
#   - paired : instances data/controlled_orders_paired (same_costs_*/diff_costs_*, instance_generator_paired.py)
#              Gantt    : results/paired_no_beam_negative_diff/no_cmax/test/...
#              Analyses : analysis_paired_no_beam_negative_diff/no_cmax/...
# Agent avec la reward normale (wT + Cmax, poids data/training_costs_ub/), sélection sur ses propres
# résultats avec beam (analysis_complete_agent, analysis_paired_complete_agent) :
#   - old_basic_reward    : analyses dans analysis_test_no_beam/basic_reward/greedy_ls/...
#   - paired_basic_reward : analyses dans analysis_paired_no_beam_negative_diff/basic_reward/...
#
# epsilon n'est PAS un paramètre : chaque expérience garde celui de son run de référence (le test avec
# beam sur toutes les instances de test), sinon la comparaison avec / sans beam mélangerait deux critères.
#   - old, old_basic_reward       : epsilon = 0   (runs de référence des 28-30 sept., ancien critère d'acceptation)
#   - paired, paired_basic_reward : epsilon = 0.2 (runs de référence des 6-8 oct., valeur par défaut du solveur)
# De même, la recherche locale suit les runs de référence : activée pour s et m, désactivée pour l et xl.
#
# Exemple :
#   python3 run_no_beam_negative_diff.py --dry_run                      # liste les instances sans rien lancer
#   python3 run_no_beam_negative_diff.py --skip_existing                # lance tout, sans refaire ce qui existe
#   python3 run_no_beam_negative_diff.py --experiments paired paired_basic_reward   # seulement les appariées
import argparse
import os
import time

import pandas as pd

from acceptation_solver_complete_agent import run_one_controlled_instance

EXPERIMENTS = {
    "old": {
        "results_csv":   "analysis_complete_agent_no_cmax/results_by_size/csv/results_all_sizes.csv",
        "data_root":     "data/controlled_orders_ub/test",
        "agent_path":    "data/training_ub_no_cmax/",
        "output_root":   "results/test_no_beam/greedy_ls/test",
        "analysis_root": "analysis_test_no_beam/greedy_ls",
        "epsilon":       0.0,
    },
    "paired": {
        "results_csv":   "analysis_paired_complete_agent_no_cmax/results_by_size/csv/results_all_sizes.csv",
        "data_root":     "data/controlled_orders_paired/test",
        "agent_path":    "data/training_ub_no_cmax/",
        "output_root":   "results/paired_no_beam_negative_diff/no_cmax/test",
        "analysis_root": "analysis_paired_no_beam_negative_diff/no_cmax",
        "epsilon":       0.2,
    },
    # agent avec la reward normale (wT + Cmax), sélection faite sur SES propres résultats avec beam
    "old_basic_reward": {
        "results_csv":   "analysis_complete_agent/results_by_size/csv/results_all_sizes.csv",
        "data_root":     "data/controlled_orders_ub/test",
        "agent_path":    "data/training_costs_ub/",
        "output_root":   "results/test_no_beam/basic_reward/greedy_ls/test",
        "analysis_root": "analysis_test_no_beam/basic_reward/greedy_ls",
        "epsilon":       0.0,
    },
    "paired_basic_reward": {
        "results_csv":   "analysis_paired_complete_agent/results_by_size/csv/results_all_sizes.csv",
        "data_root":     "data/controlled_orders_paired/test",
        "agent_path":    "data/training_costs_ub/",
        "output_root":   "results/paired_no_beam_negative_diff/basic_reward/test",
        "analysis_root": "analysis_paired_no_beam_negative_diff/basic_reward",
        "epsilon":       0.2,
    },
}

# recherche locale des runs de référence avec beam, par taille
IMPROVE_BY_SIZE = {"s": True, "m": True, "l": False, "xl": False}


def instance_path(data_root: str, row) -> str:
    return os.path.join(
        data_root,
        str(row["size"]),
        str(row["cost_variant"]),
        f"instance_{int(row['instance'])}_{row['cut_time_pos']}.json",
    )


def run_experiment(name: str, exp: dict, args) -> tuple[int, list[str]]:
    print("\n" + "=" * 80)
    print(f"EXPÉRIENCE : {name}")
    print("=" * 80)

    if not os.path.exists(exp["results_csv"]):
        print(f"❌ CSV de référence introuvable : {exp['results_csv']} (lancer build_analysis_tables.py) -> ignorée")
        return 0, [f"{name}: CSV absent"]

    df = pd.read_csv(exp["results_csv"])
    if "delta_tjE" not in df.columns:
        raise ValueError(f"Colonne delta_tjE absente de {exp['results_csv']}")
    todo = df[pd.to_numeric(df["delta_tjE"], errors="coerce") < 0].copy()
    if args.sizes:
        todo = todo[todo["size"].isin(args.sizes)]
    todo = todo.sort_values(["size", "cost_variant", "delta_ratio", "instance", "cut_time_pos"])

    print(f"{len(todo)} instances avec une différence de retard négative sur les jobs existants")
    print(f"poids={exp['agent_path']} | beam=false | improve=s,m oui / l,xl non | epsilon={exp['epsilon']}")
    print(f"Gantt -> {exp['output_root']} | analyses -> {exp['analysis_root']}\n")

    done, failed = 0, []
    for _, row in todo.iterrows():
        path = instance_path(exp["data_root"], row)
        size, variant, delta = str(row["size"]), str(row["cost_variant"]), float(row["delta_ratio"])
        improve = IMPROVE_BY_SIZE.get(size, False)
        inst_name = os.path.splitext(os.path.basename(path))[0]
        label = f"{name}: {size}/{variant}/{inst_name} delta={delta} (delta_tjE avec beam = {row['delta_tjE']})"
        if args.dry_run:
            print("  ", label)
            continue
        delta_name = f"delta_{str(delta).replace('.', '_')}"
        summary = os.path.join(exp["analysis_root"], size, delta_name, variant, inst_name, "summary.csv")
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
                output_root=exp["output_root"],
                scenario=size,
                inst=variant,
                delta_ratio=delta,
                device=args.device,
                agent_path=exp["agent_path"],
                generate_gantts=args.generate_gantts == "true",
                use_beam=False,
                improve=improve,
                analysis_root=exp["analysis_root"],
                epsilon_window=exp["epsilon"],
            )
            done += 1
            print(f"    ✓ {time.perf_counter() - t:.1f} s")
        except Exception as e:
            failed.append(label)
            print(f"    ❌ {type(e).__name__}: {e}")
    return done, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS), help="Expériences à lancer (défaut : toutes)")
    parser.add_argument("--generate_gantts", type=str, choices=["true", "false"], default="true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sizes", type=str, nargs="+", default=None, help="Limiter à certaines tailles, ex: --sizes s")
    parser.add_argument("--skip_existing", action="store_true", help="Ne pas relancer une instance dont le summary.csv existe déjà")
    parser.add_argument("--dry_run", action="store_true", help="Afficher la liste sans lancer")
    args = parser.parse_args()

    total_done, total_failed = 0, []
    for name in args.experiments:
        done, failed = run_experiment(name, EXPERIMENTS[name], args)
        total_done += done
        total_failed += failed

    if not args.dry_run:
        print(f"\nTerminé : {total_done} instances relancées, {len(total_failed)} échecs")
        for f in total_failed:
            print("   échec :", f)


if __name__ == "__main__":
    main()
