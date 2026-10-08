# run_epsilon_paired.py
# Relance, pour plusieurs valeurs d'epsilon, les instances appariées (instance_generator_paired.py)
# où le complete agent avec beam :
#   - partait d'un retard initial nul sur les jobs existants (sum w_j T_j^initial = 0), OU
#   - n'a accepté aucun nouveau job.
# La sélection est faite pour chaque reward sur SES propres résultats avec beam (epsilon = 0.2).
#
# Attention : quand le retard initial est > 0, c'est le critère cost_ref * (1 + delta) qui décide et
# epsilon n'a AUCUN effet. Les cas "aucun job accepté" avec un retard initial > 0 donnent donc le même
# résultat pour tous les epsilon (témoins). --only_zero_ref les exclut.
#
# Sorties, à part des runs de référence :
#   Gantt    : results/paired_epsilon/<reward>/eps_X/test/<size>/delta_Y/<scenario>/<instance>/
#   Analyses : analysis_paired_epsilon/<reward>/eps_X/<size>/delta_Y/<scenario>/<instance>/
#
# Exemple :
#   python3 run_epsilon_paired.py --dry_run
#   python3 run_epsilon_paired.py --skip_existing
#   python3 run_epsilon_paired.py --epsilons 0.1 0.3 --rewards no_cmax --sizes s
import argparse
import os
import time
from pathlib import Path

import pandas as pd

from acceptation_solver_complete_agent import run_one_controlled_instance
from build_analysis_tables import read_order2_csv

DATA_ROOT = "data/controlled_orders_paired/test"

# reward -> (analyses de référence avec beam, poids du GNN)
REWARDS = {
    "basic_reward": ("analysis_paired_complete_agent",         "data/training_costs_ub/"),
    "no_cmax":      ("analysis_paired_complete_agent_no_cmax", "data/training_ub_no_cmax/"),
}

IMPROVE_BY_SIZE = {"s": True, "m": True, "l": False, "xl": False}


def eps_name(eps: float) -> str:
    return "eps_" + f"{eps:g}".replace(".", "_")


def select_cases(reference_root: str, sizes: list[str], only_zero_ref: bool) -> pd.DataFrame:
    rows = []
    for summary_csv in Path(reference_root).glob("*/delta_*/*/instance_*/summary.csv"):
        size, delta_dir, scenario, instance = summary_csv.parts[-5:-1]
        if size not in sizes:
            continue
        summary = pd.read_csv(summary_csv).iloc[0]
        order2 = read_order2_csv(summary_csv.parent / "order_2_acceptance_analysis.csv")
        existing = order2[order2["Pool"] == "Existants"]
        zero_ref = pd.to_numeric(existing["Tj_initial_pondere"]).sum() <= 1e-9
        none_accepted = int(summary["nb_accepted_new_jobs"]) == 0
        if zero_ref or (none_accepted and not only_zero_ref):
            rows.append({
                "size": size,
                "scenario": scenario,
                "instance": instance,
                "delta": float(delta_dir.replace("delta_", "").replace("_", ".")),
                "zero_ref": zero_ref,
                "none_accepted": none_accepted,
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["size", "scenario", "delta", "instance"]) if not df.empty else df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.3, 0.5], help="Valeurs d'epsilon (0.2 = runs de référence, déjà faits)")
    parser.add_argument("--rewards", type=str, nargs="+", choices=list(REWARDS), default=list(REWARDS))
    parser.add_argument("--sizes", type=str, nargs="+", default=["s", "m"])
    parser.add_argument("--only_zero_ref", action="store_true", help="Ne garder que les cas à retard initial nul (les seuls où epsilon agit)")
    parser.add_argument("--generate_gantts", type=str, choices=["true", "false"], default="true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_existing", action="store_true", help="Ne pas relancer un cas dont le summary.csv existe déjà")
    parser.add_argument("--dry_run", action="store_true", help="Afficher la sélection sans lancer")
    args = parser.parse_args()

    done, failed = 0, []
    for reward in args.rewards:
        reference_root, agent_path = REWARDS[reward]
        cases = select_cases(reference_root, args.sizes, args.only_zero_ref)
        n_zero = int(cases["zero_ref"].sum()) if not cases.empty else 0
        n_none = int(cases["none_accepted"].sum()) if not cases.empty else 0
        n_witness = int((cases["none_accepted"] & ~cases["zero_ref"]).sum()) if not cases.empty else 0
        print("\n" + "=" * 80)
        print(f"REWARD : {reward} | référence : {reference_root} | poids : {agent_path}")
        print(f"{len(cases)} cas par epsilon | retard initial nul : {n_zero} | aucun accepté : {n_none} "
              f"| dont témoins (aucun accepté, retard > 0, epsilon sans effet) : {n_witness}")
        print(f"epsilons : {args.epsilons} | tailles : {args.sizes}")
        print("=" * 80)
        if args.dry_run:
            print(cases.groupby(["size", "scenario"]).size().to_string())
            continue

        for eps in args.epsilons:
            output_root = os.path.join("results", "paired_epsilon", reward, eps_name(eps), "test")
            analysis_root = os.path.join("analysis_paired_epsilon", reward, eps_name(eps))
            print(f"\n--- {reward} | epsilon = {eps} -> {analysis_root}")
            for _, c in cases.iterrows():
                delta_name = f"delta_{str(c['delta']).replace('.', '_')}"
                label = f"{reward} eps={eps}: {c['size']}/{c['scenario']}/{c['instance']} delta={c['delta']}"
                summary = os.path.join(analysis_root, c["size"], delta_name, c["scenario"], c["instance"], "summary.csv")
                if args.skip_existing and os.path.exists(summary):
                    print(f"  déjà fait : {label}")
                    continue
                path = os.path.join(DATA_ROOT, c["size"], c["scenario"], f"{c['instance']}.json")
                if not os.path.exists(path):
                    print(f"  ❌ instance introuvable : {path}")
                    failed.append(label)
                    continue
                print(f"  ▶ {label}")
                t = time.perf_counter()
                try:
                    run_one_controlled_instance(
                        input_path=path,
                        output_root=output_root,
                        scenario=c["size"],
                        inst=c["scenario"],
                        delta_ratio=c["delta"],
                        device=args.device,
                        agent_path=agent_path,
                        generate_gantts=args.generate_gantts == "true",
                        use_beam=True,
                        improve=IMPROVE_BY_SIZE.get(c["size"], False),
                        analysis_root=analysis_root,
                        epsilon_window=eps,
                    )
                    done += 1
                    print(f"    ✓ {time.perf_counter() - t:.1f} s")
                except Exception as e:
                    failed.append(label)
                    print(f"    ❌ {type(e).__name__}: {e}")

    if not args.dry_run:
        print(f"\nTerminé : {done} cas lancés, {len(failed)} échecs")
        for f in failed:
            print("   échec :", f)


if __name__ == "__main__":
    main()
