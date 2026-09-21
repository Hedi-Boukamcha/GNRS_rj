import os
import csv
import time
import argparse
from pathlib import Path
import argparse

from models.order import OrderInstance
from models.agent import Agent
from master_problem.acceptation_method import acceptation_method
from gantt_builder.gnn_gantt import gnn_gantt



def compute_weighted_tardiness(final_state):
    return sum(
        float(getattr(j.job, "cost", 1)) * j.delay
        for j in final_state.job_states
    )


def run_one_controlled_instance(
    input_path: str,
    output_root: str,
    scenario: str,
    inst: str,
    delta_ratio: float,
    device: str,
    agent_path: str,
    save_step_gantts: bool = False,
    generate_gantts: bool = True
):
    variant = os.path.splitext(os.path.basename(input_path))[0]
    delta_name = f"delta_{str(delta_ratio).replace('.', '_')}"

    output_dir = os.path.join(
        output_root,
        scenario,
        delta_name,
        inst,
        variant
    )

    analysis_dir = os.path.join(
        "analysis",
        scenario,
        delta_name,
        inst,
        variant
    )

    gantt_dir = os.path.join(
        output_dir,
        "gantts"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    if generate_gantts:
        os.makedirs(gantt_dir, exist_ok=True)
    else:
        gantt_dir = None

    print("\n==================================================")
    print(f"Instance testée : {input_path}")
    print(f"Scenario        : {scenario}")
    print(f"Inst            : {inst}")
    print(f"Variant         : {variant}")
    print(f"Delta           : {delta_ratio}")
    print(f"Output dir      : {output_dir}")
    print(f"Analysis dir    : {analysis_dir}")
    print("==================================================")

    order_instance = OrderInstance.load(input_path)

    agent = Agent(
        device=device,
        interactive=False,
        load=True,
        path=agent_path,
        train=False,
        custom=True
    )

    start = time.perf_counter()

    final_state = acceptation_method(
        order_instance=order_instance,
        agent=agent,
        device=device,
        delta_ratio=delta_ratio,
        gantt_dir=gantt_dir,
        save_step_gantts=save_step_gantts,
        analysis_dir=analysis_dir
    )

    computing_time = time.perf_counter() - start

    total_tardiness = sum(j.delay for j in final_state.job_states)
    weighted_tardiness = compute_weighted_tardiness(final_state)
    cmax = final_state.cmax
    obj = cmax + weighted_tardiness
    
    initial_jobs = len(order_instance.orders[0].jobs)
    nb_new_jobs = sum( len(order.jobs) for order in order_instance.orders[1:])
    nb_accepted_new_jobs = len(final_state.job_states) - initial_jobs
    acceptance_ratio = (nb_accepted_new_jobs / nb_new_jobs if nb_new_jobs != 0 else 0)
    accepted_over_new = f"{nb_accepted_new_jobs}/{nb_new_jobs}"
    analysis_csv_path = os.path.join( analysis_dir, "summary.csv")

    with open(analysis_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "inst",
                "variant",
                "delta_ratio",
                "nb_orders",
                "nb_jobs",
                "nb_new_jobs",
                "nb_accepted_new_jobs",
                "accepted_over_new",
                "acceptance_ratio",
                "cmax",
                "total_tardiness",
                "weighted_tardiness",
                "obj",
                "computing_time"
            ]
        )

        writer.writeheader()

        writer.writerow({
            "scenario": scenario,
            "inst": inst,
            "variant": variant,
            "delta_ratio": delta_ratio,
            "nb_orders": len(order_instance.orders),
            "nb_jobs": order_instance.nb_jobs,
            "nb_new_jobs": nb_new_jobs,
            "nb_accepted_new_jobs": nb_accepted_new_jobs,
            "accepted_over_new": accepted_over_new,
            "acceptance_ratio": round(acceptance_ratio, 2),
            "cmax": cmax,
            "total_tardiness": total_tardiness,
            "weighted_tardiness": weighted_tardiness,
            "obj": obj,
            "computing_time": computing_time
        })

    cut_times = [
        order.cut_time
        for order in order_instance.orders
        if order.cut_time > 0
    ]

    if generate_gantts:
        gantt_path = os.path.join(
            gantt_dir,
            f"{variant}_final_gantt.png"
        )
        gnn_gantt(
            gantt_path,
            final_state,
            f"{scenario}/{inst}/{variant}/delta={delta_ratio}",
            cut_times=cut_times
        )
    else:
        gantt_path = ""

    csv_path = os.path.join(
        output_dir,
        "results.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "inst",
                "variant",
                "input_path",
                "delta_ratio",
                "cut_times",
                "nb_orders",
                "nb_jobs",
                "cmax",
                "total_tardiness",
                "weighted_tardiness",
                "obj",
                "computing_time"
            ]
        )

        writer.writeheader()

        writer.writerow({
            "scenario": scenario,
            "inst": inst,
            "variant": variant,
            "input_path": input_path,
            "delta_ratio": delta_ratio,
            "cut_times": cut_times,
            "nb_orders": len(order_instance.orders),
            "nb_jobs": order_instance.nb_jobs,
            "cmax": cmax,
            "total_tardiness": total_tardiness,
            "weighted_tardiness": weighted_tardiness,
            "obj": obj,
            "computing_time": computing_time
        })

    print("\n=== Résultat sauvegardé ===")
    print(f"CSV      : {csv_path}")
    if generate_gantts:
        print(f"Gantt    : {gantt_path}")
    else:
        print("Gantt    : désactivé")
    print(f"Analysis CSV : {analysis_csv_path}")
    print(f"Cmax     : {cmax}")
    print(f"Delay    : {total_tardiness}")
    print(f"WDelay   : {weighted_tardiness}")
    print(f"Obj      : {obj}")


def run_all_controlled_instances(
    root_dir: str,
    output_root: str,
    delta_values: list[float],
    device: str,
    agent_path: str,
    scenario_filter: str = None,
    inst_filter: str = None,
    save_step_gantts: bool = False,
    generate_gantts: bool = True
):
    """
    Lance la méthode d'acceptation sur toutes les instances contenues dans root_dir.

    Structure attendue :
    root_dir/
        scenario/
            inst/
                variant.json
    """

    root_path = Path(root_dir)
    json_files = sorted(root_path.glob("*/*/*.json"))

    if not json_files:
        json_files = sorted(root_path.glob("*/*.json"))

    if not json_files:
        print(f"Aucune instance trouvée dans : {root_dir}")
        return

    print("\n" + "=" * 80)
    print(f"Lancement global sur {len(json_files)} instances")
    print(f"Root          : {root_dir}")
    print(f"Deltas        : {delta_values}")
    print(f"Generate Gantt: {generate_gantts}")
    print("=" * 80)

    for delta_ratio in delta_values:
        print("\n" + "=" * 80)
        print(f"LANCEMENT POUR DELTA = {delta_ratio}")
        print("=" * 80)

        for input_file in json_files:
            scenario = input_file.parent.parent.name
            inst = input_file.parent.name
            variant = input_file.stem

            parts = input_file.relative_to(root_path).parts
            if len(parts) == 3:
                scenario = parts[0]
                inst = parts[1]
                variant = input_file.stem
            elif len(parts) == 2:
                scenario = root_path.name
                inst = parts[0]
                variant = input_file.stem
            else:
                continue

            if scenario_filter is not None and scenario != scenario_filter:
                continue

            if inst_filter is not None and inst != inst_filter:
                continue

            print("\n" + "#" * 80)
            print(f"Scenario : {scenario}")
            print(f"Inst     : {inst}")
            print(f"Variant  : {variant}")
            print(f"Delta    : {delta_ratio}")
            print(f"Input    : {input_file}")
            print("#" * 80)

            try:
                run_one_controlled_instance(
                    input_path=str(input_file),
                    output_root=output_root,
                    scenario=scenario,
                    inst=inst,
                    delta_ratio=delta_ratio,
                    device=device,
                    agent_path=agent_path,
                    save_step_gantts=save_step_gantts,
                    generate_gantts=generate_gantts
                )

            except Exception as e:
                print(f"❌ Erreur sur {input_file} avec delta={delta_ratio}")
                print(f"   {type(e).__name__}: {e}")
                continue

    print("\n" + "=" * 80)
    print("Fin du lancement global")
    print("=" * 80)


# RUN ONE Instance avec Gantt final/subsets, mais sans Gantt étape par étape:

# inst1
# early_1_to_cmax_over_3
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs/inst1/early_1_to_cmax_over_3.json" --scenario "same_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/different_costs/inst1/early_1_to_cmax_over_3.json" --scenario "different_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs_dd_serre_N/inst1/early_1_to_cmax_over_3.json" --scenario "same_costs_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N/inst1/early_1_to_cmax_over_3.json" --scenario "E_90_100_N_40_50_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/inst1/early_1_to_cmax_over_3.json" --scenario "E_90_100_N_1_10_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true

# middle_cmax_over_3_to_cmax_over_2
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs/inst1/middle_cmax_over_3_to_cmax_over_2.json" --scenario "same_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/different_costs/inst1/middle_cmax_over_3_to_cmax_over_2.json" --scenario "different_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs_dd_serre_N/inst1/middle_cmax_over_3_to_cmax_over_2.json" --scenario "same_costs_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N/inst1/middle_cmax_over_3_to_cmax_over_2.json" --scenario "E_90_100_N_40_50_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/inst1/middle_cmax_over_3_to_cmax_over_2.json" --scenario "E_90_100_N_1_10_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true

# late_cmax_over_2_to_before_cmax
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs/inst1/late_cmax_over_2_to_before_cmax.json" --scenario "same_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/different_costs/inst1/late_cmax_over_2_to_before_cmax.json" --scenario "different_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs_dd_serre_N/inst1/late_cmax_over_2_to_before_cmax.json" --scenario "same_costs_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N/inst1/late_cmax_over_2_to_before_cmax.json" --scenario "E_90_100_N_40_50_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/inst1/late_cmax_over_2_to_before_cmax.json" --scenario "E_90_100_N_1_10_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts true



#inst2
# early_1_to_cmax_over_3
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs/inst2/early_1_to_cmax_over_3.json" --scenario "same_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/different_costs/inst2/early_1_to_cmax_over_3.json" --scenario "different_costs" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs_dd_serre_N/inst2/early_1_to_cmax_over_3.json" --scenario "same_costs_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N/inst2/early_1_to_cmax_over_3.json" --scenario "E_90_100_N_40_50_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/inst2/early_1_to_cmax_over_3.json" --scenario "E_90_100_N_1_10_dd_serre_N" --inst "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# middle_cmax_over_3_to_cmax_over_2
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs/inst2/middle_cmax_over_3_to_cmax_over_2.json" --scenario "same_costs" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/different_costs/inst2/middle_cmax_over_3_to_cmax_over_2.json" --scenario "different_costs" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs_dd_serre_N/inst2/middle_cmax_over_3_to_cmax_over_2.json" --scenario "same_costs_dd_serre_N" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N/inst2/middle_cmax_over_3_to_cmax_over_2.json" --scenario "E_90_100_N_40_50_dd_serre_N" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/inst2/middle_cmax_over_3_to_cmax_over_2.json" --scenario "E_90_100_N_1_10_dd_serre_N" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# late_cmax_over_2_to_before_cmax
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs/inst2/late_cmax_over_2_to_before_cmax.json" --scenario "same_costs" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/different_costs/inst2/late_cmax_over_2_to_before_cmax.json" --scenario "different_costs" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/same_costs_dd_serre_N/inst2/late_cmax_over_2_to_before_cmax.json" --scenario "same_costs_dd_serre_N" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N/inst2/late_cmax_over_2_to_before_cmax.json" --scenario "E_90_100_N_40_50_dd_serre_N" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode one --input "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/inst2/late_cmax_over_2_to_before_cmax.json" --scenario "E_90_100_N_1_10_dd_serre_N" --inst "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false


# RUN ALL Instances avec un delta, sans Gantt étape par étape:
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# RUN ALL Instances avec plusieurs deltas, sans Gantt étape par étape:
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --deltas 0.0 0.1 0.2 0.3 --device mps --agent_path "data/training_costs/" --generate_gantts false

# ============================================================
# RUN PAR TYPE / SCÉNARIO
# ============================================================

# RUN seulement same_costs avec un delta :
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "same_costs" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# RUN seulement same_costs avec plusieurs deltas :
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "same_costs" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "same_costs_25" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "same_costs_25" --deltas 0.7 1.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "same_costs_50" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "same_costs_100" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false

# RUN seulement different_costs avec un delta :
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# RUN seulement different_costs avec plusieurs deltas :
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs_100_50" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs_100_25" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs_50_25" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs_25_50" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs_25_100" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --scenario_filter "different_costs_50_100" --deltas 0.2 0.5 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false

# RUN tous les types / scénarios avec un delta :
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# RUN tous les types / scénarios avec plusieurs deltas :
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test" --deltas 0.0 0.1 0.2 0.3 --device mps --agent_path "data/training_costs/" --generate_gantts false

# ================
# RUN tout Inst1 et tout Inst2 :
# Inst1
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/same_costs" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/same_costs_dd_serre_N" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_50_25" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_50_25" --inst_filter "inst1" --delta_ratio 0.5 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_50_25" --inst_filter "inst1" --delta_ratio 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_100_25" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_100_25" --inst_filter "inst1" --delta_ratio 0.5 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_100_25" --inst_filter "inst1" --delta_ratio 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_100_50" --inst_filter "inst1" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_100_50" --inst_filter "inst1" --delta_ratio 0.5 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs_100_50" --inst_filter "inst1" --delta_ratio 0.8 --device mps --agent_path "data/training_costs/" --generate_gantts false


# Inst2
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/same_costs" --inst_filter "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/different_costs" --inst_filter "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/same_costs_dd_serre_N" --inst_filter "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/E_90_100_N_40_50_dd_serre_N" --inst_filter "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false
# python acceptation_solver.py.py --mode all --root_dir "data/controlled_orders/test/E_90_100_N_1_10_dd_serre_N/" --inst_filter "inst2" --delta_ratio 0.2 --device mps --agent_path "data/training_costs/" --generate_gantts false

# ========================================
# RUN: by insatnces with one value of delta
# python acceptation_solver.py --mode all --root_dir "data/controlled_orders_ub/test/s" --output_root "results/controlled_orders/test" --deltas 0.2 --device mps --agent_path "data/training_costs_ub/" --generate_gantts true


# RUN: by insatnces with multiple value of delta
# python acceptation_solver.py --mode all --root_dir "data/controlled_orders_ub/test/s" --output_root "results/controlled_orders/test" --deltas 0.1 0.5 --device mps --agent_path "data/training_costs_ub/" --generate_gantts true
# python acceptation_solver.py --mode all --root_dir "data/controlled_orders_ub/test/m" --output_root "results/controlled_orders/test" --deltas 0.1 0.2 --device mps --agent_path "data/training_costs_ub/" --generate_gantts true



# s
# python3 acceptation_solver.py --mode all --root_dir "data/controlled_orders_ub/test/s" --output_root "results/controlled_orders/test" --deltas 0.1 0.2 0.5 --device cpu --agent_path "data/training_costs_ub/" --generate_gantts false

# m
# python3 acceptation_solver.py --mode all --root_dir "data/controlled_orders_ub/test/m" --output_root "results/controlled_orders/test" --deltas 0.1 0.2 0.5 --device cpu --agent_path "data/training_costs_ub/" --generate_gantts false

# l
# python3 acceptation_solver.py --mode all --root_dir "data/controlled_orders_ub/test/l" --output_root "results/controlled_orders/test" --deltas 0.1 0.2 0.5 --device cpu --agent_path "data/training_costs_ub/" --generate_gantts false

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["one", "all"], default="one", help="Mode d'exécution : one pour une seule instance, all pour toutes les instances")
    parser.add_argument("--input", type=str, default=None, help="Chemin vers une instance JSON unique")
    parser.add_argument("--root_dir", type=str, default="data/controlled_orders/test", help="Dossier racine contenant toutes les familles d'instances")
    parser.add_argument("--output_root", type=str, default="results/controlled_orders/test", help="Dossier racine des résultats")
    parser.add_argument("--scenario", type=str, default=None, help="Nom du scénario (mode one), ex: same_costs")
    parser.add_argument("--scenario_filter", type=str, default=None, help="Nom du scénario (mode all), ex: same_costs")
    parser.add_argument("--inst_filter", type=str, default=None, help="Filtrer une instance spécifique, ex: inst1, inst2")
    parser.add_argument("--inst", type=str, default=None, help="Nom de l'instance, ex: inst1")
    parser.add_argument("--delta_ratio", type=float, default=0.2, help="Ratio delta pour la méthode d'acceptation")
    parser.add_argument("--deltas", type=float, nargs="+", default=None, help="Liste de deltas, ex: --deltas 0.0 0.1 0.2 0.3")
    parser.add_argument("--device", type=str, default="cpu", help="Device utilisé : cpu, cuda ou mps")
    parser.add_argument("--agent_path", type=str, default="data/training_costs/", help="Chemin vers les poids de l'agent")
    parser.add_argument("--save_step_gantts", action="store_true", help="Sauvegarder les Gantt étape par étape")
    parser.add_argument("--generate_gantts",type=str, choices=["true", "false"], default="true", help="Activer ou désactiver la génération des Gantt")
    args = parser.parse_args()

    delta_values = args.deltas if args.deltas is not None else [args.delta_ratio]
    generate_gantts = args.generate_gantts.lower() == "true"

    if args.mode == "all":
        run_all_controlled_instances(
            root_dir=args.root_dir,
            output_root=args.output_root,
            delta_values=delta_values,
            device=args.device,
            agent_path=args.agent_path,
            scenario_filter=args.scenario_filter,
            inst_filter=args.inst_filter,
            save_step_gantts=args.save_step_gantts,
            generate_gantts=generate_gantts
        ) 
    else:
        if args.input is None: raise ValueError("En mode one, tu dois fournir --input")
        if args.scenario is None: raise ValueError("En mode one, tu dois fournir --scenario")
        if args.inst is None: raise ValueError("En mode one, tu dois fournir --inst")
        run_one_controlled_instance(
            input_path=args.input,
            output_root=args.output_root,
            scenario=args.scenario,
            inst=args.inst,
            delta_ratio=args.delta_ratio,
            device=args.device,
            agent_path=args.agent_path,
            save_step_gantts=args.save_step_gantts,
            generate_gantts=generate_gantts
        )