import os
import csv
import time
import argparse

from models.order import OrderInstance
from models.agent import Agent
from acceptation_method import acceptation_method
from gantt_builder.gnn_gantt import gnn_gantt



def compute_weighted_tardiness(final_state):
    return sum(
        int(getattr(j.job, "cost", 1)) * j.delay
        for j in final_state.job_states
    )


def run_one_controlled_instance(
    input_path: str,
    output_root: str,
    scenario_name: str,
    inst_name: str,
    delta_ratio: float,
    device: str,
    agent_path: str
):
    os.makedirs(
        os.path.join(output_root, scenario_name, inst_name),
        exist_ok=True
    )

    output_dir = os.path.join(output_root, scenario_name, inst_name)

    instance_name = os.path.splitext(os.path.basename(input_path))[0]

    print("\n==================================================")
    print(f"Instance testée : {input_path}")
    print(f"Scenario        : {scenario_name}")
    print(f"Inst            : {inst_name}")
    print(f"Variant         : {instance_name}")
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

    step_gantt_dir = os.path.join(
    output_dir,
    instance_name,
    "step_gantts"
    )

    gantt_dir = os.path.join(
    output_dir,
    instance_name,
    "gantts"
    )
    
    final_state = acceptation_method(
        order_instance=order_instance,
        agent=agent,
        device=device,
        delta_ratio=delta_ratio,
        gantt_dir=gantt_dir,
    )

    computing_time = time.perf_counter() - start

    total_tardiness = sum(j.delay for j in final_state.job_states)
    weighted_tardiness = compute_weighted_tardiness(final_state)
    cmax = final_state.cmax
    obj = cmax + weighted_tardiness

    cut_times = [
        order.cut_time
        for order in order_instance.orders
        if order.cut_time > 0
    ]

    gantt_path = os.path.join(
        output_dir,
        f"{instance_name}_gantt.png"
    )

    gnn_gantt(
        gantt_path,
        final_state,
        f"{scenario_name}/{inst_name}/{instance_name}",
        cut_times=cut_times
    )

    csv_path = os.path.join(
        output_dir,
        "results.csv"
    )

    file_exists = os.path.exists(csv_path)

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

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "scenario": scenario_name,
            "inst": inst_name,
            "variant": instance_name,
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
    print(f"CSV   : {csv_path}")
    print(f"Gantt : {gantt_path}")
    print(f"Cmax  : {cmax}")
    print(f"Delay : {total_tardiness}")
    print(f"WDelay: {weighted_tardiness}")
    print(f"Obj   : {obj}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Chemin du fichier JSON à tester"
    )

    parser.add_argument(
        "--scenario",
        required=True,
        help="Nom du dossier scénario, ex: different_costs"
    )

    parser.add_argument(
        "--inst",
        required=True,
        help="Nom de l'instance, ex: inst1 ou inst2"
    )

    parser.add_argument(
        "--output_root",
        default="results/controlled_orders/test",
        help="Dossier racine des résultats"
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help="delta_ratio de la méthode d'acceptation"
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--agent_path",
        default="data/training_costs/"
    )

    args = parser.parse_args()

    run_one_controlled_instance(
        input_path=args.input,
        output_root=args.output_root,
        scenario_name=args.scenario,
        inst_name=args.inst,
        delta_ratio=args.delta,
        device=args.device,
        agent_path=args.agent_path
    )

    """ python run_controlled_acceptation.py \
        --input "data/controlled_orders/test/same_costs/inst1/early_1_to_cmax_over_3.json" \
        --scenario "same_costs" \
        --inst "inst1" \
        --delta 0.2 \
        --device cpu \
        --agent_path "data/training_costs/"
  """