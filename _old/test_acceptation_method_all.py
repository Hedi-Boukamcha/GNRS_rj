import os
import csv
import time
import traceback

from models.order import OrderInstance
from models.agent import Agent
from master_problem.acceptation_method import acceptation_method
from conf import *


def find_json_instances(folder_path: str) -> list[str]:
    instances = []

    if not os.path.exists(folder_path):
        return instances

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            instances.append(os.path.join(folder_path, file_name))

    instances.sort()
    return instances


def run_acceptation_on_instance(instance_path: str, size_name: str, agent: Agent, device: str, delta_ratio: float) -> dict:
    instance_name = os.path.basename(instance_path)

    print("\n" + "=" * 80)
    print(f"Instance : {instance_name}")
    print(f"Taille   : {size_name}")
    print("=" * 80)

    result = {
        "size": size_name,
        "instance": instance_name,
        "path": instance_path,
        "status": "OK",
        "nb_orders": 0,
        "nb_jobs": 0,
        "cmax": None,
        "total_delay": None,
        "computation_time_sec": None,
        "error": ""
    }

    try:
        order_instance = OrderInstance.load(instance_path)

        result["nb_orders"] = len(order_instance.orders)
        result["nb_jobs"] = order_instance.nb_jobs

        start_time = time.perf_counter()

        final_state = acceptation_method(
            order_instance=order_instance,
            agent=agent,
            device=device,
            delta_ratio=delta_ratio,
        )

        end_time = time.perf_counter()
        computation_time = end_time - start_time

        result["cmax"] = final_state.cmax
        result["total_delay"] = sum(j.delay for j in final_state.job_states)
        result["computation_time_sec"] = computation_time

        print(f"Temps total : {computation_time:.4f} secondes")
        print(f"Cmax        : {result['cmax']}")
        print(f"Delay       : {result['total_delay']}")

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["computation_time_sec"] = None

        print(f"Erreur sur {instance_name}: {e}")
        traceback.print_exc()

    return result


def compute_summary(results: list[dict]) -> list[dict]:
    summary = []

    sizes = sorted(set(r["size"] for r in results))

    for size in sizes:
        size_results = [
            r for r in results
            if r["size"] == size and r["status"] == "OK" and r["computation_time_sec"] is not None
        ]

        all_size_results = [r for r in results if r["size"] == size]

        if len(size_results) == 0:
            summary.append({
                "size": size,
                "nb_instances_total": len(all_size_results),
                "nb_instances_ok": 0,
                "nb_instances_error": len(all_size_results),
                "total_time_sec": 0.0,
                "average_time_sec": None,
                "min_time_sec": None,
                "max_time_sec": None,
                "average_nb_jobs": None,
                "average_nb_orders": None
            })
            continue

        times = [r["computation_time_sec"] for r in size_results]
        nb_jobs = [r["nb_jobs"] for r in size_results]
        nb_orders = [r["nb_orders"] for r in size_results]

        summary.append({
            "size": size,
            "nb_instances_total": len(all_size_results),
            "nb_instances_ok": len(size_results),
            "nb_instances_error": len(all_size_results) - len(size_results),
            "total_time_sec": round(sum(times), 2),
            "average_time_sec": round(sum(times) / len(times), 2),
            "min_time_sec": round(min(times), 2),
            "max_time_sec": round(max(times), 2),
            "average_nb_jobs": round(sum(nb_jobs) / len(nb_jobs), 2),
            "average_nb_orders": round(sum(nb_orders) / len(nb_orders), 2)
        })

    return summary


def save_csv(path: str, rows: list[dict], fieldnames: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    device = "cpu"
    delta_ratio = 0.0

    test_root = "data/orders_instances/test"

    size_folders = {
        "s": os.path.join(test_root, "s"),
        "m": os.path.join(test_root, "m"),
        "l": os.path.join(test_root, "l"),
        "xl": os.path.join(test_root, "xl"),
    }

    results_dir = "data/acceptation_results"
    details_csv_path = os.path.join(results_dir, "acceptation_results_details.csv")
    summary_csv_path = os.path.join(results_dir, "acceptation_results_summary.csv")

    agent = Agent(
        device=device,
        interactive=False,
        load=True,
        path="data/training/",
        train=False,
        custom=True
    )

    all_results = []

    for size_name, folder_path in size_folders.items():
        instance_paths = find_json_instances(folder_path)

        print("\n" + "#" * 80)
        print(f"Taille {size_name.upper()} | Nombre d'instances trouvées : {len(instance_paths)}")
        print("#" * 80)

        for instance_path in instance_paths:
            result = run_acceptation_on_instance(
                instance_path=instance_path,
                size_name=size_name,
                agent=agent,
                device=device,
                delta_ratio=delta_ratio
            )
            all_results.append(result)

    details_fields = [
        "size",
        "instance",
        "path",
        "status",
        "nb_orders",
        "nb_jobs",
        "cmax",
        "total_delay",
        "computation_time_sec",
        "error"
    ]

    save_csv(
        path=details_csv_path,
        rows=all_results,
        fieldnames=details_fields
    )

    summary = compute_summary(all_results)

    summary_fields = [
        "size",
        "nb_instances_total",
        "nb_instances_ok",
        "nb_instances_error",
        "total_time_sec",
        "average_time_sec",
        "min_time_sec",
        "max_time_sec",
        "average_nb_jobs",
        "average_nb_orders"
    ]

    save_csv(
        path=summary_csv_path,
        rows=summary,
        fieldnames=summary_fields
    )

    print("\n" + "=" * 80)
    print("Résumé final")
    print("=" * 80)

    for row in summary:
        print(
            f"Taille {row['size'].upper()} | "
            f"OK={row['nb_instances_ok']}/{row['nb_instances_total']} | "
            f"Temps total={row['total_time_sec']:.2f}s | "
            f"Temps moyen={row['average_time_sec']:.2f}s"
            if row["average_time_sec"] is not None
            else
            f"Taille {row['size'].upper()} | aucune instance réussie"
        )

    print(f"\nFichier détails : {details_csv_path}")
    print(f"Fichier résumé  : {summary_csv_path}")


if __name__ == "__main__":
    main()