import json
import csv
import os
from pathlib import Path


def operations_to_string(operations):
    ops = []

    for op in operations:
        machine = "M1" if op["type"] == 0 else "M2"
        ops.append(f"{machine}:{op['processing_time']}")

    return " | ".join(ops)


def export_jobs_to_csv(json_path: str, csv_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    scenario = data.get("scenario", "")
    arrival_variant = data.get("arrival_variant", "")
    arrival_cut_time = data.get("arrival_cut_time", "")
    estimated_cmax = data.get("estimated_cmax_order_1", "")
    a = data.get("a", "")

    for order in data["orders"]:
        order_id = order["id"]
        order_cut_time = order["cut_time"]

        for job_index, job in enumerate(order["jobs"], start=1):
            operations = job.get("operations", [])

            rows.append({
                "estimated_cmax": estimated_cmax,
                "order_id": order_id,
                "order_cut_time": order_cut_time,
                "job": f"J{job_index}",
                "rj": job.get("release_date", ""),
                "dj": job.get("due_date", ""),
                "cost": job.get("cost", 1),
                "nb_operations": len(operations)
            })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "estimated_cmax",
        "order_id",
        "order_cut_time",
        "job",
        "rj",
        "dj",
        "cost",
        "nb_operations"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV sauvegardé : {csv_path}")


def convert_all_json_to_individual_csv(root_dir: str):
    root_path = Path(root_dir)

    json_files = sorted(root_path.rglob("*.json"))

    if not json_files:
        print(f"Aucun fichier JSON trouvé dans : {root_dir}")
        return

    for json_file in json_files:
        csv_file = json_file.with_suffix(".csv")
        export_jobs_to_csv(str(json_file), str(csv_file))

    print(f"\nNombre de fichiers convertis : {len(json_files)}")


if __name__ == "__main__":
    root_dir = "data/controlled_orders/test"

    convert_all_json_to_individual_csv(root_dir)