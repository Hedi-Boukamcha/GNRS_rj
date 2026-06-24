import json
import csv
import os


def export_jobs_to_csv(json_path: str, csv_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for order in data["orders"]:
        order_id = order["id"]

        for job_index, job in enumerate(order["jobs"], start=1):
            rows.append({
                "order_id": order_id,
                "job": f"J{job_index}",
                "due_date": job["due_date"],
                "cost": job.get("cost", 1)
            })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["order_id", "job", "due_date", "cost"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV sauvegardé dans : {csv_path}")


if __name__ == "__main__":
    json_path = "data/instances_cost/E_high_N_low_ddLow/instance_3.json"
    csv_path = "data/instances_cost/E_high_N_low_ddLow/instance_3.csv"

    export_jobs_to_csv(json_path, csv_path)