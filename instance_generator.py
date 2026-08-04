# instance_generator.py
import os
import random
import copy
import argparse
import json

from conf import *

# ##########################
# =*= INSTANCE GENERATOR =*=
# ##########################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0"
__license__ = "MIT"

def convert_relative_jobs_to_absolute(
    relative_jobs: list[dict],
    base_time: int
) -> list[dict]:
    """
    Convertit :
    relative_release → release_date
    relative_due     → due_date
    """

    absolute_jobs = []

    for job in relative_jobs:
        new_job = copy.deepcopy(job)

        new_job["release_date"] = base_time + new_job["relative_release"]
        new_job["due_date"] = base_time + new_job["relative_due"]

        del new_job["relative_release"]
        del new_job["relative_due"]

        absolute_jobs.append(new_job)

    return absolute_jobs

def save_json(instance: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(instance, f, indent=4)

    print(f"Saved: {path}")

def randint_interval(low: int, high: int) -> int:
    low = int(low)
    high = int(high)

    if low < 1:
        low = 1

    if high < low:
        high = low

    return random.randint(low, high)

def generate_relative_job(
    max_operations_per_job: int  = 2,
    types_operations: list[int] = [0, 1],
    min_duration: int   = 10,
    max_duration: int   = 60,
    pos_time: int       = 5,
    release_spread: int = 50,
    due_slack_min: int  = 10,
    due_slack_max: int  = 80,
    cost_min: float     = 0.0,
    cost_max: float     = 1.0
) -> dict:
    """
    Génère un job avec dates relatives.
    Les dates absolues seront créées ensuite selon le cut_time.
    """

    nb_operations = random.randint(1, max_operations_per_job)
    operations = []
    last_type = None

    for _ in range(nb_operations):
        available_types = [t for t in types_operations if t != last_type]
        chosen_type = random.choice(available_types)
        proc_time = random.randint(min_duration, max_duration)

        operations.append({
            "type": chosen_type,
            "processing_time": proc_time
        })

        last_type = chosen_type

    relative_release = random.randint(0, release_spread)

    proc_total = sum(op["processing_time"] for op in operations)

    # Temps minimal approximatif :
    # chargement + déchargement = 2L
    # pos_time
    # mouvements = 2M par opération
    min_treat_time = proc_total + (2 * L + pos_time) + nb_operations * (2 * M)

    relative_due = relative_release + min_treat_time + random.randint(
        due_slack_min,
        due_slack_max
    )

    return {
        "big": random.randint(0, 1),
        "relative_due": relative_due,
        "pos_time": pos_time,
        "status": 0,
        "blocked": 0,
        "cost": random.randint(cost_min, cost_max),
        "operations": operations,
        "relative_release": relative_release
    }

def generate_order_template(
    nb_jobs: int,
    max_operations_per_job: int = 2,
    min_duration: int = 10,
    max_duration: int = 60,
    pos_time: int = 5,
    release_spread: int = 50,
    due_slack_min: int = 10,
    due_slack_max: int = 80,
    cost_min: int = 1,
    cost_max: int = 10
) -> list[dict]:
    jobs = []

    for _ in range(nb_jobs):
        job = generate_relative_job(
            max_operations_per_job=max_operations_per_job,
            min_duration=min_duration,
            max_duration=max_duration,
            pos_time=pos_time,
            release_spread=release_spread,
            due_slack_min=due_slack_min,
            due_slack_max=due_slack_max,
            cost_min=cost_min,
            cost_max=cost_max
        )

        jobs.append(job)

    return jobs

SCENARIOS_UB = {
    "same_costs": {
        "e_ratio": 0.5,
        "existing_cost_range": (1.0, 1.0),
        "new_cost_range": (1.0, 1.0),
    },
    "portion_of_3_7": {
        "e_ratio": 0.3,
        "existing_cost_range": (0.8, 1.0),
        "new_cost_range": (0.05, 0.4),
    },
    "portion_of_7_3": {
        "e_ratio": 0.7,
        "existing_cost_range": (0.8, 1.0),
        "new_cost_range": (0.05, 0.4),
    },
}

CUT_TIME_TIERS = {
    "early": lambda ub: randint_interval(0, int(ub / 3)),
    "middle": lambda ub: randint_interval(int(ub / 3), int(2 * ub / 3)),
    "late": lambda ub: randint_interval(int(2 * ub / 3), int(3 * ub / 3)),
}

def compute_execution_load(jobs: list[dict], worst_case: bool = False) -> int:
    load = 0
    for job in jobs:
        for op in job["operations"]:
            load += 2 * M + op["processing_time"]
            if worst_case and op["type"] == MACHINE_1:
                load += job["pos_time"] + M
        load += 2 * L
    return load

def estimated_ub_cmax(jobs: list[dict], start_time: int) -> int:
    return max(start_time, 1) + compute_execution_load(jobs)

def compute_ub_cmax_e(existing_jobs: list[dict], worst_case: bool = False) -> int:
    """UBe = max(rj_e) + Execute(E)"""
    sorted_jobs = sorted(existing_jobs, key=lambda j: j["relative_release"], reverse=worst_case)
    current_time = 0
    for job in sorted_jobs:
        current_time = max(current_time, job["relative_release"])
        current_time += compute_execution_load([job], worst_case=worst_case)
    return current_time

def compute_ub_cmax_e_n(new_jobs: list[dict], cut_time: int, ub_cmax_e: int, worst_case: bool = False) -> int:
    """REST = UBe - cutTime si UBe > cutTime, sinon cutTime.
    UBe+n = max(rj_n, REST) + Execute(N)"""
    sorted_jobs = sorted(new_jobs, key=lambda j: j["relative_release"], reverse=worst_case)
    current_time = max(cut_time, ub_cmax_e)
    for job in sorted_jobs:
        absolute_release = cut_time + job["relative_release"]
        current_time = max(current_time, absolute_release)
        current_time += compute_execution_load([job], worst_case=worst_case)
    return current_time

def sample_due_dates_from_ub(jobs: list[dict], min_due_date: float, max_due_date: float, base_time: int, min_slack: float) -> list[dict]:
    jobs = copy.deepcopy(jobs)
    for job in jobs:
        absolute_release = base_time + job["relative_release"]
        # plus petite date a laquelle le job peut finir s'il est traite seul :
        # release + temps de traitement + mouvements necessaires (2*M par op, pos_time+M si MACHINE_1, 2*L)
        job_min_due = estimated_ub_cmax([job], absolute_release)
        job_min_due = max(min_due_date, job_min_due)
        # garde-fou: garantir au moins min_slack de marge, meme quand max_due_date (le budget de la
        # tier) tombe sous job_min_due -- sinon job_max_due retombait exactement sur job_min_due (marge
        # nulle), rendant le job en retard des qu'il y a la moindre contention avec d'autres jobs
        # (observe empiriquement: tier "late" -> 94.5% des jobs "new" avec marge nulle/negative)
        job_max_due = max(job_min_due + min_slack, max_due_date)
        due_absolute = random.uniform(job_min_due, job_max_due)
        job["relative_due"] = int(round(due_absolute - base_time))
    return jobs


def apply_random_costs(jobs: list[dict], cost_min: float, cost_max: float) -> list[dict]:
    jobs = copy.deepcopy(jobs)
    for job in jobs:
        job["cost"] = round(random.uniform(cost_min, cost_max), 4)
    return jobs


def build_instance(
    existing_jobs: list[dict],
    new_jobs: list[dict],
    cut_time_new_order: int,
    a: int,
    scenario_name: str,
    ub_cmax_e: int,
    ub_cmax_e_n: int,
    variant_name: str
) -> dict:
    order_1_jobs = convert_relative_jobs_to_absolute(relative_jobs=existing_jobs, base_time=0)
    order_2_jobs = convert_relative_jobs_to_absolute(relative_jobs=new_jobs, base_time=cut_time_new_order)

    return {
        "a": a,
        "scenario": scenario_name,
        "ub_cmax_order_1": ub_cmax_e,           # UB(cmaxE)   -> a servi a placer cut_time et la due_date max de E
        "ub_cmax_order_1_2": ub_cmax_e_n,  # UB(cmaxE+N) -> a servi a la due_date max de N
        "arrival_variant": variant_name,
        "arrival_cut_time": cut_time_new_order,
        "orders": [
            {
                "id": 1,
                "cut_time": 0,
                "jobs": order_1_jobs
            },
            {
                "id": 2,
                "cut_time": cut_time_new_order,
                "jobs": order_2_jobs
            }
        ]
    }


def generate_one_ub_scenario_instance(
    total_jobs: int,
    e_ratio: float,
    existing_cost_range: tuple[float, float],
    new_cost_range: tuple[float, float],
    scenario_name: str,
    max_operations_per_job: int = 2,
    min_duration: int = 10,
    max_duration: int = 60,
    pos_time: int = 5,
    release_spread_existing: int = 50,
    release_spread_new: int = 50,
    due_date_min: int = 50,
) -> dict[str, dict]:
    nb_e = max(1, min(total_jobs - 1, round(total_jobs * e_ratio)))
    nb_n = total_jobs - nb_e

    existing_jobs = generate_order_template(
        nb_jobs=nb_e,
        max_operations_per_job=max_operations_per_job,
        min_duration=min_duration,
        max_duration=max_duration,
        pos_time=pos_time,
        release_spread=release_spread_existing,
    )
    new_jobs = generate_order_template(
        nb_jobs=nb_n,
        max_operations_per_job=max_operations_per_job,
        min_duration=min_duration,
        max_duration=max_duration,
        pos_time=pos_time,
        release_spread=release_spread_new, 
    )
    ub_cmax_e = compute_ub_cmax_e(existing_jobs)

    existing_jobs = apply_random_costs(
        existing_jobs,
        existing_cost_range[0],
        existing_cost_range[1]
    )

    new_jobs = apply_random_costs(
        new_jobs,
        new_cost_range[0],
        new_cost_range[1]
    )

    existing_jobs = sample_due_dates_from_ub(
        jobs=existing_jobs,
        min_due_date=due_date_min,
        max_due_date=ub_cmax_e / 3,
        base_time=0,
        min_slack=due_date_min
    )

    a = random.randint(0, 10) * 10

    used_cut_times = set()
    instances_by_tier: dict[str, dict] = {}

    for tier_name, cut_rule in CUT_TIME_TIERS.items():
        cut_time = cut_rule(ub_cmax_e)
        while cut_time in used_cut_times:
            cut_time = cut_rule(ub_cmax_e)
        used_cut_times.add(cut_time)
        ub_cmax_e_n = compute_ub_cmax_e_n(
            new_jobs,
            cut_time=cut_time,
            ub_cmax_e=ub_cmax_e
        )

        new_jobs_with_due_dates = sample_due_dates_from_ub(
            jobs=new_jobs,
            min_due_date=cut_time + due_date_min,
            max_due_date=ub_cmax_e_n / 3,
            base_time=cut_time,
            min_slack=due_date_min
        )

        instance = build_instance(
            existing_jobs=existing_jobs,
            new_jobs=new_jobs_with_due_dates,
            cut_time_new_order=cut_time,
            a=a,
            scenario_name=scenario_name,
            ub_cmax_e=ub_cmax_e,
            ub_cmax_e_n=ub_cmax_e_n,
            variant_name=tier_name
        )
        instance["total_jobs"] = total_jobs
        instance["nb_existing_jobs"] = nb_e
        instance["nb_new_jobs"] = nb_n

        instances_by_tier[tier_name] = instance

    return instances_by_tier


def generate_controlled_orders_ub_sized(
    output_root: str,
    nb_train: int,
    nb_test: int,
    max_operations_per_job: int = 2,
    min_duration: int = 10,
    max_duration: int = 60,
    pos_time: int = 5,
    release_spread_existing: int = 50,
    release_spread_new: int = 50,
    due_date_min: int = 50,
    seed: int = 1
):
    random.seed(seed)

    for size_name, job_min, job_max in INSTANCES_SIZES:
        for scenario_name, scenario in SCENARIOS_UB.items():
            for split_name, nb_instances in (("train", nb_train), ("test", nb_test)):
                folder = os.path.join(output_root, split_name, size_name, scenario_name)
                os.makedirs(folder, exist_ok=True)

                for i in range(nb_instances):
                    total_jobs = random.randint(job_min, job_max)

                    instances_by_tier = generate_one_ub_scenario_instance(
                        total_jobs=total_jobs,
                        e_ratio=scenario["e_ratio"],
                        existing_cost_range=scenario["existing_cost_range"],
                        new_cost_range=scenario["new_cost_range"],
                        scenario_name=scenario_name,
                        max_operations_per_job=max_operations_per_job,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        pos_time=pos_time,
                        release_spread_existing=release_spread_existing,
                        release_spread_new=release_spread_new,
                        due_date_min=due_date_min,
                    )

                    for tier_name, instance in instances_by_tier.items():
                        path = os.path.join(folder, f"instance_{i + 1}_{tier_name}.json")
                        save_json(instance, path)

                print(f"{size_name}/{scenario_name}/{split_name}: {nb_instances} instances x 3 tiers -> {folder}")

# TEST WITH: python instance_generator.py --train=30 --test=20 --path=./
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled Orders Generator (UB-based)")
    parser.add_argument("--path", help="path to save the instances", required=True)
    parser.add_argument("--train", help="number of training instances per size/scenario", required=True)
    parser.add_argument("--test", help="number of test instances per size/scenario", required=True)
    args = parser.parse_args()

    generate_controlled_orders_ub_sized(
        output_root=args.path + "data/controlled_orders_ub/",
        nb_train=int(args.train),
        nb_test=int(args.test),
        seed=1
    )
