import os
import random
import copy
import argparse

from conf import *
from generate_controlled_order import (
    randint_interval,
    generate_order_template,
    convert_relative_jobs_to_absolute,
    save_json,
)

__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0"
__license__ = "MIT"

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
    "early": lambda ub: randint_interval(0, int(ub / 4)),
    "middle": lambda ub: randint_interval(int(ub / 4), int(2 * ub / 4)),
    "late": lambda ub: randint_interval(int(2 * ub / 4), int(3 * ub / 4)),
}

def compute_execution_load(jobs: list[dict]) -> int:
    """Execute(jobs): temps d'execution sequentiel pire cas si tous les jobs
    s'enchainaient sans attente (somme des ops + transitions), sans offset de depart."""
    load = 0
    for job in jobs:
        for op in job["operations"]:
            load += 2 * M + op["processing_time"]
            if op["type"] == MACHINE_1:
                load += job["pos_time"] + M
        load += 2 * L
    return load


def estimated_ub_cmax(jobs: list[dict], start_time: int) -> int:
    return max(start_time, 1) + compute_execution_load(jobs)


def compute_ub_cmax_e(existing_jobs: list[dict]) -> int:
    """UBe = max(rj_e) + Execute(E)"""
    rj_e = max((job["relative_release"] for job in existing_jobs), default=0)
    return rj_e + compute_execution_load(existing_jobs)


def compute_ub_cmax_e_n(new_jobs: list[dict], cut_time: int, ub_cmax_e: int) -> int:
    """REST = UBe - cutTime si UBe > cutTime, sinon cutTime.
    UBe+n = max(rj_n, REST) + Execute(N)"""
    rest = (ub_cmax_e - cut_time) if ub_cmax_e > cut_time else cut_time
    rj_n = max((cut_time + job["relative_release"] for job in new_jobs), default=cut_time)
    return max(rj_n, rest) + compute_execution_load(new_jobs)


def sample_due_dates_from_ub(jobs: list[dict], min_due_date: float, max_due_date: float, base_time: int) -> list[dict]:
    jobs = copy.deepcopy(jobs)
    for job in jobs:
        absolute_release = base_time + job["relative_release"]
        # plus petite date a laquelle le job peut finir s'il est traite seul :
        # release + temps de traitement + mouvements necessaires (2*M par op, pos_time+M si MACHINE_1, 2*L)
        job_min_due = estimated_ub_cmax([job], absolute_release)
        job_min_due = max(min_due_date, job_min_due)
        job_max_due = max(job_min_due, max_due_date)  # garde-fou si UB/3 tombe sous job_min_due
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
        base_time=0
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
            base_time=cut_time
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


# TEST WITH: python generate_controlled_order_ub.py --train=30 --test=20 --path=./
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
