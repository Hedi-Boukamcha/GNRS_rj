# instance_generator_paired.py
import os
import copy
import math
import random
import argparse

from conf import *
from instance_generator import (
    generate_order_template,
    compute_ub_cmax_e,
    compute_ub_cmax_e_n,
    sample_due_dates_from_ub,
    apply_random_costs,
    build_instance,
    save_json,
    CUT_TIME_TIERS,
)

# ###################################
# =*= PAIRED INSTANCE GENERATOR =*=
# ###################################
# Meme generation que instance_generator.py (operations, releases, UB, cut_time, due dates),
# mais chaque instance "structurelle" est sauvegardee deux fois :
#   - same_costs_<E>_<N> : tous les couts = 1.0
#   - diff_costs_<E>_<N> : couts E dans [0.8, 1.0], couts N dans [0.05, 0.4]
# Les deux versions ont exactement les memes jobs, cut_time et due dates : seul le cout change.
# De plus, instance_i part du meme pool de jobs pour les 3 ratios (5_5, 3_7, 7_3) : seul le
# decoupage E/N change (et donc UB, cut_time et due dates, qui en dependent).
# On peut donc comparer same vs diff ET les ratios entre eux, instance par instance.
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# proportion de jobs existants (E); le reste est N
E_RATIOS = {
    "5_5": 0.5,
    "3_7": 0.3,
    "7_3": 0.7,
}

# memes intervalles de couts que SCENARIOS_UB dans instance_generator.py
COST_FAMILIES = {
    "same_costs": {
        "existing_cost_range": (1.0, 1.0),
        "new_cost_range": (1.0, 1.0),
    },
    "diff_costs": {
        "existing_cost_range": (0.8, 1.0),
        "new_cost_range": (0.05, 0.4),
    },
}

def scenario_name(family: str, ratio_name: str) -> str:
    return f"{family}_{ratio_name}"

def nb_existing(total_jobs: int, e_ratio: float) -> int:
    """Arrondi classique (0.5 -> superieur), pas l'arrondi bancaire de round() :
    avec round(), 5 jobs donnait 2 E pour 3_7 ET pour 5_5 (round(1.5) = round(2.5) = 2)."""
    nb_e = math.floor(total_jobs * e_ratio + 0.5)
    return max(1, min(total_jobs - 1, nb_e))

def generate_job_pool(
    total_jobs: int,
    max_operations_per_job: int = 2,
    min_duration: int = 10,
    max_duration: int = 60,
    pos_time: int = 5,
    release_spread: int = 50,
) -> list[dict]:
    """Jobs communs a tous les scenarios d'une instance (operations, release relative, big)."""
    return generate_order_template(
        nb_jobs=total_jobs,
        max_operations_per_job=max_operations_per_job,
        min_duration=min_duration,
        max_duration=max_duration,
        pos_time=pos_time,
        release_spread=release_spread,
    )

def generate_structure(
    job_pool: list[dict],
    e_ratio: float,
    a: int,
    due_date_min: int = 50,
) -> dict:
    """Tout ce qui ne depend pas des couts : decoupage E/N, UB, due dates, cut_time par tier.
    E = les nb_e premiers jobs du pool, N = le reste. Les E de 3_7 sont donc inclus dans
    ceux de 5_5, eux-memes inclus dans ceux de 7_3."""
    total_jobs = len(job_pool)
    nb_e = nb_existing(total_jobs, e_ratio)
    nb_n = total_jobs - nb_e

    existing_jobs = copy.deepcopy(job_pool[:nb_e])
    new_jobs = copy.deepcopy(job_pool[nb_e:])
    ub_cmax_e = compute_ub_cmax_e(existing_jobs)

    existing_jobs = sample_due_dates_from_ub(
        jobs=existing_jobs,
        min_due_date=due_date_min,
        max_due_date=ub_cmax_e / 3,
        base_time=0,
        min_slack=due_date_min
    )

    used_cut_times = set()
    tiers = {}
    for tier_name, cut_rule in CUT_TIME_TIERS.items():
        cut_time = cut_rule(ub_cmax_e)
        while cut_time in used_cut_times:
            cut_time = cut_rule(ub_cmax_e)
        used_cut_times.add(cut_time)
        ub_cmax_e_n = compute_ub_cmax_e_n(new_jobs, cut_time=cut_time, ub_cmax_e=ub_cmax_e)
        new_jobs_dd = sample_due_dates_from_ub(
            jobs=new_jobs,
            min_due_date=cut_time + due_date_min,
            max_due_date=ub_cmax_e_n / 3,
            base_time=cut_time,
            min_slack=due_date_min
        )
        tiers[tier_name] = {
            "cut_time": cut_time,
            "ub_cmax_e_n": ub_cmax_e_n,
            "new_jobs": new_jobs_dd,
        }

    return {
        "total_jobs": total_jobs,
        "nb_e": nb_e,
        "nb_n": nb_n,
        "existing_jobs": existing_jobs,
        "ub_cmax_e": ub_cmax_e,
        "a": a,
        "tiers": tiers,
    }

def instances_from_structure(structure: dict, family: str, scen_name: str) -> dict[str, dict]:
    """Applique les couts d'une famille a une structure. Les couts N sont les memes pour les 3 tiers."""
    costs = COST_FAMILIES[family]
    existing_jobs = apply_random_costs(structure["existing_jobs"], *costs["existing_cost_range"])

    # tirer les couts N une seule fois (meme job N -> meme cout dans early/middle/late)
    nb_n = structure["nb_n"]
    new_costs = [round(random.uniform(*costs["new_cost_range"]), 4) for _ in range(nb_n)]

    instances = {}
    for tier_name, tier in structure["tiers"].items():
        new_jobs = [dict(job) for job in tier["new_jobs"]]
        for job, cost in zip(new_jobs, new_costs):
            job["cost"] = cost

        instance = build_instance(
            existing_jobs=existing_jobs,
            new_jobs=new_jobs,
            cut_time_new_order=tier["cut_time"],
            a=structure["a"],
            scenario_name=scen_name,
            ub_cmax_e=structure["ub_cmax_e"],
            ub_cmax_e_n=tier["ub_cmax_e_n"],
            variant_name=tier_name
        )
        instance["cost_family"] = family
        instance["total_jobs"] = structure["total_jobs"]
        instance["nb_existing_jobs"] = structure["nb_e"]
        instance["nb_new_jobs"] = nb_n
        instances[tier_name] = instance
    return instances

def generate_paired_orders(
    output_root: str,
    nb_train: int,
    nb_test: int,
    seed: int = 1,
    due_date_min: int = 50,
    **pool_kwargs
):
    random.seed(seed)

    for size_name, job_min, job_max in INSTANCES_SIZES:
        for split_name, nb_instances in (("train", nb_train), ("test", nb_test)):
            for i in range(nb_instances):
                # instance i : meme pool de jobs et meme "a" pour les 3 ratios et les 2 familles de couts
                total_jobs = random.randint(job_min, job_max)
                job_pool = generate_job_pool(total_jobs, **pool_kwargs)
                a = random.randint(0, 10) * 10

                for ratio_name, e_ratio in E_RATIOS.items():
                    structure = generate_structure(job_pool, e_ratio, a, due_date_min=due_date_min)

                    for family in COST_FAMILIES:
                        scen = scenario_name(family, ratio_name)
                        folder = os.path.join(output_root, split_name, size_name, scen)
                        for tier_name, instance in instances_from_structure(structure, family, scen).items():
                            save_json(instance, os.path.join(folder, f"instance_{i + 1}_{tier_name}.json"))

            print(f"{size_name}/{split_name}: {nb_instances} instances x {len(E_RATIOS)} ratios x {len(COST_FAMILIES)} cost families x 3 tiers")

# TEST WITH: python instance_generator_paired.py --train=30 --test=20 --path=./
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired same/diff cost Controlled Orders Generator (UB-based)")
    parser.add_argument("--path", help="path to save the instances", required=True)
    parser.add_argument("--train", help="number of training instances per size/ratio", required=True)
    parser.add_argument("--test", help="number of test instances per size/ratio", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    generate_paired_orders(
        output_root=args.path + "data/controlled_orders_paired/",
        nb_train=int(args.train),
        nb_test=int(args.test),
        seed=args.seed
    )
