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

# ======================================================
# CONTROLLED ORDERS GENERATOR - UB(cmax) VERSION
# Meme convention que instance_generator_with_costs.py : 4 tailles
# (s, m, l, xl, via INSTANCES_SIZES), chacune avec nb_train instances
# train et nb_test instances test. Pour chaque taille/instance, 3
# scenarios de couts (same_costs, portion_of_3_7, portion_of_7_3) et,
# pour chaque scenario, 3 variantes de cut_time (early, middle, late).
#
# Chaque instance est un couple d'orders (E = Order 1, N = Order 2) dont :
# - cut_time est place par rapport a UB(cmaxE), la vraie borne superieure
#   du Cmax de E seul ;
# - les due dates de E et N sont bornees par UB(cmaxE) et UB(cmaxE+N)
#   respectivement (memes formules que State.compute_obj_values_and_upper_bounds
#   dans models/state.py).
# ======================================================
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0"
__license__ = "MIT"


# Ratio E / (E+N) pour chaque scenario, + bornes de couts (des nombres
# decimaux, PAS des poids entiers : same_costs=1 partout, les deux autres
# tirent des couts decimaux -> voir le fix int()->float() dans models/order.py
# et acceptation_method.py, sinon ces couts <1 seraient tronques a 0 au chargement).
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
    "early": lambda ub: randint_interval(1, int(ub / 3)),
    "middle": lambda ub: randint_interval(int(ub / 3), int(ub / 2)),
    "late": lambda ub: randint_interval(int(ub / 2), int(ub) - 1),
}


def _sum_worst_case_time(jobs: list[dict], start_time: int) -> int:
    """
    Calcul commun aux deux fonctions UB ci-dessous : reproduit exactement la
    partie "2. Maximal remaining delays and production time (makespan)" de
    State.compute_obj_values_and_upper_bounds (models/state.py), pour une
    liste de jobs dont AUCUNE operation n'est encore executee (on est en
    train de generer l'instance, rien n'a encore ete schedule) :
      - pour chaque job, pour chaque operation :
          + 2*M (deplacements robot) + processing_time
          + pos_time + M en plus si l'operation est de type MACHINE_1
      - + 2*L par job (chargement ET dechargement, puisque "has_one_done"
        est toujours False avant toute execution)
      - le tout part de start_time (deja calcule par l'appelant, voir
        compute_ub_cmax_e / compute_ub_cmax_e_n : c'est le premier instant ou
        TOUS les jobs de la liste sont a la fois "current" et disponibles,
        i.e. leur relative_release est passe).

    Dans state.py, cette somme est accumulee en parcourant les jobs tries par
    due_date/cost (pour aussi calculer ub_delay au passage), mais le total
    final ne depend pas de cet ordre : c'est une simple somme par job.
    Contrairement a state.py (ou current_time est deja >= relative_release de
    tous les jobs actifs, puisqu'un job non encore libere n'est pas propose
    au GNN), ici on genere les jobs AVANT toute execution : rien ne garantit
    que r_j <= start_time, d'ou le calcul de start_time par l'appelant.
    """
    ub_cmax = max(start_time, 1)
    for job in jobs:
        for op in job["operations"]:
            ub_cmax += 2 * M + op["processing_time"]
            if op["type"] == MACHINE_1:
                ub_cmax += job["pos_time"] + M
        ub_cmax += 2 * L  # aucune operation "done" au moment de la generation
    return ub_cmax


def compute_ub_cmax_e(existing_jobs: list[dict], current_time: int = 0) -> int:
    """
    UB(cmaxE) : vraie borne superieure du Cmax de E (Order 1) seul.
    Sert a placer cut_time et a borner la due_date de E.

    Tient compte de r_j (relative_release, absolu ici puisque E demarre a
    base_time=0) : un job ne peut pas commencer avant son r_j, donc le calcul
    ne peut pas partir plus tot que le plus tardif des r_j du groupe (borne
    valide : a partir de cet instant, tous les jobs sont forcement
    disponibles, et le reste du travail peut etre serialise apres).
    """
    max_release = max((job["relative_release"] for job in existing_jobs), default=0)
    start_time = max(current_time, max_release)
    return _sum_worst_case_time(existing_jobs, start_time)


def compute_ub_cmax_e_n(existing_jobs: list[dict], new_jobs: list[dict], current_time: int) -> int:
    """
    UB(cmaxE+N) : vraie borne superieure du Cmax du systeme complet, E (pas
    encore fini) + N (a son arrivee), calculee en combinant les DEUX vraies
    listes de jobs (pas un simple compte) -> c'est ce qui differencie E de N.
    Sert a borner la due_date de N.

    Tient compte de r_j des DEUX groupes : existing_jobs["relative_release"]
    est absolu (base_time=0), new_jobs["relative_release"] est relatif a
    cut_time (=current_time). On les convertit en releases absolues pour
    trouver le plus tardif r_j du groupe combine (meme logique que
    compute_ub_cmax_e) : c'est souvent un job N qui arrive tard apres
    cut_time qui repousse ce start_time.
    """
    max_release_e = max((job["relative_release"] for job in existing_jobs), default=0)
    max_release_n = max((current_time + job["relative_release"] for job in new_jobs), default=current_time)
    start_time = max(current_time, max_release_e, max_release_n)
    return _sum_worst_case_time(existing_jobs + new_jobs, start_time)


def sample_due_dates_from_ub(jobs: list[dict], min_due_date: float, max_due_date: float, base_time: int) -> list[dict]:
    """
    Tire relative_due pour chaque job independamment dans [min_due_date, max_due_date]
    (bornes ABSOLUES, i.e. par rapport au temps 0 global) :
      - E (Order 1, base_time=0)        : min=due_date_min, max=UB(cmaxE)/3
      - N (Order 2, base_time=cut_time) : min=cut_time+due_date_min, max=UB(cmaxE+N)/3
    base_time est soustrait pour obtenir relative_due, coherent avec
    convert_relative_jobs_to_absolute qui rajoute base_time plus tard.
    """
    jobs = copy.deepcopy(jobs)
    max_due_date = max(min_due_date, max_due_date)  # garde-fou si UB/3 tombe sous min_due_date
    for job in jobs:
        due_absolute = random.uniform(min_due_date, max_due_date)
        job["relative_due"] = int(round(due_absolute - base_time))
    return jobs


def apply_random_costs(jobs: list[dict], cost_min: float, cost_max: float) -> list[dict]:
    """
    Comme generate_controlled_order.apply_costs_to_template, mais avec des
    couts decimaux (random.uniform), puisque les scenarios ici utilisent des
    couts fractionnaires (ex: [0.05, 0.4]) et non plus des poids entiers.
    """
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
    """
    Genere une liste de jobs E (Order 1) / N (Order 2), repartie selon e_ratio
    (nb_e = round(total_jobs * e_ratio), le reste va a N), et retourne un
    dict {tier: instance_json} pour les 3 variantes de cut_time (early,
    middle, late) definies dans CUT_TIME_TIERS. Les 3 variantes partagent les
    memes jobs E/N (memes operations/couts) ; seul cut_time (et donc les due
    dates de N, qui en dependent via UB(cmaxE+N)) change.
    """
    nb_e = max(1, min(total_jobs - 1, round(total_jobs * e_ratio)))
    nb_n = total_jobs - nb_e

    existing_jobs = generate_order_template(
        nb_jobs=nb_e,
        max_operations_per_job=max_operations_per_job,
        min_duration=min_duration,
        max_duration=max_duration,
        pos_time=pos_time,
        release_spread=release_spread_existing,
        due_slack_min=0,
        due_slack_max=0,
        cost_min=1,
        cost_max=1
    )
    new_jobs = generate_order_template(
        nb_jobs=nb_n,
        max_operations_per_job=max_operations_per_job,
        min_duration=min_duration,
        max_duration=max_duration,
        pos_time=pos_time,
        release_spread=release_spread_new,
        due_slack_min=0,
        due_slack_max=0,
        cost_min=1,
        cost_max=1
    )

    # 1) UB(cmaxE) : vraie borne sup du Cmax de E seul
    #    -> sert a placer cut_time ET a borner la due_date de E
    ub_cmax_e = compute_ub_cmax_e(existing_jobs, current_time=0)

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

        # 2) UB(cmaxE+N) : distingue E et N (deux vraies listes de jobs
        #    combinees) -> sert a borner la due_date de N
        ub_cmax_e_n = compute_ub_cmax_e_n(
            existing_jobs,
            new_jobs,
            current_time=cut_time
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
    """
    Pour chaque taille de INSTANCES_SIZES (s, m, l, xl), genere nb_train
    instances "train" et nb_test instances "test". Chaque instance est
    declinee dans les 3 scenarios de SCENARIOS_UB, chacun en 3 variantes de
    cut_time (early/middle/late) -> fichiers
    <output_root>/<train|test>/<size>/<scenario>/instance_<i>_<tier>.json
    """
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
