import json
import os
import random
import copy

from conf import *


# ======================================================
# CONTROLLED ORDERS GENERATOR
# Same jobs inside each inst folder, only cut_time changes
# ======================================================


def randint_interval(low: int, high: int) -> int:
    low = int(low)
    high = int(high)

    if low < 1:
        low = 1

    if high < low:
        high = low

    return random.randint(low, high)


def estimate_cmax_order(
    nb_jobs: int,
    max_operations_per_job: int,
    max_duration: int,
    pos_time: int
) -> int:
    """
    Estimation grossière du Cmax de l'Order 1.
    Utilisée seulement pour générer les dates d'arrivée.
    """
    return (
        (2 * L + pos_time) * nb_jobs
        + max_operations_per_job * (2 * M + max_duration)
    )


def generate_relative_job(
    max_operations_per_job: int = 2,
    types_operations: list[int] = [0, 1],
    min_duration: int = 10,
    max_duration: int = 60,
    pos_time: int = 5,
    release_spread: int = 50,
    due_slack_min: int = 10,
    due_slack_max: int = 80,
    cost_min: int = 1,
    cost_max: int = 10
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


def build_instance_from_templates(
    initial_jobs_template: list[dict],
    new_jobs_template: list[dict],
    cut_time_new_order: int,
    a: int,
    scenario_name: str,
    estimated_cmax: int,
    variant_name: str
) -> dict:
    order_1_jobs = convert_relative_jobs_to_absolute(
        relative_jobs=initial_jobs_template,
        base_time=0
    )

    order_2_jobs = convert_relative_jobs_to_absolute(
        relative_jobs=new_jobs_template,
        base_time=cut_time_new_order
    )

    return {
        "a": a,
        "scenario": scenario_name,
        "estimated_cmax_order_1": estimated_cmax,
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


def save_json(instance: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(instance, f, indent=4)

    print(f"Saved: {path}")


def recompute_relative_due_dates(
    jobs_template: list[dict],
    due_slack_min: int,
    due_slack_max: int
) -> list[dict]:
    jobs = copy.deepcopy(jobs_template)

    for job in jobs:
        nb_operations = len(job["operations"])
        proc_total = sum(op["processing_time"] for op in job["operations"])

        min_treat_time = (
            proc_total
            + (2 * L + job["pos_time"])
            + nb_operations * (2 * M)
        )

        job["relative_due"] = (
            job["relative_release"]
            + min_treat_time
            + random.randint(due_slack_min, due_slack_max)
        )

    return jobs


def apply_costs_to_template(
    jobs_template: list[dict],
    cost_min: int,
    cost_max: int
) -> list[dict]:
    jobs = copy.deepcopy(jobs_template)

    for job in jobs:
        job["cost"] = random.randint(cost_min, cost_max)

    return jobs


def generate_controlled_orders(
    output_root: str = "data/controlled_orders/test",
    max_operations_per_job: int = 2,
    min_duration: int = 10,
    max_duration: int = 60,
    pos_time: int = 5,
    release_spread_existing: int = 50,
    release_spread_new: int = 50,
    existing_due_slack: tuple[int, int] = (10, 80),
    seed: int = 1
):
    """
    Génère des instances contrôlées.

    Principe :
    - même instance de base pour tous les scénarios
    - mêmes opérations
    - mêmes big/small
    - mêmes release dates relatives
    - mêmes cut_time
    - seuls les coûts changent selon le scénario
    - les due dates des nouveaux jobs changent seulement pour les scénarios dd_serre_N
    """

    random.seed(seed)

    cut_time_variants = {
        "early_1_to_cmax_over_3": lambda cmax: randint_interval(
            1,
            int(cmax / 3)
        ),

        "middle_cmax_over_3_to_cmax_over_2": lambda cmax: randint_interval(
            int(cmax / 3),
            int(cmax / 2)
        ),

        "late_cmax_over_2_to_before_cmax": lambda cmax: randint_interval(
            int(cmax / 2),
            int(cmax) - 1
        )
    }

    # Slacks normaux pour les nouveaux jobs
    normal_new_due_slack_min, normal_new_due_slack_max = SCENARIOS["same_costs"]["new_due_slack"]

    # Slacks serrés pour les nouveaux jobs
    tight_new_due_slack_min, tight_new_due_slack_max = SCENARIOS["same_costs_dd_serre_N"]["new_due_slack"]

    for family_name, config in FAMILIES_CONFIG.items():

        nb_initial_jobs = config["nb_initial_jobs"]
        nb_new_jobs = config["nb_new_jobs"]

        estimated_cmax = estimate_cmax_order(
            nb_jobs=nb_initial_jobs,
            max_operations_per_job=max_operations_per_job,
            max_duration=max_duration,
            pos_time=pos_time
        )

        # Même a pour tous les scénarios de cette instance
        a = random.randint(0, 10) * 10

        # --------------------------------------------------
        # 1) Génération de l'instance de base UNE SEULE FOIS
        # --------------------------------------------------

        base_initial_jobs_template = generate_order_template(
            nb_jobs=nb_initial_jobs,
            max_operations_per_job=max_operations_per_job,
            min_duration=min_duration,
            max_duration=max_duration,
            pos_time=pos_time,
            release_spread=release_spread_existing,
            due_slack_min=existing_due_slack[0],
            due_slack_max=existing_due_slack[1],
            cost_min=1,
            cost_max=1
        )

        base_new_jobs_template_normal = generate_order_template(
            nb_jobs=nb_new_jobs,
            max_operations_per_job=max_operations_per_job,
            min_duration=min_duration,
            max_duration=max_duration,
            pos_time=pos_time,
            release_spread=release_spread_new,
            due_slack_min=normal_new_due_slack_min,
            due_slack_max=normal_new_due_slack_max,
            cost_min=1,
            cost_max=1
        )

        # Même structure et mêmes release dates, mais due dates serrées
        base_new_jobs_template_tight = recompute_relative_due_dates(
            jobs_template=base_new_jobs_template_normal,
            due_slack_min=tight_new_due_slack_min,
            due_slack_max=tight_new_due_slack_max
        )

        # --------------------------------------------------
        # 2) Génération des cut_time UNE SEULE FOIS
        # --------------------------------------------------

        used_cut_times = set()
        fixed_cut_times = {}

        for variant_name, cut_rule in cut_time_variants.items():

            cut_time = cut_rule(estimated_cmax)

            while cut_time in used_cut_times:
                cut_time = cut_rule(estimated_cmax)

            used_cut_times.add(cut_time)
            fixed_cut_times[variant_name] = cut_time

        print("\n==================================================")
        print(f"FAMILY: {family_name}")
        print("==================================================")
        print(f"  total jobs      = {nb_initial_jobs + nb_new_jobs}")
        print(f"  existing jobs   = {nb_initial_jobs}")
        print(f"  new jobs        = {nb_new_jobs}")
        print(f"  estimated_cmax  = {estimated_cmax}")
        print(f"  cut_times       = {fixed_cut_times}")

        # --------------------------------------------------
        # 3) Application des scénarios
        # --------------------------------------------------

        for scenario_name, scenario in SCENARIOS.items():

            existing_cost_min, existing_cost_max = scenario["existing_cost_range"]
            new_cost_min, new_cost_max = scenario["new_cost_range"]

            scenario_folder = os.path.join(output_root, scenario_name)
            family_folder = os.path.join(scenario_folder, family_name)
            os.makedirs(family_folder, exist_ok=True)

            print(f"\n{scenario_name}/{family_name}")

            # Même instance de base, seuls les coûts changent
            initial_jobs_template = apply_costs_to_template(
                jobs_template=base_initial_jobs_template,
                cost_min=existing_cost_min,
                cost_max=existing_cost_max
            )

            # Si scénario avec dj serré pour les nouveaux jobs
            if "dd_serre_N" in scenario_name:
                selected_new_jobs_base = base_new_jobs_template_tight
            else:
                selected_new_jobs_base = base_new_jobs_template_normal

            new_jobs_template = apply_costs_to_template(
                jobs_template=selected_new_jobs_base,
                cost_min=new_cost_min,
                cost_max=new_cost_max
            )

            for variant_name, cut_time in fixed_cut_times.items():

                instance = build_instance_from_templates(
                    initial_jobs_template=initial_jobs_template,
                    new_jobs_template=new_jobs_template,
                    cut_time_new_order=cut_time,
                    a=a,
                    scenario_name=scenario_name,
                    estimated_cmax=estimated_cmax,
                    variant_name=variant_name
                )

                instance["total_jobs"] = nb_initial_jobs + nb_new_jobs
                instance["nb_existing_jobs"] = nb_initial_jobs
                instance["nb_new_jobs"] = nb_new_jobs

                output_path = os.path.join(
                    family_folder,
                    f"{variant_name}.json"
                )

                save_json(instance, output_path)

if __name__ == "__main__":
    generate_controlled_orders(
        output_root="data/controlled_orders/test",
        max_operations_per_job=2,
        min_duration=10,
        max_duration=60,
        pos_time=5,
        release_spread_existing=50,
        release_spread_new=50,
        existing_due_slack=(10, 80),
        seed=1
    )