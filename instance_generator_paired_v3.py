# instance_generator_paired_v3.py
import os
import random
import argparse

from conf import *
from instance_generator import estimated_ub_cmax, save_json
from instance_generator_paired import E_RATIOS, COST_FAMILIES, scenario_name
from instance_generator_paired_v2 import (
    TIERS_V2,
    generate_job_pool_v2,
    generate_structure_v2,
    instances_from_structure_v2,
)

# #########################################
# =*= PAIRED INSTANCE GENERATOR (v3) =*=
# #########################################
# Identique a v2 (memes jobs, E, dates dues des E, cut, arrivees des N, couts) SAUF les dates dues des N :
#   - v2 : d_j ~ U(max(cut + 50, r_j + charge), max(., UB(cmaxE+N)))
#   - v3 : la date due d'un N commence au tier qui SUIT son arrivee et ne depasse pas le tier d'apres
#          (fenetre de 2 tiers, tiers en fraction de C_E) :
#            r_j en early  (0.05 - 1/3 C_E) -> d_j dans middle + late          (1/3 - 1   C_E)
#            r_j en middle (1/3  - 2/3 C_E) -> d_j dans late + 1 tier apres    (2/3 - 4/3 C_E)
#            r_j en late   (2/3  - 1   C_E) -> d_j dans 2 tiers apres          (1   - 5/3 C_E)  (par analogie)
#          d_j ~ U(max(lo * C_E, r_j + charge), max(., hi * C_E)) : la borne r_j + charge (job seul, sans
#          attente) est conservee pour qu'un N puisse toujours finir a l'heure ; si elle depasse la fenetre,
#          d_j = r_j + charge (marge nulle, hors fenetre).
# Le tier d'arrivee d'un N est celui de l'instance : cut >= debut du tier et r_j <= min(fin du tier, 0.85) C_E.
# Appariement avec v2 : le tirage principal (random) est exactement celui de v2 (meme seed -> memes instances) ;
# les dates dues v3 des N utilisent un generateur separe (due_rng), donc seules ces dates different de v2.
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "3.0.0"
__license__ = "MIT"

# tier d'arrivee -> fenetre de la date due (fraction de C_E)
DUE_TIERS_V3 = {
    # "early":  (1 / 3, 2 / 3),   # OLD : middle seul
    # "middle": (2 / 3, 1.0),     # OLD : late seul
    # "late":   (1.0, 4 / 3),     # OLD : tier d'apres seul
    "early":  (1 / 3, 1.0),     # NEW : middle + late (ne depasse pas late)
    "middle": (2 / 3, 4 / 3),   # NEW : late + 1 tier apres
    "late":   (1.0, 5 / 3),     # NEW : 2 tiers apres (meme regle, par analogie)
}

def sample_due_dates_next_tier(new_jobs: list[dict], cut_time: int, c_e: int, tier_name: str, rng: random.Random) -> list[dict]:
    """N : d_j ~ U(max(lo * C_E, r_j + charge), max(., hi * C_E)), relative au cut comme en v2."""
    lo, hi = DUE_TIERS_V3[tier_name]
    out = []
    for job in new_jobs:
        j = dict(job)
        absolute_release = cut_time + j["relative_release"]
        d_min = max(lo * c_e, estimated_ub_cmax([j], absolute_release))
        d_max = max(d_min, hi * c_e)
        j["relative_due"] = int(round(rng.uniform(d_min, d_max))) - cut_time
        out.append(j)
    return out

def generate_structure_v3(job_pool: list[dict], e_ratio: float, a: int, due_rng: random.Random) -> dict:
    structure = generate_structure_v2(job_pool, e_ratio, a)  # meme tirage que v2
    c_e = structure["cmax_e_heuristic"]
    for tier_name, tier in structure["tiers"].items():
        tier["new_jobs"] = sample_due_dates_next_tier(tier["new_jobs"], tier["cut_time"], c_e, tier_name, due_rng)
    return structure

def generate_paired_orders_v3(output_root: str, nb_train: int, nb_test: int, seed: int = 1, sizes: list[str] | None = None):
    random.seed(seed)
    due_rng = random.Random(seed + 7919)
    for size_name, job_min, job_max in INSTANCES_SIZES:
        if sizes and size_name not in sizes:
            continue
        for split_name, nb_instances in (("train", nb_train), ("test", nb_test)):
            for i in range(nb_instances):
                total_jobs = random.randint(job_min, job_max)
                job_pool = generate_job_pool_v2(total_jobs)
                a = random.randint(0, 10) * 10
                for ratio_name, e_ratio in E_RATIOS.items():
                    structure = generate_structure_v3(job_pool, e_ratio, a, due_rng)
                    for family in COST_FAMILIES:
                        scen = scenario_name(family, ratio_name)
                        folder = os.path.join(output_root, split_name, size_name, scen)
                        for tier_name, instance in instances_from_structure_v2(structure, family, scen).items():
                            instance["generator"] = "paired_v3"
                            instance["due_window_n"] = [round(f * structure["cmax_e_heuristic"], 2) for f in DUE_TIERS_V3[tier_name]]
                            save_json(instance, os.path.join(folder, f"instance_{i + 1}_{tier_name}.json"))
            print(f"{size_name}/{split_name}: {nb_instances} instances x {len(E_RATIOS)} ratios x {len(COST_FAMILIES)} cost families x {len(TIERS_V2)} tiers")

# TEST WITH: python instance_generator_paired_v3.py --train=30 --test=20 --path=./
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired Controlled Orders Generator v3 (dates dues des N dans le tier suivant leur arrivee)")
    parser.add_argument("--path", help="path to save the instances", required=True)
    parser.add_argument("--train", help="number of training instances per size/ratio", required=True)
    parser.add_argument("--test", help="number of test instances per size/ratio", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sizes", nargs="+", default=None, help="sous-ensemble de tailles, ex: --sizes s m")
    args = parser.parse_args()
    generate_paired_orders_v3(output_root=args.path + "data/controlled_orders_paired_v3/",
                              nb_train=int(args.train), nb_test=int(args.test), seed=args.seed, sizes=args.sizes)
