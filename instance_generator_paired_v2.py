# instance_generator_paired_v2.py
import io
import os
import copy
import random
import argparse
import contextlib

from conf import *
from models.instance import Instance, Job, Operation
from models.state import Decision
from heuristic.local_search import ls as LS
from instance_generator import (
    generate_order_template,
    compute_ub_cmax_e,
    compute_ub_cmax_e_n,
    estimated_ub_cmax,
    build_instance,
    save_json,
)
from instance_generator_paired import E_RATIOS, COST_FAMILIES, scenario_name, nb_existing

# #########################################
# =*= PAIRED INSTANCE GENERATOR (v2) =*=
# #########################################
# Meme principe d'appariement que instance_generator_paired.py :
#   - instance_i part du meme pool de jobs pour les 3 ratios (5_5, 3_7, 7_3), E = les nb_e premiers jobs ;
#   - same_costs_* et diff_costs_* ont exactement les memes jobs, cut_time, releases et due dates.
#
# Ce qui change par rapport a v1 (le cut et les arrivees des N sont ancres sur le VRAI planning des E) :
#   1. C_E = fin des E ordonnances seuls par une heuristique fixe : recherche locale depuis plusieurs ordres
#      (FIFO, SPT, LPT, 20 ordres aleatoires), on garde le plus petit Cmax.
#   2. Tiers definis en fraction de C_E. Le cut est tire dans la PREMIERE MOITIE de son tier, pour que la
#      fenetre d'arrivee des N ne soit jamais quasi nulle (au moins MIN_WINDOW_FRAC * C_E).
#   3. Les N arrivent uniformement entre le cut et min(fin du tier, ARRIVAL_MARGIN * C_E) :
#      tous les N arrivent pendant le planning des E (plus de U(0, 50) fixe).
#   4. Inchange par rapport a v1 : releases des E dans U(0, 50). Dates dues toujours par UB, mais plafond UB
#      au lieu de UB/3 et sans garde-fou +50 (E : U(d_min, max(d_min, UB(cmaxE))) ; N : idem avec UB(cmaxE+N)).
#   5. Controle a la generation : chaque N doit arriver avant C_E (assert).
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0"
__license__ = "MIT"

# tiers en fraction de C_E : (debut, fin). Le cut est tire dans [debut, debut + (fin - debut) / 2].
TIERS_V2 = {
    "early":  (0.05, 1 / 3),
    "middle": (1 / 3, 2 / 3),
    "late":   (2 / 3, 1.0),
}
ARRIVAL_MARGIN  = 0.85         # les N arrivent au plus tard a ARRIVAL_MARGIN * C_E (l agent finit les E jusqu a 13 % avant l heuristique)
MAX_CUT_RETRIES = 50
MIN_WINDOW_FRAC = 0.1          # fenetre d'arrivee des N d'au moins 10 % de C_E
HEURISTIC_RESTARTS = 20         # ordres aleatoires testes pour estimer C_E
DUE_FACTOR = float(os.environ.get("GNRS_DUE_FACTOR", 1.0))  # plafond des dates dues = DUE_FACTOR * UB (v1 : 1/3, avec garde-fou)

# ---------------------------------------------------------------------------------------------
# Planning heuristique (fixe et deterministe : ne depend pas de l'agent)
# ---------------------------------------------------------------------------------------------
def _to_instance(jobs: list[dict]) -> Instance:
    """Due date = release : la recherche locale minimise alors temps de sejour + Cmax,
    sans dependre des dates dues qu'on est justement en train de construire."""
    out, n = [], 0
    for j in jobs:
        ops = [Operation(type=o["type"], machineing_time=o["processing_time"]) for o in j["operations"]]
        out.append(Job(big=j["big"], due_date=j["release_date"], release_date=j["release_date"],
                       pos_time=j["pos_time"], operations=ops, status=0, blocked=0, cost=1.0))
        n += len(ops)
    return Instance(jobs=out, n=n)

def _run_order(jobs: list[dict], instance: Instance, order: list[int]):
    """order = sequence de jobs au niveau des OPERATIONS : la k-ieme apparition du job j designe sa
    k-ieme operation (representation classique du job shop). Permet d'entrelacer les operations
    de jobs differents, comme le fait l'agent."""
    seen = {}
    decisions = []
    for k in order:
        o_id = seen.get(k, 0)
        seen[k] = o_id + 1
        decisions.append(Decision(job_id=k, operation_id=o_id, job_id_in_graph=0,
                                  machine=jobs[k]["operations"][o_id]["type"], parallel=False, comp=-1))
    with contextlib.redirect_stdout(io.StringIO()):  # la recherche locale est tres bavarde
        return LS(instance, decisions)

def heuristic_completion_times(jobs: list[dict], restarts: int = 0) -> tuple[list[int], int]:
    """Recherche locale existante a partir d'un ordre de depart. restarts = 0 : FIFO seul.
    restarts > 0 : FIFO, SPT, LPT + `restarts` ordres aleatoires, on garde le plus petit Cmax
    (estimation prudente de la fin des E : l'agent explore aussi plusieurs ordres).
    Generateur aleatoire local et deterministe : ne perturbe pas le tirage des instances."""
    instance = _to_instance(jobs)
    idx = list(range(len(jobs)))
    proc = lambda k: sum(o["processing_time"] for o in jobs[k]["operations"])
    contiguous = lambda job_order: [k for k in job_order for _ in jobs[k]["operations"]]
    orders = [contiguous(sorted(idx, key=lambda k: (jobs[k]["release_date"], k)))]
    if restarts > 0:
        orders += [contiguous(sorted(idx, key=lambda k: (proc(k), k))),
                   contiguous(sorted(idx, key=lambda k: (-proc(k), k)))]
        rng = random.Random(1000 * len(jobs) + sum(j["release_date"] for j in jobs))
        base = contiguous(idx)
        for _ in range(restarts):  # ordres aleatoires au niveau des operations (entrelacement)
            o = base[:]
            rng.shuffle(o)
            orders.append(o)
    best = None
    for order in orders:
        state = _run_order(jobs, instance, order)
        if best is None or state.cmax < best.cmax:
            best = state
    ends = {js.id: js.end for js in best.job_states}
    return [int(ends[k]) for k in idx], int(best.cmax)

# ---------------------------------------------------------------------------------------------
# Pool de jobs commun aux 3 ratios
# ---------------------------------------------------------------------------------------------
def generate_job_pool_v2(total_jobs: int, max_operations_per_job: int = 2, min_duration: int = 10,
                         max_duration: int = 60, pos_time: int = 5, release_spread: int = 50) -> list[dict]:
    """Comme v1 (operations, big, pos_time, relative_release dans U(0, release_spread)).
    On ajoute un tirage u_release dans U(0,1) par job, reutilise par tous les ratios et tous les tiers :
    il fixe la position d'arrivee d'un job N dans la fenetre [cut, fin de fenetre]."""
    pool = generate_order_template(nb_jobs=total_jobs, max_operations_per_job=max_operations_per_job,
                                   min_duration=min_duration, max_duration=max_duration, pos_time=pos_time,
                                   release_spread=release_spread)
    for job in pool:
        job.pop("relative_due", None)
        job["u_release"] = random.random()
    return pool

def _clean(job: dict) -> dict:
    j = copy.deepcopy(job)
    j.pop("u_release", None)
    return j

# ---------------------------------------------------------------------------------------------
# Structure d'une instance (tout sauf les couts)
# ---------------------------------------------------------------------------------------------
def sample_due_dates_no_guard(jobs: list[dict], min_due_date: float, max_due_date: float, base_time: int) -> list[dict]:
    """Comme instance_generator.sample_due_dates_from_ub, SANS le garde-fou de +50 :
    d ~ U(d_min, max(d_min, b_max)). Si b_max < d_min, la date due vaut d_min (marge nulle)."""
    jobs = copy.deepcopy(jobs)
    for job in jobs:
        absolute_release = base_time + job["relative_release"]
        job_min_due = max(min_due_date, estimated_ub_cmax([job], absolute_release))
        job_max_due = max(job_min_due, max_due_date)
        job["relative_due"] = int(round(random.uniform(job_min_due, job_max_due) - base_time))
    return jobs

def generate_structure_v2(job_pool: list[dict], e_ratio: float, a: int, due_date_min: int = 50) -> dict:
    total_jobs = len(job_pool)
    nb_e = nb_existing(total_jobs, e_ratio)
    nb_n = total_jobs - nb_e
    pool_e, pool_n = job_pool[:nb_e], job_pool[nb_e:]

    # 1. E : releases U(0, 50) comme v1, puis vraie fin des E par l'heuristique (sert au cut et aux arrivees N)
    existing = [_clean(job) for job in pool_e]
    _, c_e = heuristic_completion_times([{**j, "release_date": j["relative_release"]} for j in existing],
                                        restarts=HEURISTIC_RESTARTS)

    # 2. dates dues des E : U(max(50, r + charge), DUE_FACTOR * UB(cmaxE)), sans garde-fou
    ub_e = compute_ub_cmax_e(existing)
    existing = sample_due_dates_no_guard(jobs=existing, min_due_date=due_date_min, max_due_date=DUE_FACTOR * ub_e, base_time=0)

    tiers = {}
    for tier_name, (lo, hi) in TIERS_V2.items():
        arrival_end_frac = min(hi, ARRIVAL_MARGIN)
        for _ in range(MAX_CUT_RETRIES):
            cut_hi = min(lo + (hi - lo) / 2, arrival_end_frac - MIN_WINDOW_FRAC)
            cut_frac = random.uniform(lo, max(lo, cut_hi))
            cut_time = max(1, int(round(cut_frac * c_e)))
            arrival_end = int(arrival_end_frac * c_e)
            if arrival_end > cut_time:
                break
        else:
            raise RuntimeError(f"fenetre d'arrivee vide (C_E={c_e}, tier={tier_name})")

        # 3. arrivees des N entre le cut et la fin de la fenetre (toujours avant C_E), relatives au cut
        new_jobs = []
        for job in pool_n:
            j = _clean(job)
            j["relative_release"] = int(round(job["u_release"] * (arrival_end - cut_time)))
            new_jobs.append(j)
        assert all(cut_time + j["relative_release"] < c_e for j in new_jobs), "un N arrive apres la fin des E"

        # 4. dates dues des N : U(max(cut + 50, r + charge), DUE_FACTOR * UB(cmaxE+N)), sans garde-fou
        ub_en = compute_ub_cmax_e_n(new_jobs, cut_time=cut_time, ub_cmax_e=ub_e)
        new_jobs = sample_due_dates_no_guard(jobs=new_jobs, min_due_date=cut_time + due_date_min,
                                             max_due_date=DUE_FACTOR * ub_en, base_time=cut_time)
        tiers[tier_name] = {"cut_time": cut_time, "arrival_end": arrival_end, "ub_cmax_e_n": ub_en, "new_jobs": new_jobs}

    return {"total_jobs": total_jobs, "nb_e": nb_e, "nb_n": nb_n, "existing_jobs": existing,
            "cmax_e_heuristic": c_e, "ub_cmax_e": ub_e, "a": a, "tiers": tiers}

def instances_from_structure_v2(structure: dict, family: str, scen_name: str) -> dict[str, dict]:
    """Couts tires une fois par job (memes couts N dans early/middle/late), comme v1."""
    costs = COST_FAMILIES[family]
    existing = [dict(j, cost=round(random.uniform(*costs["existing_cost_range"]), 4)) for j in structure["existing_jobs"]]
    new_costs = [round(random.uniform(*costs["new_cost_range"]), 4) for _ in range(structure["nb_n"])]
    instances = {}
    for tier_name, tier in structure["tiers"].items():
        new_jobs = [dict(j, cost=c) for j, c in zip(tier["new_jobs"], new_costs)]
        instance = build_instance(existing_jobs=existing, new_jobs=new_jobs, cut_time_new_order=tier["cut_time"],
                                  a=structure["a"], scenario_name=scen_name, ub_cmax_e=structure["ub_cmax_e"],
                                  ub_cmax_e_n=tier["ub_cmax_e_n"], variant_name=tier_name)
        instance["generator"] = "paired_v2"
        instance["cmax_e_heuristic"] = structure["cmax_e_heuristic"]
        instance["arrival_window"] = [tier["cut_time"], tier["arrival_end"]]
        instance["cost_family"] = family
        instance["total_jobs"] = structure["total_jobs"]
        instance["nb_existing_jobs"] = structure["nb_e"]
        instance["nb_new_jobs"] = structure["nb_n"]
        instances[tier_name] = instance
    return instances

def generate_paired_orders_v2(output_root: str, nb_train: int, nb_test: int, seed: int = 1, sizes: list[str] | None = None):
    random.seed(seed)
    for size_name, job_min, job_max in INSTANCES_SIZES:
        if sizes and size_name not in sizes:
            continue
        for split_name, nb_instances in (("train", nb_train), ("test", nb_test)):
            for i in range(nb_instances):
                total_jobs = random.randint(job_min, job_max)
                job_pool = generate_job_pool_v2(total_jobs)
                a = random.randint(0, 10) * 10
                for ratio_name, e_ratio in E_RATIOS.items():
                    structure = generate_structure_v2(job_pool, e_ratio, a)
                    for family in COST_FAMILIES:
                        scen = scenario_name(family, ratio_name)
                        folder = os.path.join(output_root, split_name, size_name, scen)
                        for tier_name, instance in instances_from_structure_v2(structure, family, scen).items():
                            save_json(instance, os.path.join(folder, f"instance_{i + 1}_{tier_name}.json"))
            print(f"{size_name}/{split_name}: {nb_instances} instances x {len(E_RATIOS)} ratios x {len(COST_FAMILIES)} cost families x {len(TIERS_V2)} tiers")

# TEST WITH: python instance_generator_paired_v2.py --train=30 --test=20 --path=./
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired same/diff cost Controlled Orders Generator v2 (ancre sur le planning reel des E)")
    parser.add_argument("--path", help="path to save the instances", required=True)
    parser.add_argument("--train", help="number of training instances per size/ratio", required=True)
    parser.add_argument("--test", help="number of test instances per size/ratio", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sizes", nargs="+", default=None, help="sous-ensemble de tailles, ex: --sizes s m")
    args = parser.parse_args()
    generate_paired_orders_v2(output_root=args.path + "data/controlled_orders_paired_v2/",
                              nb_train=int(args.train), nb_test=int(args.test), seed=args.seed, sizes=args.sizes)
