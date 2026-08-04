# master_problem/acceptation_method.py
import csv
import os
import random

from gantt_builder.gnn_gantt import gnn_gantt
from models.order import OrderInstance
from models.state import State, JobState
from models.instance import Job
from models.agent import Agent
from simulators.gnn_simulator import build_state_from_cut, display_cut_snapshot, validate_cut_state
from models.environment import Environment
from gnn_solver import search_possible_decisions, take_one_step
from conf import *
import time


# ############################################
# =*= THE ACCEPTATION METHOD FOR NEW JOBS  =*=
# ############################################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0" 
__license__ = "MIT"


def save_step_gantt(env, gantt_dir: str | None, label: str, step: int, cut_times: list[int] | None = None):
    if gantt_dir is None:
        return
    os.makedirs(gantt_dir, exist_ok=True)
    path = os.path.join(gantt_dir, f"{label}_step_{step:03d}.png")

    gnn_gantt(path, env.state, f"{label} | step {step}", cut_times=cut_times or [])

# Definition de la valeur du retard actuel (de reference)
def compute_current_schedule_ref(state: State, all_weights: dict) -> tuple[float, int]:
    """
    Calcule le coût de référence à partir de la cédule actuelle.
    Aucun rescheduling n'est effectué.
    """

    cost_ref = sum(
        all_weights[id(j.job)] * j.delay
        for j in state.job_states
    )

    cmax_ref = state.cmax
    #breakpoint()
    print("\n  === Référence cédule actuelle ===")
    print(f"  tardiness ref existants : {[j.delay for j in state.job_states]}")
    print(f"  weights ref existants   : {[round(all_weights[id(j.job)], 4) for j in state.job_states]}")
    print(f"  cost_ref                : {cost_ref:.4f}")
    print(f"  cmax_ref                : {cmax_ref}")

    return cost_ref, cmax_ref

# calculer les cout
def compute_urgency_weight(j: JobState) -> float:
    r = j.job.release_date
    d = j.job.due_date
    return 1.0 / (1.0 + max(0, d - r))

# calculer les cout du retard
def compute_weighted_tardiness(job_states: list[JobState]) -> float:
    return sum(
        compute_urgency_weight(j) * j.delay
        for j in job_states
    )

def subset_name(subset, new_jobs):
    if not subset:
        return "empty"

    return "_".join(
        f"J{new_jobs.index(job) + 1}"
        for job in subset
    )


def safe_position_name(pos):
    if pos is None:
        return "None"

    if hasattr(pos, "position_type"):
        try:
            return LOCATION_NAMES[pos.position_type]
        except Exception:
            return f"pos_type={pos.position_type}"

    return str(pos)


def safe_event_type_name(event_type):
    try:
        return EVENT_NAMES[event_type]
    except Exception:
        return str(event_type)


def safe_job_name(job):
    if job is None:
        return "None"
    return f"J{job.id + 1}"


def safe_operation_name(operation):
    if operation is None:
        return "None"
    return f"O{operation.id + 1}"


def safe_station_name(station):
    if station is None:
        return "None"
    return f"S{station.id + 1}"


def print_calendar(calendar, title: str):
    print(f"\n--- {title} ---")

    if calendar is None or not calendar.has_events():
        print("  Aucun événement")
        return

    events = sorted(calendar.events, key=lambda e: (e.start, e.end))

    for e in events:
        print(
            f"  [{e.start:>4} -> {e.end:>4}] "
            f"type={safe_event_type_name(e.event_type):<12} | "
            f"job={safe_job_name(e.job):<5} | "
            f"op={safe_operation_name(e.operation):<5} | "
            f"station={safe_station_name(e.station):<5} | "
            f"source={safe_position_name(e.source):<12} | "
            f"dest={safe_position_name(e.dest):<12}"
        )


def check_calendar_overlaps(calendar, title: str):
    if calendar is None or not calendar.has_events():
        return

    events = sorted(calendar.events, key=lambda e: (e.start, e.end))

    for prev, curr in zip(events, events[1:]):
        if curr.start < prev.end:
            print(
                f"  ⚠️ OVERLAP dans {title}: "
                f"[{prev.start}->{prev.end}] "
                f"{safe_event_type_name(prev.event_type)} "
                f"{safe_job_name(prev.job)} "
                f"avec "
                f"[{curr.start}->{curr.end}] "
                f"{safe_event_type_name(curr.event_type)} "
                f"{safe_job_name(curr.job)}"
            )


def print_state_calendars(state: State, title: str = ""):
    print("\n" + "=" * 100)
    print(f"📅 CALENDRIERS {title}")
    print("=" * 100)

    print(
        f"\nSTATE | cmax={state.cmax} | "
        f"start_time={state.start_time} | "
        f"total_delay={state.total_delay}"
    )

    print(
        f"\nMACHINE 1 | free_at={state.machine1.free_at} | "
        f"current_job={safe_job_name(state.machine1.current_job)} | "
        f"pos_is_full={getattr(state.machine1, 'pos_is_full', None)}"
    )
    print_calendar(state.machine1.calendar, "MACHINE 1")

    print(
        f"\nMACHINE 2 | free_at={state.machine2.free_at} | "
        f"current_job={safe_job_name(state.machine2.current_job)}"
    )
    print_calendar(state.machine2.calendar, "MACHINE 2")

    print(
        f"\nROBOT | free_at={state.robot.free_at} | "
        f"current_job={safe_job_name(state.robot.current_job)} | "
        f"location={safe_position_name(state.robot.location)}"
    )
    print_calendar(state.robot.calendar, "ROBOT")

    for s in state.all_stations.stations:
        print(
            f"\nSTATION S{s.id + 1} | free_at={s.free_at} | "
            f"current_job={safe_job_name(s.current_job)} | "
            f"accept_big={s.accept_big}"
        )
        print_calendar(s.calendar, f"STATION S{s.id + 1}")

    for j in state.job_states:
        print(
            f"\nJOB J{j.id + 1} | "
            f"status={j.status} | "
            f"location={safe_position_name(j.location)} | "
            f"current_station={safe_station_name(j.current_station)} | "
            f"release={j.job.release_date} | "
            f"due_date={j.job.due_date} | "
            f"cost={j.job.cost} | "
            f"end={j.end} | "
            f"delay={j.delay}"
        )

        for o in j.operation_states:
            print(
                f"    O{o.id + 1} | "
                f"type={o.operation.type} | "
                f"p={o.operation.processing_time} | "
                f"remaining={o.remaining_time} | "
                f"status={o.status} | "
                f"start={o.start} | "
                f"end={o.end}"
            )

        print_calendar(j.calendar, f"JOB J{j.id + 1}")

    print("\n--- CHECK OVERLAPS ---")

    check_calendar_overlaps(state.machine1.calendar, "MACHINE 1")
    check_calendar_overlaps(state.machine2.calendar, "MACHINE 2")
    check_calendar_overlaps(state.robot.calendar, "ROBOT")

    for s in state.all_stations.stations:
        check_calendar_overlaps(s.calendar, f"STATION S{s.id + 1}")

    for j in state.job_states:
        check_calendar_overlaps(j.calendar, f"JOB J{j.id + 1}")

    print("\n" + "=" * 100)
    print("FIN CALENDRIERS")
    print("=" * 100 + "\n")


# Evaluation des nouveaux jobs (sous ensembles): qq soit un seul job ou bien une combinaison de plusieurs jobs
def _find_wait_time(cut_state: State, cut_time: int) -> int | None:
    """
    Prochain instant où une opération existante en cours d'exécution au cut se termine.
    Retourne None si aucune opération n'est en exécution au cut.
    """
    ends = []
    for j in cut_state.job_states:
        for o in j.operation_states:
            if o.remaining_time > 0:
                if o.status == IN_EXECUTION:
                    last_event = j.calendar.get_last_event()
                    if last_event and last_event.end > cut_time:
                        ends.append(last_event.end)
                break
    return min(ends) if ends else None


def _greedy_rollout(cut_state: State, start_time: int, agent: Agent, device: str) -> Environment | None:
    """
    Rollout greedy du GNN depuis start_time sur cut_state (modifié en place).
    Retourne l'environnement final, ou None si aucune séquence faisable.
    """
    cut_state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=start_time)
    graph = cut_state.to_hyper_graph(last_job_in_pos=-1, current_time=start_time, device=device)
    env = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states), action_time=start_time)
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    while env.possible_decisions:
        q_values = agent.get_all_q_values(env.graph, env.decisionsT)
        ranked_actions = sorted(
            range(len(env.possible_decisions)),
            key=lambda i: q_values[i].item(),
            reverse=True
        )
        success = False
        last_error = None
        for action_id in ranked_actions:
            try:
                env = take_one_step(
                    agent=agent,
                    last_env=env,
                    action_id=action_id,
                    device=device,
                    clone=True
                )
                success = True
                break
            except RuntimeError as e:
                last_error = e
                continue
        if not success:
            print(f"    ⚠️ Aucune décision faisable à cette étape : {last_error}")
            return None
    return env


def evaluate_subset(state: State, subset: list[Job], new_jobs: list[Job], cut_time: int, nb_existing: int, agent: Agent, device: str, all_weights: dict, gantt_dir: str | None = None) -> tuple[float, float, float, int]:
    """
    Reschedule le pool existant + le sous-ensemble S en deux branches :
      - immédiate : re-cédulation dès cut_time (les nouveaux peuvent passer avant les existants) ;
      - différée  : re-cédulation à la fin de l'opération en cours, où existants et nouveaux
        sont simultanément actionnables et le GNN arbitre librement selon ses Q-valeurs.
    La meilleure branche (coût pondéré total, puis cmax) est retenue.
    Retourne cost_existants, cost_nouveaux, total_cost, cmax.
    """
    if gantt_dir is not None:
        os.makedirs(gantt_dir, exist_ok=True)

        existing_gantt_path = os.path.join(
            gantt_dir,
            f"existing_only_cut_{cut_time}.png"
        )

        gnn_gantt(
            existing_gantt_path,
            state,
            f"Existing jobs only | cut={cut_time}",
            cut_times=[cut_time]
        )

        print(f"    📊 Gantt existants sauvegardé : {existing_gantt_path}")

    cut_state = build_state_from_cut(state, cut_time)
    #display_cut_snapshot(cut_state, cut_time)
    #validate_cut_state(cut_state, cut_time)

    cut_state.add_jobs_to_state(subset)
    wait_time = _find_wait_time(cut_state, cut_time)

    branches = [("immédiate", cut_time, cut_state)]
    if wait_time is not None and wait_time > cut_time:
        wait_state = build_state_from_cut(state, wait_time)
        wait_state.add_jobs_to_state(subset)
        branches.append(("différée", wait_time, wait_state))

    best_env = None
    best_costs = None
    for branch_name, start_time, branch_state in branches:
        b_env = _greedy_rollout(branch_state, start_time, agent, device)
        if b_env is None:
            print(f"    ⚠️ Branche {branch_name} (start={start_time}) infaisable")
            continue
        b_existing = b_env.state.job_states[:nb_existing]
        b_new = b_env.state.job_states[nb_existing:]
        b_cost_e = sum(all_weights[id(j.job)] * j.delay for j in b_existing)
        b_cost_n = sum(all_weights[id(j.job)] * j.delay for j in b_new)
        b_total = b_cost_e + b_cost_n
        print(
            f"    branche {branch_name} (start={start_time}): "
            f"cost_existants={b_cost_e:.4f} | cost_nouveaux={b_cost_n:.4f} | "
            f"total={b_total:.4f} | cmax={b_env.state.cmax}"
        )
        if best_env is None or (b_total, b_env.state.cmax) < (best_costs[2], best_env.state.cmax):
            best_env = b_env
            best_costs = (b_cost_e, b_cost_n, b_total)

    if best_env is None:
        print_state_calendars(
            cut_state,
            title=f"| subset={subset_name(subset, new_jobs)} | cut={cut_time} | ECHEC"
        )
        return float("inf"), float("inf"), float("inf"), float("inf")

    env = best_env
    existing_jobs = env.state.job_states[:nb_existing]
    new_jobs_states = env.state.job_states[nb_existing:]
    cost_existants, cost_nouveaux, total_cost = best_costs

    print(f"    tardiness existants: {[j.delay for j in existing_jobs]}")
    print(f"    weights existants: {[round(all_weights[id(j.job)], 4) for j in existing_jobs]}")
    print(f"    weighted tardiness existants: {[round(all_weights[id(j.job)] * j.delay, 4) for j in existing_jobs]}")
    print(f"    cost_existants={cost_existants:.4f}")

    print(f"    tardiness nouveaux: {[j.delay for j in new_jobs_states]}")
    print(f"    weights nouveaux: {[round(all_weights[id(j.job)], 4) for j in new_jobs_states]}")
    print(f"    weighted tardiness nouveaux: {[round(all_weights[id(j.job)] * j.delay, 4) for j in new_jobs_states]}")
    print(f"    cost_nouveaux={cost_nouveaux:.4f}")
    
    if gantt_dir is not None:
        os.makedirs(gantt_dir, exist_ok=True)
        s_name = subset_name(subset, new_jobs)

        gantt_path = os.path.join(
            gantt_dir,
            f"subset_{s_name}_cut_{cut_time}_cmax_{env.state.cmax}_costE_{int(cost_existants)}.png"
        )

        gnn_gantt(
            gantt_path,
            env.state,
            f"Subset {s_name} | cut={cut_time} | costE={cost_existants:.2f} | cmax={env.state.cmax}",
            cut_times=[cut_time]
        )

        print(f"    📊 Gantt subset sauvegardé : {gantt_path}")

    """print_state_calendars(
        env.state,
        title=f"| subset={subset_name(subset, new_jobs)} | cut={cut_time} | cmax={env.state.cmax}"
    )"""
    return cost_existants, cost_nouveaux, total_cost, env.state.cmax


# Recherche en largeur des nouveaux jobs dans l'arbre
def bfs_forward(state: State, new_jobs: list[Job], cut_time: int, agent: Agent, device: str, all_weights: dict, delta_ratio: float = 0.2, gantt_dir: str | None = None) -> list[Job]:
    """
    BFS Forward : explore les sous-ensembles de new_jobs à accepter.
    """

    nb_existing = len(state.job_states)

    start_time = time.perf_counter()

    n_evaluated = 0
    n_valid = 0
    n_pruned = 0
    n_skipped = 0

    print("\n  === Weights utilisés ===")

    print("  Jobs existants:")
    for j in state.job_states:
        print(
            f"    J{j.id + 1} | "
            f"dd={j.job.due_date} | "
            f"weight={all_weights[id(j.job)]:.4f}"
        )

    print("  Nouveaux jobs:")
    for i, job in enumerate(new_jobs):
        print(
            f"    New J{i + 1} | "
            #f"dd={job.due_date} | "
            f"weight={all_weights[id(job)]:.4f}"
        )

    cost_ref, cmax_ref = compute_current_schedule_ref(state, all_weights)
    cost_max = cost_ref * (1 + delta_ratio)

    print(f"\n  cost_ref={cost_ref:.4f} | cost_max={cost_max:.4f}")

    best_subset = []
    best_cmax = float("inf")
    best_cost = float("inf")

    visited = set()
    pruned_subsets = set()

    queue = [[job] for job in new_jobs]

    def subset_key(subset):
        return frozenset(id(j) for j in subset)

    def contains_pruned_subset(key):
        return any(pruned_key.issubset(key) for pruned_key in pruned_subsets)

    while queue:
        current_subset = queue.pop(0)
        current_key = subset_key(current_subset)

        if current_key in visited:
            continue

        if contains_pruned_subset(current_key):
            print(
                f"⏭️ Skip {[f'J{new_jobs.index(j)+1}' for j in current_subset]} "
                f"car contient un subset déjà élagué"
            )
            n_skipped += 1
            visited.add(current_key)
            continue
        visited.add(current_key)
        print(
            f"\n  → Subset="
            f"{[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in current_subset]} "
            f"| size={len(current_subset)}"
        )
        n_evaluated += 1
        cost_existants, cost_nouveaux, total_cost, cmax = evaluate_subset(
            state,
            current_subset,
            new_jobs,
            cut_time,
            nb_existing,
            agent,
            device,
            all_weights,
            gantt_dir=gantt_dir
        )
        print(
            f"cost_existants={cost_existants:.4f} | "
            f"cost_nouveaux={cost_nouveaux:.4f} | "
            f"total_cost={total_cost:.4f} | "
            f"cost_max={cost_max:.4f} | "
            f"cmax={cmax}"
        )
        if cost_existants <= cost_max:
            n_valid += 1
            print("     ✅ Valide → extension")
            if len(current_subset) > len(best_subset):
                best_subset = current_subset
                best_cmax = cmax
                best_cost = total_cost
            elif len(current_subset) == len(best_subset):
                if total_cost < best_cost:
                    best_subset = current_subset
                    best_cmax = cmax
                    best_cost = total_cost
                elif total_cost == best_cost and cmax < best_cmax:
                    best_subset = current_subset
                    best_cmax = cmax
                    best_cost = total_cost
            for job in new_jobs:
                if job not in current_subset:
                    new_subset = current_subset + [job]
                    new_key = subset_key(new_subset)
                    if new_key not in visited:
                        queue.append(new_subset)
        else:
            n_pruned += 1
            print(
                f"     ❌ Élagué "
                f"(cost_existants={cost_existants:.4f} > cost_max={cost_max:.4f})"
            )
            pruned_subsets.add(current_key)

    end_time = time.perf_counter()
    acceptance_time = end_time - start_time

    print("\n  === Statistiques méthode d'acceptation ===")
    #print(f"  Temps computationnel : {acceptance_time:.4f} secondes")
    print(f"  Subsets évalués     : {n_evaluated}")
    print(f"  Subsets valides     : {n_valid}")
    print(f"  Subsets élagués     : {n_pruned}")
    print(f"  Subsets skippés     : {n_skipped}")

    print("\n  === Résultat BFS ===")
    print(
        f"  Best subset : "
        f"{[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in best_subset]}"
    )
    print(f"  Best size   : {len(best_subset)}")
    print(f"  Best cost   : {best_cost:.4f}")
    print(f"  Best cmax   : {best_cmax}")

    return best_subset


# Recherche en largeur inverse : part de l'ensemble complet et retire des jobs
def bfs_backward(state: State, new_jobs: list[Job], cut_time: int, agent: Agent, device: str, all_weights: dict, delta_ratio: float = 0.2, gantt_dir: str | None = None) -> list[Job]:
    """
    BFS Backward (inverse de bfs_forward) : part du sous-ensemble complet des
    new_jobs et retire des jobs un a un, niveau par niveau, jusqu'a trouver un
    niveau contenant au moins un sous-ensemble faisable (cost_existants <=
    cost_max). Comme cost_existants croit avec le nombre de nouveaux jobs
    acceptes (meme hypothese de monotonie que celle utilisee par bfs_forward
    pour elaguer les sur-ensembles), le premier niveau faisable contient
    forcement le/les plus grand(s) sous-ensemble(s) faisable(s) : inutile
    d'explorer les niveaux plus petits. Beaucoup plus rapide que bfs_forward
    quand la plupart des nouveaux jobs sont habituellement acceptables,
    puisque bfs_forward doit alors explorer tout le powerset pour y arriver
    en grandissant depuis les singletons.
    """
    nb_existing = len(state.job_states)

    start_time = time.perf_counter()

    n_evaluated = 0
    n_valid = 0
    n_pruned = 0

    print("\n  === Weights utilisés (BFS backward) ===")

    print("  Jobs existants:")
    for j in state.job_states:
        print(
            f"    J{j.id + 1} | "
            f"dd={j.job.due_date} | "
            f"weight={all_weights[id(j.job)]:.4f}"
        )

    print("  Nouveaux jobs:")
    for i, job in enumerate(new_jobs):
        print(
            f"    New J{i + 1} | "
            f"weight={all_weights[id(job)]:.4f}"
        )

    cost_ref, cmax_ref = compute_current_schedule_ref(state, all_weights)
    cost_max = cost_ref * (1 + delta_ratio)

    print(f"\n  cost_ref={cost_ref:.4f} | cost_max={cost_max:.4f}")

    def subset_key(subset):
        return frozenset(id(j) for j in subset)

    visited = set()
    level = [list(new_jobs)]  # niveau 0 : ensemble complet (tous acceptes)

    best_subset, best_cost, best_cmax = [], cost_ref, cmax_ref

    while level:
        level_results = []  # (subset, total_cost, cmax) des sous-ensembles faisables de ce niveau

        for subset in level:
            key = subset_key(subset)
            if key in visited:
                continue
            visited.add(key)

            print(
                f"\n  → Subset="
                f"{[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in subset]} "
                f"| size={len(subset)}"
            )
            n_evaluated += 1
            cost_existants, cost_nouveaux, total_cost, cmax = evaluate_subset(
                state,
                subset,
                new_jobs,
                cut_time,
                nb_existing,
                agent,
                device,
                all_weights,
                gantt_dir=gantt_dir
            )
            print(
                f"cost_existants={cost_existants:.4f} | "
                f"cost_nouveaux={cost_nouveaux:.4f} | "
                f"total_cost={total_cost:.4f} | "
                f"cost_max={cost_max:.4f} | "
                f"cmax={cmax}"
            )
            if cost_existants <= cost_max:
                n_valid += 1
                print("     ✅ Valide")
                level_results.append((subset, total_cost, cmax))
            else:
                n_pruned += 1
                print(
                    f"     ❌ Élagué "
                    f"(cost_existants={cost_existants:.4f} > cost_max={cost_max:.4f})"
                )

        if level_results:
            best_subset, best_cost, best_cmax = min(level_results, key=lambda r: (r[1], r[2]))
            break

        # Aucun sous-ensemble faisable a ce niveau -> niveau suivant : tous les
        # sous-ensembles obtenus en retirant exactement 1 job supplementaire
        next_level = []
        next_keys = set()
        for subset in level:
            for job in subset:
                child = [j for j in subset if j is not job]
                child_key = subset_key(child)
                if child_key not in visited and child_key not in next_keys:
                    next_keys.add(child_key)
                    next_level.append(child)
        level = next_level

    end_time = time.perf_counter()
    acceptance_time = end_time - start_time

    print("\n  === Statistiques méthode d'acceptation (backward) ===")
    print(f"  Temps computationnel : {acceptance_time:.4f} secondes")
    print(f"  Subsets évalués     : {n_evaluated}")
    print(f"  Subsets valides     : {n_valid}")
    print(f"  Subsets élagués     : {n_pruned}")

    print("\n  === Résultat BFS backward ===")
    print(
        f"  Best subset : "
        f"{[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in best_subset]}"
    )
    print(f"  Best size   : {len(best_subset)}")
    print(f"  Best cost   : {best_cost:.4f}")
    print(f"  Best cmax   : {best_cmax}")

    return best_subset


# Forward et backward tournent ensemble, niveau par niveau, en partageant les
# sous-ensembles elagues trouves par l'un ou l'autre sens de recherche.
def bfs_bidirectional(state: State, new_jobs: list[Job], cut_time: int, agent: Agent, device: str, all_weights: dict, delta_ratio: float = 0.2, gantt_dir: str | None = None) -> list[Job]:
    """
    Fait avancer bfs_forward (grandit depuis les singletons) et bfs_backward
    (retrecit depuis l'ensemble complet) EN MEME TEMPS, un niveau a la fois,
    en partageant un seul ensemble `pruned_keys` de sous-ensembles confirmes
    infaisables (regle de monotonie : si S est infaisable, tout sur-ensemble
    de S l'est aussi). Des qu'un cote evalue un sous-ensemble infaisable,
    l'AUTRE cote en beneficie immediatement au prochain niveau : il saute
    l'evaluation (couteuse, rollout GNN) de tout candidat qui contient ce
    sous-ensemble elague, sans avoir a le decouvrir lui-meme.

    Arret :
      - backward termine des qu'un niveau contient >=1 sous-ensemble faisable
        (comme bfs_backward seul : c'est alors forcement le plus grand
        possible) -> reponse exacte immediate, forward est arrete aussi.
      - si forward trouve un sous-ensemble faisable strictement plus grand
        que la taille du prochain niveau backward a explorer, backward ne
        peut plus faire mieux -> on l'arrete plus tot.
    """
    nb_existing = len(state.job_states)
    start_time = time.perf_counter()

    n_evaluated = 0
    n_pruned = 0
    n_skipped = 0

    cost_ref, cmax_ref = compute_current_schedule_ref(state, all_weights)
    cost_max = cost_ref * (1 + delta_ratio)
    print("\n  === BFS bidirectionnel (forward + backward partages) ===")
    print(f"  cost_ref={cost_ref:.4f} | cost_max={cost_max:.4f}")

    def subset_key(subset):
        return frozenset(id(j) for j in subset)

    pruned_keys: set = set()  # partage entre forward et backward

    def contains_pruned(key) -> bool:
        return any(pk.issubset(key) for pk in pruned_keys)

    def evaluate(subset):
        nonlocal n_evaluated
        n_evaluated += 1
        return evaluate_subset(
            state, subset, new_jobs, cut_time, nb_existing, agent, device, all_weights, gantt_dir=gantt_dir
        )

    best_subset, best_cost, best_cmax = [], cost_ref, cmax_ref

    forward_level = [[job] for job in new_jobs]
    forward_visited = set()

    backward_level = [list(new_jobs)]
    backward_visited = set()

    while forward_level or backward_level:
        # ---------- un niveau backward (retrecit) ----------
        if backward_level:
            level_results = []
            next_backward = []
            next_keys = set()
            for subset in backward_level:
                key = subset_key(subset)
                if key in backward_visited:
                    continue
                backward_visited.add(key)

                if contains_pruned(key):
                    n_skipped += 1
                else:
                    cost_existants, _, total_cost, cmax = evaluate(subset)
                    if cost_existants <= cost_max:
                        level_results.append((subset, total_cost, cmax))
                    else:
                        n_pruned += 1
                        pruned_keys.add(key)

                # on retrecit meme si ce sous-ensemble est infaisable ou skippe :
                # un infaisable n'implique rien sur ses propres sous-ensembles
                for job in subset:
                    child = [j for j in subset if j is not job]
                    child_key = subset_key(child)
                    if child_key not in backward_visited and child_key not in next_keys:
                        next_keys.add(child_key)
                        next_backward.append(child)

            if level_results:
                best_subset, best_cost, best_cmax = min(level_results, key=lambda r: (r[1], r[2]))
                print(f"  ✅ Backward a trouvé un niveau faisable (size={len(best_subset)}) -> arrêt")
                backward_level, forward_level = [], []
                break

            backward_level = next_backward

        # ---------- un niveau forward (grandit) ----------
        if forward_level:
            next_forward = []
            for subset in forward_level:
                key = subset_key(subset)
                if key in forward_visited:
                    continue
                forward_visited.add(key)

                if contains_pruned(key):
                    n_skipped += 1
                    continue  # sur-ensemble d'un infaisable connu -> inutile d'étendre

                cost_existants, _, total_cost, cmax = evaluate(subset)
                if cost_existants <= cost_max:
                    if len(subset) > len(best_subset) or (
                        len(subset) == len(best_subset) and (total_cost, cmax) < (best_cost, best_cmax)
                    ):
                        best_subset, best_cost, best_cmax = subset, total_cost, cmax
                    for job in new_jobs:
                        if job not in subset:
                            child = subset + [job]
                            if subset_key(child) not in forward_visited:
                                next_forward.append(child)
                else:
                    n_pruned += 1
                    pruned_keys.add(key)

            forward_level = next_forward

            # backward ne peut plus battre ce que forward a déjà trouvé
            if backward_level and len(best_subset) > len(backward_level[0]):
                print(
                    f"  ⏭️ Forward a déjà trouvé mieux (size={len(best_subset)}) "
                    f"que ce que backward peut encore atteindre (size={len(backward_level[0])}) -> arrêt backward"
                )
                backward_level = []

    end_time = time.perf_counter()
    print("\n  === Statistiques méthode d'acceptation (bidirectionnel) ===")
    print(f"  Temps computationnel : {end_time - start_time:.4f} secondes")
    print(f"  Subsets évalués     : {n_evaluated}")
    print(f"  Subsets élagués     : {n_pruned}")
    print(f"  Subsets skippés     : {n_skipped} (grâce au partage forward/backward)")

    print("\n  === Résultat BFS bidirectionnel ===")
    print(
        f"  Best subset : "
        f"{[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in best_subset]}"
    )
    print(f"  Best size   : {len(best_subset)}")
    print(f"  Best cost   : {best_cost:.4f}")
    print(f"  Best cmax   : {best_cmax}")

    return best_subset


def save_acceptation_analysis_csv(
    csv_path: str,
    existing_jobs_final,
    accepted_new_jobs_final,
    reference_completion_times: dict
):
    import os
    import csv

    def r2(x):
        if x == "":
            return ""
        if x is None:
            return ""
        return round(x, 2)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Pool",
            "job",
            "Tj_initial",
            "Tj_final",
            "Diff_Tj",
            "dj",
            "(Tj-dj)/dj",
            "((Tj_final)-(Tj_initial))/(Tj_initial)"
        ])

        for j in existing_jobs_final:
            dj = j.job.due_date

            # Cj dans la cédule de référence, avant acceptation
            Cj_ref = reference_completion_times.get(id(j.job), None)

            # Retard initial avant acceptation
            Tj_initial = (
                max(0, Cj_ref - dj)
                if Cj_ref is not None
                else None
            )

            # Cj après acceptation / insertion des nouveaux jobs
            Cj_final = j.end

            # Retard final après acceptation
            Tj_final = max(0, Cj_final - dj)

            diff_Tj = (
                Tj_final - Tj_initial
                if Tj_initial is not None
                else None
            )

            ratio_due = (
                Tj_final - dj / dj
                if dj != 0
                else None
            )

            ratio_Tj = (
                diff_Tj / Tj_initial
                if Tj_initial is not None and Tj_initial != 0
                else None
            )

            writer.writerow([
                "Existants",
                f"J{j.id + 1}",
                r2(Tj_initial),
                r2(Tj_final),
                r2(diff_Tj),
                r2(dj),
                r2(ratio_due),
                r2(ratio_Tj),
                ""
            ])

        for j in accepted_new_jobs_final:
            dj = j.job.due_date

            Cj_new = j.end
            Tj_new = max(0, Cj_new - dj)

            ratio_due = (
                Tj_new / dj
                if dj != 0
                else None
            )

            writer.writerow([
                "Nouveaux acceptés",
                f"J{j.id + 1}",
                "",              # Tj_initial : pas applicable pour les nouveaux
                r2(Tj_new),       # Tj_final
                "",              # Diff_Tj : pas applicable
                r2(dj),
                r2(ratio_due),
                "",              # Ratio_diff_Tj_initial : pas applicable
                ""
            ])            
def acceptation_method(order_instance: OrderInstance, agent: Agent, device: str, delta_ratio: float = 0.2, gantt_dir: str | None = None, save_step_gantts: bool = False, analysis_dir: str | None = None) -> State:
    """
    Pipeline complet : schedule Order 1 puis applique le Master Problem pour chaque order suivant.
    """
    # 1. Scheduler le premier order normalement
    first_order = order_instance.orders[0]
    instance    = first_order.to_instance()
    state       = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
    graph       = state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device=device)
    env         = Environment(graph=graph, state=state, n=len(instance.jobs))
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    all_weights = {}

    # Temps total de toute la méthode d'acceptation
    global_start_time = time.perf_counter()

    step = 0

    """while env.possible_decisions:
        action_id = agent.select_next_decision(
            graph=env.graph,
            decisionsT=env.decisionsT,
            greedy=True
        )

        env = take_one_step(
            agent=agent,
            last_env=env,
            action_id=action_id,
            device=device
        )

        step += 1

        if save_step_gantts:
            save_step_gantt(
                env=env,
                gantt_dir=gantt_dir,
                label="order_1",
                step=step,
                cut_times=[]
            )"""
    
    while env.possible_decisions:
        q_values = agent.get_all_q_values(env.graph, env.decisionsT)

        ranked_actions = sorted(
            range(len(env.possible_decisions)),
            key=lambda i: q_values[i].item(),
            reverse=True
        )

        success = False
        last_error = None

        for action_id in ranked_actions:
            try:
                trial_env = take_one_step(
                    agent=agent,
                    last_env=env,
                    action_id=action_id,
                    device=device,
                    clone=True
                )

                env = trial_env
                success = True
                step += 1

                if save_step_gantts:
                    save_step_gantt(
                        env=env,
                        gantt_dir=gantt_dir,
                        label="order_1",
                        step=step,
                        cut_times=[]
                    )

                break

            except RuntimeError as e:
                last_error = e
                continue

        if not success:
            print(f"⚠️ Scheduling Order 1 incomplet : {last_error}")
            break

    print(f"Order 1 schedulé | Cmax={env.state.cmax} | Tardiness={sum(j.delay for j in env.state.job_states)}")
    first_order = order_instance.orders[0]
    for job in first_order.jobs:
        #all_weights[id(job)] = random.uniform(EXISTING_COST_MIN, EXISTING_COST_MAX)
        all_weights[id(job)] = job.cost

    # 2. Pour chaque order suivant → Master Problem
    for order in order_instance.orders[1:]:
        cut_time  = order.cut_time
        new_jobs  = order.jobs

        for job in new_jobs:
            if id(job) not in all_weights:
                #all_weights[id(job)] = random.uniform(NEW_COST_MIN, NEW_COST_MAX)
                all_weights[id(job)] = job.cost

        # Gantt avant le cut
        if gantt_dir is not None:
            before_cut_path = os.path.join(gantt_dir, f"before_cut_{order.id}.png")
            gnn_gantt(before_cut_path, env.state, f"before cut {order.id}", cut_times=[cut_time])

        print(f"\n=== Order {order.id} | cut_time={cut_time} | {len(new_jobs)} nouveaux jobs ===")

        # 3. Master Problem → choisir le meilleur sous-ensemble
        #best_subset = bfs_forward(env.state, new_jobs, cut_time, agent, device, all_weights, delta_ratio)
        #best_subset = bfs_backward(env.state, new_jobs, cut_time, agent, device, all_weights, delta_ratio)
        subset_gantt_dir = None

        if gantt_dir is not None:
            subset_gantt_dir = os.path.join(
                gantt_dir,
                f"order_{order.id}_subset_tests"
            )

        best_subset = bfs_bidirectional(
            env.state,
            new_jobs,
            cut_time,
            agent,
            device,
            all_weights,
            delta_ratio,
            gantt_dir=subset_gantt_dir
        )
        
        print(f"\n  === Résultat Order {order.id} ===")
        print(f"  Jobs acceptés  : {[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in best_subset]}")
        print(f"  Jobs rejetés   : {[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in new_jobs if j not in best_subset]}")
        print(f"  {len(best_subset)}/{len(new_jobs)} jobs acceptés")


        # 4. Reschedule avec le meilleur sous-ensemble
        if len(best_subset) == 0:
            print(f"  Aucun job accepté → planning original conservé")

            if analysis_dir is not None:
                csv_path = os.path.join(
                    analysis_dir,
                    f"order_{order.id}_acceptance_analysis.csv"
                )

                reference_completion_times = {
                    id(j.job): j.end
                    for j in env.state.job_states
                }

                save_acceptation_analysis_csv(
                    csv_path=csv_path,
                    existing_jobs_final=env.state.job_states,
                    accepted_new_jobs_final=[],
                    reference_completion_times=reference_completion_times
                )

                print(f"  📄 Tableau analyse sauvegardé : {csv_path}")

            continue

        # Re-cédulation finale avec les deux mêmes branches que evaluate_subset,
        # pour que la cédule retenue corresponde à celle qui a justifié l'acceptation.
        base_state = env.state
        reference_completion_times = {id(j.job): j.end for j in base_state.job_states}

        if save_step_gantts:
            print("  ℹ️ save_step_gantts non supporté avec la re-cédulation à deux branches — ignoré")

        cut_state = build_state_from_cut(base_state, cut_time)
        cut_state.add_jobs_to_state(best_subset)
        wait_time = _find_wait_time(cut_state, cut_time)

        branches = [("immédiate", cut_time, cut_state)]
        if wait_time is not None and wait_time > cut_time:
            wait_state = build_state_from_cut(base_state, wait_time)
            wait_state.add_jobs_to_state(best_subset)
            branches.append(("différée", wait_time, wait_state))

        best_env = None
        best_total = None
        for branch_name, start_time, branch_state in branches:
            b_env = _greedy_rollout(branch_state, start_time, agent, device)
            if b_env is None:
                print(f"  ⚠️ Scheduling final : branche {branch_name} (start={start_time}) infaisable")
                continue
            b_total = sum(all_weights[id(j.job)] * j.delay for j in b_env.state.job_states)
            print(
                f"  Scheduling final : branche {branch_name} (start={start_time}) → "
                f"total_cost={b_total:.4f} | cmax={b_env.state.cmax}"
            )
            if best_env is None or (b_total, b_env.state.cmax) < (best_total, best_env.state.cmax):
                best_env = b_env
                best_total = b_total

        if best_env is None:
            print(f"  ⚠️ Scheduling final incomplet pour Order {order.id} : aucune branche faisable")
        else:
            env = best_env

        if gantt_dir is not None:
            os.makedirs(gantt_dir, exist_ok=True)

            final_gantt_path = os.path.join(
                gantt_dir,
                f"order_{order.id}_final_after_acceptance_cmax_{env.state.cmax}.png"
            )

            gnn_gantt(
                final_gantt_path,
                env.state,
                f"Order {order.id} final after acceptance | cmax={env.state.cmax}",
                cut_times=[cut_time]
            )

            print(f"  📊 Gantt final sauvegardé : {final_gantt_path}")
        #print(f"\n=== État J4 après scheduling Order {order.id} ===")
        #print(f"J4 status={env.state.job_states[3].status}")
        #print(f"J4 ops={[(o.status, o.end, o.remaining_time) for o in env.state.job_states[3].operation_states]}")
        #for e in env.state.job_states[3].calendar.events:
            #print(f"  start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}")
        if analysis_dir is not None:
            csv_path = os.path.join(
                analysis_dir,
                f"order_{order.id}_acceptance_analysis.csv"
            )
            nb_existing = len(env.state.job_states) - len(best_subset)
            existing_jobs_final = env.state.job_states[:nb_existing]
            accepted_jobs_final = env.state.job_states[nb_existing:]
            save_acceptation_analysis_csv(
                csv_path=csv_path,
                existing_jobs_final=existing_jobs_final,
                accepted_new_jobs_final=accepted_jobs_final,
                reference_completion_times=reference_completion_times
            )
            print(f"  📄 Tableau analyse sauvegardé : {csv_path}")

        print(f"Order {order.id} schedulé | Cmax={env.state.cmax} | Tardiness={sum(j.delay * j.job.cost for j in env.state.job_states)}")
    global_end_time = time.perf_counter()
    total_acceptance_time = global_end_time - global_start_time

    print("\n=== Statistiques globales méthode d'acceptation ===")
    print(f"  Temps computationnel total : {total_acceptance_time:.4f} secondes")
    print(f"  Nombre d'orders            : {len(order_instance.orders)}")
    print(f"  Nombre total de jobs       : {order_instance.nb_jobs}")


    return env.state