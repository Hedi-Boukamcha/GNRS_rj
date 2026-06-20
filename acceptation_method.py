import random

from gantt_builder.gnn_gantt import gnn_gantt
from models.order import OrderInstance
from models.state import State, JobState
from models.instance import Job
from models.agent import Agent
from simulators.gnn_simulator import build_state_from_cut
from models.environment import Environment
from gnn_solver_costs import search_possible_decisions, take_one_step
from conf import *
import time


# ############################################
# =*= THE ACCEPTATION METHOD FOR NEW JOBS  =*=
# ############################################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0" 
__license__ = "MIT"


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

# Evaluation des nouveaux jobs (sous ensembles): qq soit un seul job ou bien une combinaison de plusieurs jobs
def evaluate_subset(state: State, subset: list[Job], new_jobs: list[Job], cut_time: int, nb_existing: int, agent: Agent, device: str, all_weights: dict) -> tuple[float, float, float, int]:
    """
    Reschedule le pool existant + le sous-ensemble S.
    Retourne cost_existants, cost_nouveaux, total_cost, cmax.
    """

    cut_state = build_state_from_cut(state, cut_time)
    cut_state.add_jobs_to_state(subset)
    graph = cut_state.to_hyper_graph_costs(last_job_in_pos=-1, current_time=cut_time, device=device)
    env = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states), action_time=cut_time)
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    while env.possible_decisions:
        action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)

    existing_jobs = env.state.job_states[:nb_existing]
    new_jobs_states = env.state.job_states[nb_existing:]

    cost_existants = sum(
        all_weights[id(j.job)] * j.delay
        for j in existing_jobs
    )

    cost_nouveaux = sum(
        all_weights[id(j.job)] * j.delay
        for j in new_jobs_states
    )

    total_cost = cost_existants + cost_nouveaux

    print(f"    tardiness existants: {[j.delay for j in existing_jobs]}")
    print(f"    weights existants: {[round(all_weights[id(j.job)], 4) for j in existing_jobs]}")
    print(f"    weighted tardiness existants: {[round(all_weights[id(j.job)] * j.delay, 4) for j in existing_jobs]}")
    print(f"    cost_existants={cost_existants:.4f}")

    print(f"    tardiness nouveaux: {[j.delay for j in new_jobs_states]}")
    print(f"    weights nouveaux: {[round(all_weights[id(j.job)], 4) for j in new_jobs_states]}")
    print(f"    weighted tardiness nouveaux: {[round(all_weights[id(j.job)] * j.delay, 4) for j in new_jobs_states]}")
    print(f"    cost_nouveaux={cost_nouveaux:.4f}")

    return cost_existants, cost_nouveaux, total_cost, env.state.cmax

# Recherche en largeur des nouveaux jobs dans l'arbre
def bfs_forward(state: State, new_jobs: list[Job], cut_time: int, agent: Agent, device: str, all_weights: dict, delta_ratio: float = 0.2) -> list[Job]:
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
            #f"dd={j.job.due_date} | "
            f"weight={all_weights[id(j.job)]:.4f}"
        )

    print("  Nouveaux jobs:")
    for i, job in enumerate(new_jobs):
        print(
            f"    New J{i + 1} | "
            #f"dd={job.due_date} | "
            f"weight={all_weights[id(job)]:.4f}"
        )

    cost_ref, cmax_ref = compute_current_schedule_ref(
        state,
        all_weights
    )

    if cost_ref == 0:
        cost_max = cmax_ref * delta_ratio
    else:
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
            all_weights
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


def acceptation_method(order_instance: OrderInstance, agent: Agent, device: str, delta_ratio: float = 0.2) -> State:
    """
    Pipeline complet : schedule Order 1 puis applique le Master Problem pour chaque order suivant.
    """
    # 1. Scheduler le premier order normalement
    first_order = order_instance.orders[0]
    instance    = first_order.to_instance()
    state       = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
    graph       = state.to_hyper_graph_costs(last_job_in_pos=-1, current_time=0, device=device)
    env         = Environment(graph=graph, state=state, n=len(instance.jobs))
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    all_weights = {}

    # Temps total de toute la méthode d'acceptation
    global_start_time = time.perf_counter()

    while env.possible_decisions:
        action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)

    print(f"Order 1 schedulé | Cmax={env.state.cmax} | Tardiness={sum(j.delay for j in env.state.job_states)}")
    first_order = order_instance.orders[0]
    for job in first_order.jobs:
        #all_weights[id(job)] = random.uniform(EXISTING_COST_MIN, EXISTING_COST_MAX)
        all_weights[id(job)] = int(getattr(job, "cost", 1))

    # 2. Pour chaque order suivant → Master Problem
    for order in order_instance.orders[1:]:
        cut_time  = order.cut_time
        new_jobs  = order.jobs

        for job in new_jobs:
            if id(job) not in all_weights:
                #all_weights[id(job)] = random.uniform(NEW_COST_MIN, NEW_COST_MAX)
                all_weights[id(job)] = int(getattr(job, "cost", 1))

        # Gantt avant le cut
        #gnn_gantt(f"data/gantts/test/same_costs/before_cut_{order.id}.png", env.state, f"before cut {order.id}", cut_times=[cut_time])
        #gnn_gantt(f"data/gantts/test/different_costs/before_cut_{order.id}.png", env.state, f"before cut {order.id}", cut_times=[cut_time])
        #gnn_gantt(f"data/gantts/test/E_high_N_low_ddLow/before_cut_{order.id}.png", env.state, f"before cut {order.id}", cut_times=[cut_time])
        gnn_gantt(f"data/gantts/test/E_high_N_high_ddLow/before_cut_{order.id}.png", env.state, f"before cut {order.id}", cut_times=[cut_time])


        print(f"\n=== Order {order.id} | cut_time={cut_time} | {len(new_jobs)} nouveaux jobs ===")

        # 3. Master Problem → choisir le meilleur sous-ensemble
        best_subset = bfs_forward(env.state, new_jobs, cut_time, agent, device, all_weights, delta_ratio)
        
        print(f"\n  === Résultat Order {order.id} ===")
        print(f"  Jobs acceptés  : {[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in best_subset]}")
        print(f"  Jobs rejetés   : {[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in new_jobs if j not in best_subset]}")
        print(f"  {len(best_subset)}/{len(new_jobs)} jobs acceptés")


        # 4. Reschedule avec le meilleur sous-ensemble
        if len(best_subset) == 0:
            print(f"  Aucun job accepté → planning original conservé")
            continue

        cut_state = build_state_from_cut(env.state, cut_time)
        #cut_state.display_calendars()
        #cut_state.all_stations.stations[1].calendar.display_calendar("STATION 2")
        #gnn_gantt(f"data/gantts/after_cut_{order.id}.png", env.state, f"after cut {order.id}", cut_times=[cut_time])
        #cut_state.robot.calendar.display_calendar("ROBOT après cut 3")
        #cut_state.get_job_by_id(3).calendar.display_calendar("JOB 4 après cut 3")
        #ut_state.all_stations.stations[2].calendar.display_calendar("STATION 3 après cut 3")
        #j4 = cut_state.get_job_by_id(3)
        #print(f"J4 status={j4.status}, location={j4.location}, end={j4.end}")
        #cut_state.machine2.calendar.display_calendar("MACHINE 2 après cut 3")
        """print(f"J4 current_station={cut_state.job_states[3].current_station}")
        print(f"\n=== Calendriers Stations après cut={cut_time} ===")
        for i, station in enumerate(cut_state.all_stations.stations):
            print(f"Station {i+1} | free_at={station.free_at} | current_job={station.current_job.id+1 if station.current_job else None}")
            for e in station.calendar.events:
                print(f"  start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}, job=J{e.job.id+1 if e.job else None}")
        print(f"J4 o1.is_last = {cut_state.job_states[3].operation_states[0].is_last}")
        print(f"\n=== Calendriers après cut={cut_time} ===")
        print(f"Robot free_at={cut_state.robot.free_at}, location={cut_state.robot.location}")
        print(f"M1 free_at={cut_state.machine1.free_at}")
        print(f"M2 free_at={cut_state.machine2.free_at}")"""
        #for j in cut_state.job_states:
            #print(f"Job {j.id+1} | status={j.status} | location={j.location} | ops={[(o.status, o.remaining_time) for o in j.operation_states]}")
        
        
        """print(f"\n=== Calendrier J4 cut={cut_time} ===")
        for e in cut_state.job_states[3].calendar.events:
            print(f"start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}, op={e.operation.id if e.operation else None}")"""

        #print(f"M1 free_at={cut_state.machine1.free_at}")
        #print(f"M2 free_at={cut_state.machine2.free_at}")
        #print(f"J2 status={cut_state.job_states[1].status} | location={cut_state.job_states[1].location}")
        #print(f"J2 ops={[(o.status, o.remaining_time) for o in cut_state.job_states[1].operation_states]}")
        #for j in cut_state.job_states:
            #print(f"Job {j.id+1} | status={j.status} | location={j.location} | ops={[(o.status, o.remaining_time) for o in j.operation_states]}")
        cut_state.add_jobs_to_state(best_subset)
        graph     = cut_state.to_hyper_graph_costs(last_job_in_pos=-1, current_time=cut_time, device=device)
        env       = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states), action_time=cut_time)
        env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)

        while env.possible_decisions:
            action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
            env = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)

        #print(f"\n=== État J4 après scheduling Order {order.id} ===")
        #print(f"J4 status={env.state.job_states[3].status}")
        #print(f"J4 ops={[(o.status, o.end, o.remaining_time) for o in env.state.job_states[3].operation_states]}")
        #for e in env.state.job_states[3].calendar.events:
            #print(f"  start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}")


        print(f"Order {order.id} schedulé | Cmax={env.state.cmax} | Tardiness={sum(j.delay for j in env.state.job_states)}")
    global_end_time = time.perf_counter()
    total_acceptance_time = global_end_time - global_start_time

    print("\n=== Statistiques globales méthode d'acceptation ===")
    print(f"  Temps computationnel total : {total_acceptance_time:.4f} secondes")
    print(f"  Nombre d'orders            : {len(order_instance.orders)}")
    print(f"  Nombre total de jobs       : {order_instance.nb_jobs}")

    return env.state