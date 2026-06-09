from models.order import OrderInstance
from models.state import State, JobState
from models.instance import Job
from models.agent import Agent
from simulators.gnn_simulator import build_state_from_cut
from models.environment import Environment
from gnn_solver import search_possible_decisions, take_one_step
from conf import *


# ############################################
# =*= THE ACCEPTATION METHOD FOR NEW JOBS  =*=
# ############################################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0" 
__license__ = "MIT"


# Definition de la valeur du retard actuel (de reference)
def compute_tardiness_ref(state: State, cut_time: int, agent: Agent, device: str) -> int:
    """
    Reschedule le pool existant seul sans nouveaux jobs.
    Retourne le total_delay de référence.
    """
    # 1. Reconstruire l'état au cut_time
    cut_state = build_state_from_cut(state, cut_time)
    
    # 2. GNN reschedule sans nouveaux jobs
    graph = cut_state.to_hyper_graph(last_job_in_pos=-1, current_time=cut_time, device=device)
    env   = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states), action_time=cut_time)
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    
    while env.possible_decisions:
        action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)
    
    # 3. Calculer le total_delay des jobs existants
    tardiness_ref = sum(j.delay for j in env.state.job_states)
    return tardiness_ref, env.state.cmax

# Evaluation des nouveaux jobs (sous ensembles): qq soit un seul job ou bien une combinaison de plusieurs jobs
def evaluate_subset(state: State, subset: list[Job], cut_time: int, nb_existing: int, agent: Agent, device: str) -> tuple[int, int]:
    """
    Reschedule le pool existant + le sous-ensemble S.
    Retourne (total_delay_existants, cmax).
    """
    # 1. Reconstruire l'état au cut_time
    cut_state = build_state_from_cut(state, cut_time)
    
    # 2. Ajouter seulement les jobs du sous-ensemble
    cut_state.add_jobs_to_state(subset)
    
    # 3. GNN reschedule
    graph = cut_state.to_hyper_graph(last_job_in_pos=-1, current_time=cut_time, device=device)
    env   = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states), action_time=cut_time)
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)
    
    while env.possible_decisions:
        action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)
    
    # 4. Calculer le total_tardiness des jobs EXISTANTS seulement (pas les nouveaux)
    nb_existing = len(state.job_states)
    tardiness_existants = sum(j.delay for j in env.state.job_states[:nb_existing])
    print(f"    delays existants: {[j.delay for j in env.state.job_states[:nb_existing]]}")
    print(f"    delays nouveaux: {[j.delay for j in env.state.job_states[nb_existing:]]}")
    
    return tardiness_existants, env.state.cmax

# Recherche en largeur des nouveaux jobs dans l'arbre
def bfs_forward(state: State, new_jobs: list[Job], cut_time: int, agent: Agent, device: str, delta_ratio: float = 0.2) -> list[Job]:
    """
    BFS Forward : explore les sous-ensembles de new_jobs à accepter.
    Retourne le sous-ensemble avec le max de jobs qui respecte le retard de ref.
    """
    nb_existing      = len(state.job_states)
    
    # 1. Calculer δ_ref et δ_max
    tardiness_ref, cmax_ref = compute_tardiness_ref(state, cut_time, agent, device)
    # Si δ_ref=0, utiliser un seuil absolu basé sur le Cmax
    if tardiness_ref == 0:
        tardiness_max = cmax_ref * delta_ratio  # ex: 20% du Cmax comme tolérance
    else:
        tardiness_max = tardiness_ref * (1 + delta_ratio)
    print(f"  δ_ref={tardiness_ref} | δ_max={tardiness_max:.1f}")

    # 2. BFS : file de sous-ensembles à explorer
    best_subset  = []          # meilleur sous-ensemble trouvé
    best_cmax   = float('inf')
    visited      = set()       # sous-ensembles déjà évalués (déduplication)
    queue        = [[]]        # on commence par ∅

    while queue:
        current_subset = queue.pop(0)  # ← garder seulement celui-ci

        # Déduplication
        key = frozenset(id(j) for j in current_subset)
        if key in visited:
            continue
        visited.add(key)

        print(f"\n  → Subset={[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in current_subset]} | size={len(current_subset)} | queue={len(queue)}")

        # Évaluer le sous-ensemble courant
        tardiness_existants, cmax = evaluate_subset(state, current_subset, cut_time, nb_existing, agent, device)
        print(f"     tardiness_existants={tardiness_existants} | δ_max={tardiness_max:.1f} | cmax={cmax}")


        if tardiness_existants <= tardiness_max:
            print(f"     ✅ Valide → extension")
            if len(current_subset) > len(best_subset):
                best_subset = current_subset
                best_cmax   = cmax
            elif len(current_subset) == len(best_subset) and cmax < best_cmax:
                best_subset = current_subset
                best_cmax   = cmax
            # Ajouter les extensions
            for job in new_jobs:
                if job not in current_subset:
                    queue.append(current_subset + [job])
        # Sinon → élaguer
        else:
            print(f"     ❌ Élagué (tardiness={tardiness_existants} > δ_max={tardiness_max:.1f})")

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
    graph       = state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device=device)
    env         = Environment(graph=graph, state=state, n=len(instance.jobs))
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)

    while env.possible_decisions:
        action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)

    print(f"Order 1 schedulé | Cmax={env.state.cmax} | Tardiness={sum(j.delay for j in env.state.job_states)}")

    # 2. Pour chaque order suivant → Master Problem
    for order in order_instance.orders[1:]:
        cut_time  = order.cut_time
        new_jobs  = order.jobs

        print(f"\n=== Order {order.id} | cut_time={cut_time} | {len(new_jobs)} nouveaux jobs ===")

        # 3. Master Problem → choisir le meilleur sous-ensemble
        best_subset = bfs_forward(env.state, new_jobs, cut_time, agent, device, delta_ratio)
        
        print(f"\n  === Résultat Order {order.id} ===")
        print(f"  Jobs acceptés  : {[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in best_subset]}")
        print(f"  Jobs rejetés   : {[f'J{new_jobs.index(j)+1}(dd={j.due_date})' for j in new_jobs if j not in best_subset]}")
        print(f"  {len(best_subset)}/{len(new_jobs)} jobs acceptés")

        # 4. Reschedule avec le meilleur sous-ensemble
        cut_state = build_state_from_cut(env.state, cut_time)
        cut_state.add_jobs_to_state(best_subset)
        graph     = cut_state.to_hyper_graph(last_job_in_pos=-1, current_time=cut_time, device=device)
        env       = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states), action_time=cut_time)
        env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device=device)

        while env.possible_decisions:
            action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
            env = take_one_step(agent=agent, last_env=env, action_id=action_id, device=device)

        print(f"Order {order.id} schedulé | Cmax={env.state.cmax} | Tardiness={sum(j.delay for j in env.state.job_states)}")

    return env.state