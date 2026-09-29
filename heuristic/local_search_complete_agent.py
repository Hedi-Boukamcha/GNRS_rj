# heuristic/local_search_complete_agent.py
# Copie de heuristic/local_search.py + variante ls_from_cut() pour la méthode d'acceptation
# (complete agent). La fonction ls() d'origine est conservée telle quelle.
from models.state import State, Decision
from models.instance import Instance
from conf import *
from simulators.gnn_simulator import simulate
from typing import Tuple

# ##################################
# =*= LOCAL IMPROVEMENT OPERATOR =*=
# ##################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0"
__license__ = "MIT"

def ls(instance: Instance, decisions: list[Decision]):
    state, obj   = _simulate_one(instance, decisions)
    idx: int     = 0
    last_M1: int = -1
    while idx < len(decisions):
        d: Decision = decisions[idx].clone()
        to_test: list[Decision] = decisions[:idx] + [d] + decisions[idx+1:]
        if d.machine == MACHINE_1:
            last_M1 = d.job_id
        if d.parallel == True: # case 1: maybe the parallel decision was a mistake?
            d.parallel = False
            idx += 1
            while idx < len(decisions) and to_test[idx].machine == MACHINE_2 and to_test[idx].parallel == True:
                d_next: Decision = to_test[idx].clone()
                d_next.parallel  = False
                d_next.comp      = -1
                to_test[idx]     = d_next
                idx              += 1
            new_state, new_obj = _simulate_one(instance, to_test)
            if new_obj <= obj:
                print("LOCAL SEARCH found a better solution: case 1 (remove useless parallel)...")
                decisions = to_test
                state     = new_state
                obj       = new_obj
        elif d.parallel == False and idx < len(decisions) -1: # case 2: maybe i should put in parallel?
            d.parallel  = True
            idx        += 1
            idz: int    = idx
            while idz < len(decisions) and to_test[idz].machine == MACHINE_2 and to_test[idz].parallel == False:
                d_next: Decision = to_test[idz].clone()
                d_next.parallel  = True
                d_next.comp      = last_M1
                to_test[idz]     = d_next
                idz += 1
                new_state, new_obj = _simulate_one(instance, to_test)
                if new_obj <= obj:
                    print("LOCAL SEARCH found a better solution: case 2 (add more parallel)...")
                    decisions = to_test
                    state     = new_state
                    obj       = new_obj
                    idx      += 1    
        else:
            idx += 1
    return state

def _simulate_one(instance: Instance, decisions: list[Decision]) -> Tuple[State, int]:
    state: State = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    for d in decisions:
        state = simulate(state, d=d, clone=False) 
    obj: int = state.total_delay + state.cmax
    return state, obj

# ##########################################################################
# =*= LOCAL IMPROVEMENT OPERATOR FROM A CUT STATE (acceptation method)  =*=
# ##########################################################################
# Différences avec ls() :
#   1. On ne repart pas d'une instance neuve au temps 0 : on rejoue seulement les décisions
#      prises APRÈS le cut, à partir d'un clone de l'état au cut. La partie déjà exécutée
#      (avant le cut) n'est donc jamais modifiée.
#   2. Chaque décision rejouée doit exister dans les décisions possibles de l'environnement
#      (même job, même opération, même flag parallel). Une modification infaisable est rejetée
#      au lieu de faire planter la simulation. Le champ comp est recalculé par l'environnement.
#   3. Le critère est celui de la méthode d'acceptation : (retard pondéré total, cmax).

def weighted_key(state: State) -> tuple[float, int]:
    return (sum(float(js.job.cost) * js.delay for js in state.job_states), state.cmax)

def _replay_step(env, d: Decision):
    # Miroir de gnn_solver.take_one_step (clone=False, train=False), sans construire le graphe.
    from gnn_solver import search_possible_decisions
    if d.parallel:
        if env.state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == MACHINE_1:
            env.last_job_in_pos  = d.job_id
            env.next_M2_parallel = True
        else:
            env.total_m2_parallel += 1
            env.m2                += 1
            env.next_M2_parallel   = False
    else:
        env.next_M2_parallel = False
        env.last_job_in_pos  = -1
        if env.state.get_job_by_id(d.job_id).operation_states[d.operation_id].operation.type == MACHINE_2:
            env.m2 += 1
    env.state = simulate(env.state, d=d, clone=False)
    if d.parallel and d.machine == MACHINE_2 and env.last_job_in_pos >= 0:
        m1_job = env.state.get_job_by_id(env.last_job_in_pos)
        if m1_job is not None and m1_job.is_done():
            env.last_job_in_pos  = -1
            env.next_M2_parallel = False
    env.action_time = env.state.min_action_time()
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device="cpu")

def replay_from_cut(start_state: State, start_time: int, n: int, suffix: list[Decision]):
    """Rejoue suffix depuis un clone de start_state. Retourne l'Environment final ou None si infaisable/incomplet."""
    from gnn_solver import search_possible_decisions
    from models.environment import Environment
    env = Environment(graph=None, state=start_state.clone(), n=n, action_time=start_time)
    try:
        env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device="cpu")
        for wanted in suffix:
            match = next((p for p in env.possible_decisions
                          if p.job_id == wanted.job_id
                          and p.operation_id == wanted.operation_id
                          and bool(p.parallel) == bool(wanted.parallel)), None)
            if match is None:
                return None
            _replay_step(env, match)
    except RuntimeError:
        return None
    if env.possible_decisions:  # des opérations restent à céduler -> solution incomplète
        return None
    return env

def ls_from_cut(start_state: State, start_time: int, n: int, suffix: list[Decision], verbose: bool = False):
    """
    Recherche locale (mêmes deux mouvements que ls) sur les décisions postérieures au cut.
    start_state : état au cut, avec les nouveaux jobs ajoutés et compute_obj_values_and_upper_bounds
                  déjà appelé, AVANT toute décision du rollout.
    suffix      : décisions prises par le rollout à partir de start_state.
    Retourne (env, key) de la meilleure solution trouvée, ou (None, None) si le rejeu de départ échoue.
    """
    best_env = replay_from_cut(start_state, start_time, n, suffix)
    if best_env is None:
        return None, None
    best_key  = weighted_key(best_env.state)
    decisions = [d.clone() for d in suffix]
    idx: int     = 0
    last_M1: int = -1
    while idx < len(decisions):
        d: Decision = decisions[idx].clone()
        to_test: list[Decision] = decisions[:idx] + [d] + decisions[idx+1:]
        if d.machine == MACHINE_1:
            last_M1 = d.job_id
        if d.parallel == True: # case 1: maybe the parallel decision was a mistake?
            d.parallel = False
            idx += 1
            while idx < len(decisions) and to_test[idx].machine == MACHINE_2 and to_test[idx].parallel == True:
                d_next: Decision = to_test[idx].clone()
                d_next.parallel  = False
                d_next.comp      = -1
                to_test[idx]     = d_next
                idx              += 1
            new_env = replay_from_cut(start_state, start_time, n, to_test)
            if new_env is not None and weighted_key(new_env.state) <= best_key:
                if verbose: print("LOCAL SEARCH (cut) found a better solution: case 1 (remove useless parallel)...")
                decisions = to_test
                best_env  = new_env
                best_key  = weighted_key(new_env.state)
        elif d.parallel == False and idx < len(decisions) -1: # case 2: maybe i should put in parallel?
            d.parallel  = True
            idx        += 1
            idz: int    = idx
            while idz < len(decisions) and to_test[idz].machine == MACHINE_2 and to_test[idz].parallel == False:
                d_next: Decision = to_test[idz].clone()
                d_next.parallel  = True
                d_next.comp      = last_M1
                to_test[idz]     = d_next
                idz += 1
                new_env = replay_from_cut(start_state, start_time, n, to_test)
                if new_env is not None and weighted_key(new_env.state) <= best_key:
                    if verbose: print("LOCAL SEARCH (cut) found a better solution: case 2 (add more parallel)...")
                    decisions = to_test
                    best_env  = new_env
                    best_key  = weighted_key(new_env.state)
                    idx      += 1
        else:
            idx += 1
    return best_env, best_key
