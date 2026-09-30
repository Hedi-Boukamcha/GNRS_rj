# heuristic/local_search.py
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

# CHANGEMENT (reward sans Cmax) : option delay_first. Par défaut (False) le critère reste retard + Cmax,
# donc l'heuristique de référence et la recherche tabou ne changent pas ; le solveur GNN passe
# delay_first=True pour juger au retard, avec le Cmax seulement pour départager.
# def ls(instance: Instance, decisions: list[Decision]):  # OLD
def ls(instance: Instance, decisions: list[Decision], delay_first: bool = False):  # NEW
    state, obj   = _simulate_one(instance, decisions, delay_first)
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
            new_state, new_obj = _simulate_one(instance, to_test, delay_first)
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
                new_state, new_obj = _simulate_one(instance, to_test, delay_first)
                if new_obj <= obj:
                    print("LOCAL SEARCH found a better solution: case 2 (add more parallel)...")
                    decisions = to_test
                    state     = new_state
                    obj       = new_obj
                    idx      += 1    
        else:
            idx += 1
    return state

# def _simulate_one(instance: Instance, decisions: list[Decision]) -> Tuple[State, int]:  # OLD
def _simulate_one(instance: Instance, decisions: list[Decision], delay_first: bool = False):  # NEW
    state: State = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
    for d in decisions:
        state = simulate(state, d=d, clone=False) 
    obj: int = state.total_delay + state.cmax
    if delay_first:  # NEW : critère lexicographique (retard, Cmax) ; les comparaisons "<=" marchent sur les tuples
        obj = (state.total_delay, state.cmax)  # NEW
    return state, obj