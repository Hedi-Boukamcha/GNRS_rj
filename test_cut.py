import random
from models.instance import Instance
from models.state import State
from simulators.gnn_simulator import simulate, build_state_from_cut
from gnn_solver import search_possible_decisions as spd
from models.agent import Agent
from conf import *
from torch_geometric.data import HeteroData
import torch

# 1. Charger une instance simple
instance = Instance.load("data/instances/test/s/instance_1.json")

# 2. Builder le state initial
state = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)

# 3. Simuler quelques décisions avec un agent random
agent = Agent(device="cpu", interactive=False, load=False, path="data/training/", train=False, custom=True)
graph = state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device="cpu")

from models.environment import Environment
env = Environment(graph=graph, state=state, n=len(instance.jobs))

from gnn_solver import search_possible_decisions as spd
env.possible_decisions, env.decisionsT = spd(env=env, device="cpu")

# Avancer de quelques décisions
nb_steps = 3
for _ in range(nb_steps):
    if not env.possible_decisions:
        break
    action_id = random.randint(0, len(env.possible_decisions) - 1)
    from gnn_solver import take_one_step
    env = take_one_step(agent=agent, last_env=env, action_id=action_id, device="cpu")

print(f"\n=== Etat après {nb_steps} décisions ===")
for j in env.state.job_states:
    print(f"Job {j.id+1} | status={j.status} | location={j.location} | nb_events={len(j.calendar.events)}")

# 4. Appliquer build_state_from_cut
cut_time = env.state.robot.free_at // 2
print(f"\n=== Cut à t={cut_time} ===")
print("\n=== Calendrier J4 ===")
for e in env.state.job_states[3].calendar.events:
    print(e)
print(f"\ncut_time = {cut_time}")
cut_state = build_state_from_cut(env.state, cut_time)
print("\n=== Calendrier J1 après clone ===")
for e in cut_state.job_states[0].calendar.events:
    print(f"start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}")
print(f"\n=== Calendrier J2 après clone ===")
for e in cut_state.job_states[1].calendar.events:
    print(f"start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}")
print("\n=== Calendrier J5 après clone ===")
for e in cut_state.job_states[4].calendar.events:
    print(f"start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}")
print(f"\n=== Etat après cut ===")
print(f"Robot free_at={cut_state.robot.free_at}")
print(f"Machine1 free_at={cut_state.machine1.free_at}")
print(f"Machine2 free_at={cut_state.machine2.free_at}")
for j in cut_state.job_states:
    print(f"Job {j.id+1} | status={j.status} | remaining_ops={sum(o.remaining_time for o in j.operation_states)}")

# TEST WITH : python test_cut.py
