from os import mkdir

from gantt_builder.gnn_gantt import gnn_gantt
from models.order import Order, OrderInstance
from models.state import State
from simulators.gnn_simulator import simulate, build_state_from_cut
from models.agent import Agent
from models.environment import Environment
from gnn_solver import search_possible_decisions, take_one_step
from conf import *

# #######################################################
# =*= TEST THE RESCHEDULING OF JOBS AFTER A CUT_TIME  =*=
# #######################################################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0" 
__license__ = "MIT"


# 1. Charger une instance de commande
order_instance = OrderInstance.load("data/orders_instances/test/s/instance_17.json")
order_instance.display()

# 2. Scheduler la première commande (cut_time=0, commande initiale a t=0)
agent      = Agent(device="cpu", interactive=False, load=True, path="data/training/", train=False, custom=True)
first_order = order_instance.orders[0]
instance   = first_order.to_instance()
state      = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
graph      = state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device="cpu")
env        = Environment(graph=graph, state=state, n=len(instance.jobs))
env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device="cpu")

while env.possible_decisions:
    action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
    env = take_one_step(agent=agent, last_env=env, action_id=action_id, device="cpu")

print(f"\n=== Order 1 schedulé | Cmax={env.state.cmax} ===")
for j in env.state.job_states:
    print(f"Job {j.id+1} | status={j.status} | release_date={j.job.release_date}")

# 3. Pour chaque ordre suivant
for order in order_instance.orders[1:]:
    cut_time = order.cut_time
    print(f"\n=== Cut à t={cut_time} | Order {order.id} arrive ===")

    # 4. Reconstruire l'état au cut_time
    cut_state = build_state_from_cut(env.state, cut_time)
    print(f"Robot free_at={cut_state.robot.free_at}")
    for j in cut_state.job_states:
        print(f"Job {j.id+1} | status={j.status} | remaining={sum(o.remaining_time for o in j.operation_states)}")

    # 5. Ajouter les nouveaux jobs
    cut_state.add_jobs_to_state(order.jobs)
    print(f"\n=== Après ajout des {len(order.jobs)} nouveaux jobs ===")
    for j in cut_state.job_states:
        print(f"Job {j.id+1} | status={j.status} | release_date={j.job.release_date}")

    # 6. GNN reschedule
    graph = cut_state.to_hyper_graph(last_job_in_pos=-1, current_time=cut_time, device="cpu")
    env   = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states))
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device="cpu")

    while env.possible_decisions:
        action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, device="cpu")

    print(f"\n=== Order {order.id} schedulé | Cmax={env.state.cmax} ===")

cut_times = [o.cut_time for o in order_instance.orders if o.cut_time > 0]
gnn_gantt("data/gantts/test_order_pipeline.png", env.state, "order pipeline test", cut_times=cut_times)

# TEST WITH : python test_order_pipeline.py
