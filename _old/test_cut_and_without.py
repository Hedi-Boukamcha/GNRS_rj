from models.order import Order, OrderInstance
from models.state import State
from simulators.gnn_simulator import simulate, build_state_from_cut
from models.agent import Agent
from models.environment import Environment
from _old.gnn_solver import search_possible_decisions, take_one_step
from models.instance import Instance
from gantt_builder.gnn_gantt import gnn_gantt
from conf import *


# ##########################################
# =*= TEST THE IMPACT OF USING CUT_TIME  =*=
# ##########################################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__ = "2.0.0" 
__license__ = "MIT"


order_instance = OrderInstance.load("data/orders_instances/test/s/instance_7.json")
agent = Agent(device="cpu", interactive=False, load=True, path="data/training/", train=False, custom=True)

# ===== CAS 1 : AVEC CUT =====
first_order = order_instance.orders[0]
instance    = first_order.to_instance()
state       = State(instance, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
state.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
graph       = state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device="cpu")
env         = Environment(graph=graph, state=state, n=len(instance.jobs))
env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device="cpu")

while env.possible_decisions:
    action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
    env = take_one_step(agent=agent, last_env=env, action_id=action_id, device="cpu")

for order in order_instance.orders[1:]:
    cut_time  = order.cut_time
    cut_state = build_state_from_cut(env.state, cut_time)
    j1 = cut_state.get_job_by_id(0)
    #print(f"J1 status={j1.status}, location={j1.location}, ops={[(o.status, o.remaining_time) for o in j1.operation_states]}")
    cut_state.add_jobs_to_state(order.jobs)
    graph = cut_state.to_hyper_graph(last_job_in_pos=-1, current_time=cut_time, device="cpu")
    env   = Environment(graph=graph, state=cut_state, n=len(cut_state.job_states), action_time=cut_time)
    env.possible_decisions, env.decisionsT = search_possible_decisions(env=env, device="cpu")
    while env.possible_decisions:
        action_id = agent.select_next_decision(graph=env.graph, decisionsT=env.decisionsT, greedy=True)
        env = take_one_step(agent=agent, last_env=env, action_id=action_id, device="cpu")

cut_times = [o.cut_time for o in order_instance.orders if o.cut_time > 0]
total_delay = sum(j.delay for j in env.state.job_states)
print(f"Avec cut → Cmax={env.state.cmax} | Total Delay={total_delay}")
gnn_gantt("data/gantts/avec_cut.png", env.state, "avec cut", cut_times=cut_times)

# Afficher tous les calendriers
env.state.display_calendars()


# ===== CAS 2 : SANS CUT =====
all_jobs = [j for o in order_instance.orders for j in o.jobs]
instance2 = Instance(jobs=all_jobs, n=sum(len(j.operations) for j in all_jobs))
state2    = State(instance2, M, L, NB_STATIONS, BIG_STATION, [], automatic_build=True)
state2.compute_obj_values_and_upper_bounds(unloading_time=0, current_time=0)
graph2    = state2.to_hyper_graph(last_job_in_pos=-1, current_time=0, device="cpu")
env2      = Environment(graph=graph2, state=state2, n=len(all_jobs))
env2.possible_decisions, env2.decisionsT = search_possible_decisions(env=env2, device="cpu")

while env2.possible_decisions:
    action_id = agent.select_next_decision(graph=env2.graph, decisionsT=env2.decisionsT, greedy=True)
    env2 = take_one_step(agent=agent, last_env=env2, action_id=action_id, device="cpu")

total_delay2 = sum(j.delay for j in env2.state.job_states)
print(f"Sans cut → Cmax={env2.state.cmax} | Total Delay={total_delay2}")
gnn_gantt("data/gantts/sans_cut.png", env2.state, "sans cut", cut_times=[])
env2.state.display_calendars()