import torch

from models.order import OrderInstance
from models.agent import Agent
from acceptation_method import acceptation_method
from gantt_builder.gnn_gantt import gnn_gantt
from conf import *

#order_instance = OrderInstance.load("data/orders_instances/test/s/instance_7.json")
#order_instance = OrderInstance.load("data/instances_cost/same_costs/instance_3.json")
order_instance = OrderInstance.load("data/instances_cost/different_costs/instance_4.json")
#order_instance = OrderInstance.load("data/instances_cost/different_near_costs/instance_3.json")
#order_instance = OrderInstance.load("data/instances_cost/E_high_N_low_ddLow/instance_3.json")
#order_instance = OrderInstance.load("data/instances_cost/E_high_N_high_ddLow/instance_4.json")

device: str      = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
agent          = Agent(device=device, interactive=False, load=True, path="data/training_costs/", train=False, custom=True)

final_state = acceptation_method(order_instance, agent, device=device, delta_ratio=0.0)

print(f"\n=== Résultat final ===")
print(f"Cmax={final_state.cmax} | Delay={sum(j.delay for j in final_state.job_states)}")

cut_times = [o.cut_time for o in order_instance.orders if o.cut_time > 0]
#gnn_gantt("data/gantts/test/same_costs/acceptation.png", final_state, "acceptation method", cut_times=cut_times)
gnn_gantt("data/gantts/test/different_costs/acceptation.png", final_state, "acceptation method", cut_times=cut_times)
#gnn_gantt("data/gantts/test/different_near_costs/acceptation.png", final_state, "acceptation method", cut_times=cut_times)
#gnn_gantt("data/gantts/test/E_high_N_low_ddLow/acceptation.png", final_state, "acceptation method", cut_times=cut_times)
#gnn_gantt("data/gantts/test/E_high_N_high_ddLow/acceptation.png", final_state, "acceptation method", cut_times=cut_times)