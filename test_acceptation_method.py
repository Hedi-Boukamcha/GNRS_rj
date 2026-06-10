from models.order import OrderInstance
from models.agent import Agent
from acceptation_method import acceptation_method
from gantt_builder.gnn_gantt import gnn_gantt
from conf import *

order_instance = OrderInstance.load("data/orders_instances/test/s/instance_17.json")
agent          = Agent(device="cpu", interactive=False, load=True, path="data/training/", train=False, custom=True)

final_state = acceptation_method(order_instance, agent, device="cpu", delta_ratio=0.0)

print(f"\n=== Résultat final ===")
print(f"Cmax={final_state.cmax} | Delay={sum(j.delay for j in final_state.job_states)}")

cut_times = [o.cut_time for o in order_instance.orders if o.cut_time > 0]
gnn_gantt("data/gantts/acceptation.png", final_state, "acceptation method", cut_times=cut_times)