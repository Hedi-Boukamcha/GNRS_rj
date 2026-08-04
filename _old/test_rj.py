# test_rj.py
from models.instance import Instance
from models.state import State
from conf import *

# Charger une instance générée
instance = Instance.load("data/instances/train/s/instance_1.json")

# Afficher les release dates
for j in instance.jobs:
    print(f"Job {j} | release_date={j.release_date} | due_date={j.due_date}")

# Construire le state et le graphe
state = State(i=instance, M=M, L=L, nb_stations=NB_STATIONS, station_large=BIG_STATION)
graph = state.to_hyper_graph(last_job_in_pos=-1, current_time=0, device="cpu")
print(f"\nJob features shape: {graph['job'].x.shape}")  # (n_jobs, 14)