from conf import *
from models.instance import Operation, Job, Instance
import json

# ##########################################
# =*= ORDER: INSTANCE STRUCTURE AND LOAD =*=
# ##########################################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__  = "1.0.0"
__license__ = "MIT"

class Order:
    def __init__(self, id: int = 0, cut_time: int = 0, jobs: list[Job] = []):
        self.id       : int       = id
        self.cut_time : int       = cut_time
        self.jobs     : list[Job] = jobs
        self.n        : int       = sum(len(j.operations) for j in jobs)

    def to_instance(self) -> Instance:
        """Convert a single order to an Instance for the GNN solver."""
        return Instance(jobs=self.jobs, n=self.n)

    def __str__(self) -> str:
        return f"{{'id':{self.id}, 'cut_time':{self.cut_time}, 'nb_jobs':{len(self.jobs)}}}"

class OrderInstance:
    def __init__(self, orders: list[Order] = [], a: int = 0):
        self.orders  : list[Order] = orders
        self.a       : int         = a
        self.nb_jobs : int         = sum(len(o.jobs) for o in orders)

    def __str__(self) -> str:
        return f"{[o.__str__() for o in self.orders]}"

    @staticmethod
    def load(path: str) -> 'OrderInstance':
        with open(path, 'r') as f:
            _data = json.load(f)
        orders = []
        for order_data in _data["orders"]:
            jobs = []
            for job_data in order_data["jobs"]:
                operations = [Operation(type=op["type"], machineing_time=op["processing_time"]) for op in job_data["operations"]]
                job = Job(
                    big          = job_data["big"],
                    due_date     = job_data["due_date"],
                    release_date = job_data["release_date"],
                    pos_time     = job_data["pos_time"],
                    operations   = operations,
                    status       = job_data["status"],
                    blocked      = job_data["blocked"],
                    cost         = job_data["cost"]
                )
                jobs.append(job)
            orders.append(Order(
                id       = order_data["id"],
                cut_time = order_data["cut_time"],
                jobs     = jobs
            ))
        return OrderInstance(orders=orders, a=_data.get("a", 0))

    def display(self):
        for o in self.orders:
            print(f"Order {o.id} | cut_time={o.cut_time} | nb_jobs={len(o.jobs)}")
            for j in o.jobs:
                print(f"  {j}")