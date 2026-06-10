import json
import os
import random
import argparse

from conf import ORDERS_SIZES, M, L

# #############################
# =*= ORDERS GENERATOR =*=
# #############################
__author__  = "Hedi Boukamcha"
__email__   = "hedi.boukamcha.1@ulaval.ca"
__version__  = "1.0.0"
__license__ = "MIT"

def generate_order(
        numbre_jobs: int = 3,
        max_operations_per_job: int = 2,
        types_operations: list = [0, 1],
        min_duration: int = 10,
        max_duration: int = 60,
        pos_time: int = 5,
        due_date_min: int = 50,
        cut_time: int = 0,
    ):
    """Generate one order (= one batch of jobs arriving at cut_time)."""

    jobs = []
    for _ in range(numbre_jobs):
        nb_operations = random.randint(1, max_operations_per_job)
        last_type     = None
        due_date_max  = (2 * L + pos_time) * numbre_jobs + nb_operations * (2 * M + 60)
        borne_inf     = max(due_date_min, due_date_max // 3)
        due_date      = random.randint(borne_inf, due_date_max)

        # Shift due_date to be >= cut_time
        due_date = max(due_date, cut_time + due_date_min)

        job = {
            "big"      : random.randint(0, 1),
            "due_date" : due_date,
            "pos_time" : pos_time,
            "status"   : 0,
            "blocked"  : 0,
            "operations": []
        }

        for _ in range(nb_operations):
            available_types = [t for t in types_operations if t != last_type]
            chosen_type     = random.choice(available_types)
            proc_time       = random.randint(min_duration, max_duration)
            op = {
                "type"           : chosen_type,
                "processing_time": proc_time
            }
            last_type = chosen_type
            job["operations"].append(op)

        # calcule du temps min necessaire pour traiter un job (somme de ces processing time + cout de transport total)
        proc_total     = sum(o["processing_time"] for o in job["operations"])
        """transport      = (2 * L + pos_time) + nb_operations * (2 * M)
        min_treat_time = proc_total + transport

        # release_date >= cut_time
        # release_date ne doit pas depasser due_date - min_treat_time
        borne_sup_release = max(0, min(due_date_max // 3, due_date - min_treat_time)) 

        release_date      = random.randint(cut_time, max(cut_time, borne_sup_release))
        job["release_date"] = release_date"""

        transport      = (2 * L + pos_time) + nb_operations * (2 * M)
        min_treat_time = proc_total + transport

        borne_sup    = max(cut_time, due_date - min_treat_time)
        release_date = random.randint(cut_time, borne_sup)
        due_date     = max(due_date, release_date + min_treat_time)

        job["due_date"]     = due_date
        job["release_date"] = release_date


        jobs.append(job)
    return jobs

def generate_one_instance(
        numbre_orders: int = 3,
        numbre_jobs_per_order: int = 3,
        max_operations_per_job: int = 2,
        types_operations: list = [0, 1],
        min_duration: int = 10,
        max_duration: int = 60,
        pos_time: int = 5,
        due_date_min: int = 50,
    ):
    """Generate one full instance made of multiple orders."""

    orders = []
    cut_time  = 0

    for c in range(numbre_orders):
        jobs = generate_order(
            numbre_jobs          = numbre_jobs_per_order,
            max_operations_per_job = max_operations_per_job,
            types_operations     = types_operations,
            min_duration            = min_duration,
            max_duration            = max_duration,
            pos_time             = pos_time,
            due_date_min         = due_date_min,
            cut_time             = cut_time,
        )

        orders.append({
            "id"      : c + 1,
            "cut_time": cut_time,
            "jobs"    : jobs
        })

        # Next cut_time = current cut_time + fraction of UB_cmax of this order
        UB_cmax   = (2 * L + pos_time) * numbre_jobs_per_order + max_operations_per_job * (2 * M + max_duration)
        cut_time += int(UB_cmax * random.uniform(0.3, 0.7))

    return {'a': random.randint(0, 10) * 10, 'orders': orders}

def save_instances_json(folder, instances):
    os.makedirs(folder, exist_ok=True)
    for i, instance in enumerate(instances):
        file_name = f"instance_{i+1}.json"
        path = os.path.join(folder, file_name)
        with open(path, "w") as f:
            json.dump(instance, f, indent=4)

# TEST WITH: python orders_generator.py --train=150 --test=50 --path=./
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Order Generator")
    parser.add_argument("--path",  help="path to save the instances", required=True)
    parser.add_argument("--train", help="number of training instances", required=True)
    parser.add_argument("--test",  help="number of test instances",     required=True)
    args = parser.parse_args()

    nb_train: int  = int(args.train)
    nb_test: int   = int(args.test)
    base_path: str = args.path + "data/orders_instances/"

    for size_name, job_min, job_max, nb_orders in ORDERS_SIZES:
        train_instances: list = []
        for _ in range(nb_train):
            numbre_jobs = random.randint(job_min, job_max)
            train_instances.append(generate_one_instance(
                numbre_orders         = nb_orders,
                numbre_jobs_per_order = numbre_jobs,
                max_operations_per_job   = 2,
            ))
        save_instances_json(base_path + "train/" + size_name + "/", train_instances)

        test_instances: list = []
        for _ in range(nb_test):
            numbre_jobs = random.randint(job_min, job_max)
            test_instances.append(generate_one_instance(
                numbre_orders         = nb_orders,
                numbre_jobs_per_order = numbre_jobs,
                max_operations_per_job   = 2,
            ))
        save_instances_json(base_path + "test/" + size_name + "/", test_instances)