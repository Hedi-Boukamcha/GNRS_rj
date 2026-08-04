# test_order_loader.py
from models.order import Order, OrderInstance

order_instance = OrderInstance.load("data/orders_instances/test/s/instance_1.json")
order_instance.display()

# TEST WITH: python test_order_loader.py