import json
import paho.mqtt.client as client
import logging
import datetime
import argparse
import sys
import time
sys.path.append("../")
from src.utils import *
from src.add_config import *

from src.main.strategies_fl.FedAvg import FedAvg_Server, FedAvg_Client

def run():
    server_config['host'] = args.host
    server_config['port'] = args.port
    server_config['seed'] = args.seed
    server_config['num_rounds'] = args.num_rounds
    server_config['num_clients'] = args.num_clients
    server_config['strategy'] = args.strategy
    server_config['dataset'] = args.dataset
    server_config['num_classes'] = args.num_classes
    server_config['partition'] = args.partition
    server_config['data_volume_each_client'] = args.data_volume_each_client
    server_config['beta'] = args.beta
    server_config['rho'] = args.rho

    if args.strategy == "FedAvg":
        server_running = FedAvg_Server("server", server_config=server_config)
    else:
        raise ValueError("Invalid strategy!")

    server_running.connect(server_config['host'], port=server_config['port'], keepalive=3600)
    server_running.on_connect
    server_running.on_disconnect
    server_running.on_message
    server_running.on_subscribe
    server_running.loop_start()
    server_running.subscribe(topic = "dynamicFL/join")
    print_log(f"server sub to dynamicFL/join of {server_config['host']}")
    print_log("server is waiting for clients to join the topic ...")

    while (server_running.NUM_DEVICE > len(server_running.client_dict)):
       time.sleep(1)

    server_running.start_round
    server_running._thread.join()
    time.sleep(10)
    print_log("server exits", "red", show_time=True)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning_MQTT_1.0")
    parser.add_argument("--host", default="192.168.1.119", type=str, help="broker host")
    parser.add_argument("--port", default=1883, type=int, help="broker port")
    parser.add_argument("--seed", default=2024, type=int, help="random seed")
    parser.add_argument("--num_rounds", default=10, type=int, help="number of rounds FL communications")
    parser.add_argument("--num_clients", default=10, type=int, help="number of all clients join to FL")
    parser.add_argument("--strategy", default='FedAvg', type=str, help="strategy for trainning_aggregation")
    parser.add_argument("--dataset", default='Cifar10', type=str, help="dataset using")
    parser.add_argument("--num_classes", default=10, type=int, help="num classes of the dataset")
    parser.add_argument("--partition", default="noniid_label_distribution", type=str, help="type of split dataset to each clients")
    parser.add_argument("--data_volume_each_client", default="equal", type=str, help="equal or unequal")
    parser.add_argument("--beta", default=0.7, type=float, help="beta in dirichlet")
    parser.add_argument("--rho", default=0.9, type=float, help="rho in DGA")
    args = parser.parse_args()

    run()