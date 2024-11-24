import argparse
import datetime
import json
import logging
import sys
import time

import paho.mqtt.client as client
import torch

sys.path.append("../")
import paho.mqtt.client as mqtt
from paho.mqtt.client import Client as MqttClient
from paho.mqtt.client import MQTTv311

from src.add_config import *
from src.logging import *
from src.main.strategies_fl.FedAvg import FedAvg_Client, FedAvg_Server
from src.main.strategies_fl.Local import local_running
from src.utils import *


def run():
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    logger.info(f"-----------Server start Federated Learning----------- \n")
    # logger.info(f"Print server_config: \n {server_config}")
    # for input, value in server_config.items():
    #     print_log(f"{input}: {value}", color_='yellow')

    server_config["host"] = args.host
    server_config["port"] = args.port
    server_config["seed"] = args.seed
    server_config["num_rounds"] = args.num_rounds
    server_config["num_clients"] = args.num_clients
    server_config["strategy"] = args.strategy

    if args.strategy != "Local":
        if args.strategy == "FedAvg":
            server_running = FedAvg_Server(client_fl_id="server")
        else:
            raise ValueError("Invalid strategy!")

        server_running.connect(
            host=server_config["host"], port=server_config["port"], keepalive=3600
        )
        server_running.on_connect
        server_running.on_disconnect
        server_running.on_message
        server_running.on_subscribe
        server_running.loop_start()
        server_running.subscribe(topic="dynamicFL/join")

        while server_running.NUM_DEVICE > len(server_running.client_dict):
            #   logger.debug("NUM_DEVICE join to broker: "+ str(server_running.NUM_DEVICE))
            #   logger.debug(len(server_running.client_dict))
            time.sleep(1)

        server_running.start_round()
        server_running._thread.join()
        time.sleep(10)
        print_log("server exits", "red", show_time=True)

    else:
        # do trainning local here.
        num_clients = server_config["num_clients"]
        num_rounds = server_config["num_rounds"]
        local_running(num_clients, num_rounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning_MQTT_1.0")
    parser.add_argument(
        "--host", default=server_config["host"], type=str, help="broker host"
    )
    parser.add_argument(
        "--port", default=server_config["port"], type=int, help="broker port"
    )
    parser.add_argument(
        "--seed", default=server_config["seed"], type=int, help="random seed"
    )
    parser.add_argument(
        "--num_rounds",
        default=server_config["num_rounds"],
        type=int,
        help="number of rounds FL communications",
    )
    parser.add_argument(
        "--num_clients",
        default=server_config["num_clients"],
        type=int,
        help="number of all clients join to FL",
    )
    parser.add_argument(
        "--strategy",
        default=server_config["strategy"],
        type=str,
        help="strategy for trainning_aggregation",
    )
    args = parser.parse_args()

    run()
