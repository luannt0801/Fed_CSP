import argparse
import datetime
import json
import logging
import sys
import time

import paho.mqtt.client as client

sys.path.append("../")
from paho.mqtt.client import Client as MqttClient
from paho.mqtt.client import MQTTv311

from src.add_config import *
from src.logging import *
from src.main.strategies_fl.FedAvg import FedAvg_Client
from src.main.strategies_fl.FedCluster_CS import FedAvg_CS_Client
from src.utils import *


def run():

    client_config["ID"] = args.ID
    client_config["host"] = args.host
    client_config["strategy"] = args.strategy
    client_config["model"] = args.model
    client_config["epochs"] = args.epochs

    if client_config["ID"] is None:
        raise ValueError("Please input client ID", color="red")

    logger.debug(f"\n Client config: \n {client_config}")
    client_id = f"client_{args.ID}"

    print_log(client_id, color_="red", show_time=True)
    time.sleep(1)

    logger.debug("show strategy client: %s", args.strategy)

    if args.strategy == "FedAvg":
        client_running = FedAvg_Client(
            client_id=client_id, broker_host=client_config["host"]
        )
    elif args.strategy == "FedAvg_CS":
        client_running = FedAvg_CS_Client(
            client_id=client_id, broker_host=client_config["host"]
        )
    else:
        raise ValueError("Invalid strategy!")

    client_running.start()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning_MQTT_1.0")
    parser.add_argument(
        "--ID", default=client_config["ID"], type=str, help="broker host"
    )
    parser.add_argument(
        "--host", default=client_config["host"], type=str, help="broker host"
    )
    parser.add_argument(
        "--strategy",
        default=client_config["strategy"],
        type=str,
        help="strategy for trainning_aggregation",
    )
    parser.add_argument(
        "--model",
        default=model_config["model_run"],
        type=str,
        help="Model used for trainning in client.",
    )
    parser.add_argument(
        "--epochs",
        default=client_config["num_epochs"],
        type=str,
        help="Number of epochs for trainning in client.",
    )
    args = parser.parse_args()

    run()
