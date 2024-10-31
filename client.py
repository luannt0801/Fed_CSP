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
from src.logging import *
 
from src.main.strategies_fl.draft import FedAvg_Server, FedAvg_Client
from paho.mqtt.client import Client as MqttClient, MQTTv311

def run():
    client_id = f"client_+{args.ID}"
    if len(sys.argv) == '':
        print("Usage: python client.py [client_id]")
        sys.exit(1)

    client_id = "client_" + sys.argv[1]
    print(client_id)
    time.sleep(5)
    if args.strategy == "FedAvg":
        client_running = FedAvg_Client(client_id=client_id, broker_host= client_config['host'], clean_session=None, userdata=None, protocol=MQTTv311, transport="tcp")
    else:
        raise ValueError("Invalid strategy!")
    
    client_running.start(broker_host = client_config['host'], port=1883, keepalive=3600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning_MQTT_1.0")
    parser.add_argument("--ID", type=str, help="broker host")
    parser.add_argument("--host", default="192.168.1.119", type=str, help="broker host")
    parser.add_argument("--strategy", default='FedAvg', type=str, help="strategy for trainning_aggregation")
    parser.add_argument("--model", default="LSTMModel", type=str, help="Model used for trainning in client.")
    parser.add_argument("--epochs", default="1", type=str, help="Number of epochs for trainning in client.")
    args = parser.parse_args()
 
    run()
