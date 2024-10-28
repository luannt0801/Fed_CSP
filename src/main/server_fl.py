from paho.mqtt.client import Client as MqttClient, MQTTv311
from collections import OrderedDict
from ..utils import *
from ..add_config import *
import paho.mqtt.client as mqtt


import torch
import threading
import json
import time
import sys
sys.path.append("../")


class Server(MqttClient):
    def __init__(self, client_fl_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311, **kwargs):
        super().__init__(client_fl_id, clean_session, userdata, protocol, **kwargs)

        # Set callbacks
        self.on_connect = self.on_connect_callback
        self.on_message = self.on_message_callback
        self.on_disconnect = self.on_disconnect_callback
        self.on_subscribe = self.on_subscribe_callback

        self.client_dict = {}
        self.client_trainres_dict = {}

        # default install
        self.NUM_ROUND = kwargs['num_rounds']
        self.NUM_DEVICE = kwargs['num_clients']
        self.time_between_two_round = 1
        self.round_state = "finished"
        self.n_round = 0

    def on_connect_callback(self, client, userdata, flags, rc):
        raise NotImplementedError("Please write a testing method for the server.")

    def on_disconnect_callback(self, client, userdata, rc):
        raise NotImplementedError("Please write a testing method for the server.")

    def on_message_callback(self, client, userdata, msg):
        raise NotImplementedError("Please write a testing method for the server.")

    def on_subscribe_callback(self, mosq, obj, mid, granted_qos):
        raise NotImplementedError("Please write a testing method for the server.")

    def send_task(self, task_name, client, this_client_id):
        raise NotImplementedError("Please write a testing method for the server.")

    def send_model(self, path, client, this_client_id):
       raise NotImplementedError("Please write a testing method for the server.")

    def handle_res(self, this_client_id, msg):
        raise NotImplementedError("Please write a testing method for the server.")

    def handle_join(self, client, userdata, msg):
        raise NotImplementedError("Please write a testing method for the server.")
        
    def handle_wait(self):
        raise NotImplementedError("Please write a testing method for the server.")
 
    def handle_pingres(self, this_client_id, msg):
        raise NotImplementedError("Please write a testing method for the server.")

    def handle_trainres(self, this_client_id, msg):
        raise NotImplementedError("Please write a testing method for the server.")
        
    def handle_update_writemodel(self, this_client_id, msg):
        raise NotImplementedError("Please write a testing method for the server.")


    def start_round(self):
        raise NotImplementedError("Please write a testing method for the server.")

    def do_aggregate(self):
        raise NotImplementedError("Please write a testing method for the server.")
        
    def handle_next_round_duration(self):
        raise NotImplementedError("Please write a testing method for the server.")
    
    def end_round(self):
        raise NotImplementedError("Please write a testing method for the server.")
        

    def aggregated_models(self):
        raise NotImplementedError("Please write a testing method for the server.")