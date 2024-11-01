from paho.mqtt.client import Client as MqttClient
import paho.mqtt.client as mqtt

import torch
import sys
sys.path.append("../")
 
class Client(MqttClient):
    def __init__(self, client_id="", broker_host="", clean_session=None, userdata=None, protocol=mqtt.MQTTv311, transport="tcp"):
        super().__init__(client_id, broker_host, clean_session, userdata, protocol, transport)

        self.on_connect = self.on_connect_callback
        self.on_disconnect = self.on_disconnect_callback
        self.on_message = self.on_message_callback
        self.on_subscribe = self.on_subscribe_callback

    def on_connect_callback(self, client, userdata, flags, rc):
        raise NotImplementedError("Please write a testing method for the client.")

    def on_disconnect_callback(self, client, userdata, flags, rc):
        raise NotImplementedError("Please write a testing method for the client.")

    def on_message_callback(self, client, userdata, flags, rc):
        raise NotImplementedError("Please write a testing method for the client.")

    def on_subscribe_callback(self, client, userdata, flags, rc):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def do_evaluate_connection(self):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def do_train(self):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_evaluate_data(self):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_test(self):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_update_model(self):
        raise NotImplementedError("Please write a testing method for the client.")

    def handle_task(self, msg):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def join_dFL_topic(self):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_add_errors(self):
        raise NotImplementedError("Please write a testing method for the client.")

    def handle_cmd(self, msg):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def handle_model(self, client, userdata, msg):
        raise NotImplementedError("Please write a testing method for the client.")

    def handle_recall(self, msg):
        raise NotImplementedError("Please write a testing method for the client.")

    def start(self):
        raise NotImplementedError("Please write a testing method for the client.")