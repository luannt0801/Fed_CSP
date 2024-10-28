from paho.mqtt.client import Client as MqttClient, MQTTv311

import torch

class Client(MqttClient):
    def __init__(self, client_id="", clean_session=None, userdata=None, protocol=..., transport="tcp", reconnect_on_failure=True, **kwargs):
        super().__init__(client_id, clean_session, userdata, protocol, transport, reconnect_on_failure, **kwargs)
        
        self._client_id = client_id

        if 'broker_name' not in kwargs:
            raise ValueError("Please input broker host")
        else:
            self.broker_host = kwargs['broker_host']
        
        broker_name = self.broker_host

        self.on_connect = self.on_connect_callback
        self.on_disconnect = self.on_disconnect_callback
        self.on_message = self.on_message_callback
        self.on_subscribe = self.on_subscribe_callback

    def on_connect_callback(self, client, userdata, flags, rc, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def on_disconnect_callback(self, client, userdata, flags, rc, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def on_message_callback(self, client, userdata, flags, rc, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def on_subscribe_callback(self, client, userdata, flags, rc, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def do_evaluate_connection(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def do_train(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_evaluate_data(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_test(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_update_model(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def handle_task(self, msg, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def join_dFL_topic(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def do_add_errors(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def handle_cmd(self, msg, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")
    
    def handle_model(self, client, userdata, msg, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def handle_recall(self, msg, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")

    def start(self, **kwargs):
        raise NotImplementedError("Please write a testing method for the client.")