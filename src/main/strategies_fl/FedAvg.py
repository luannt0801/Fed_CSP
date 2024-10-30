from src.add_config import *
from src.utils import *
from ..client_fl import Client
from ..server_fl import Server
from src.model_install.run_model import trainning_model
from src.logging import *

from paho.mqtt.client import Client as MqttClient, MQTTv311

import torch
import json
import paho.mqtt.client as mqtt
import time
import threading
from collections import OrderedDict
import sys
sys.path.append("../")


#################################################
################ CLIENT SIDE ####################
#################################################

class FedAvg_Client(Client):

    def __init__(self, client_id="", clean_session=None, userdata=None, protocol=..., transport="tcp", reconnect_on_failure=True):
        super().__init__(client_id, clean_session, userdata, protocol, transport, reconnect_on_failure)


    def on_connect(self, client, userdata, flags, rc, **kwargs):
        print_log(f"Connected with result code {rc}")
        self.join_dFL_topic()

    def on_disconnect(self, client, userdata, rc, **kwargs):
        print_log(f"Disconnected with result code {rc}")
        self.reconnect()

    def on_message(self, client, userdata, msg, **kwargs):
        print_log(f"on_message {self._client_id.decode()}")
        print_log(f"RECEIVED msg from {msg.topic}")
        topic = msg.topic
        if topic == "dynamicFL/req/"+self._client_id:
            self.handle_cmd(msg)

    def on_subscribe(self, client, userdata, mid, granted_qos, **kwargs):
        print_log(f"Subscribed: {mid} {granted_qos}")

    def do_evaluate_connection(self, **kwargs):
        print_log("doing ping")
        result = ping_host(self.broker_host)
        result["client_id"] = self._client_id
        result["task"] = "EVA_CONN"
        self.publish(topic="dynamicFL/res/"+self._client_id, payload=json.dumps(result))
        print_log(f"Published to topic dynamicFL/res/{self._client_id}")
        return result

    def do_train(self, **kwargs):
        print_log("Client start trainning . . .")
        client_id = self._client_id

        result = trainning_model()

        torch.save(result, f'model_client/model_client_{client_id}.pt')

        # Convert tensors to numpy arrays
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "TRAIN",
            "weight": result_np
        }
        self.publish(topic="dynamicFL/res/" + client_id, payload=json.dumps(payload))
        print_log(f"end training")

    def do_evaluate_data(self, **kwargs):
        pass

    def do_test(self, **kwargs):
        pass

    def do_update_model(self, **kwargs):
        pass

    def do_stop_client(self, **kwargs):
        print_log("stop client")
        self.loop_stop()

    def handle_task(self, msg, **kwargs):
        task_name = msg.payload.decode("utf-8")
        print(task_name)
        if task_name == "EVA_CONN":
            self.do_evaluate_connection()
        elif task_name == "EVA_DATA":
            self.do_evaluate_data()
        elif task_name == "TRAIN":
            self.do_train()
        elif task_name == "TEST":
            self.do_test()
        elif task_name == "UPDATE":
            self.do_update_model()
        elif task_name == "REJECTED":
            self.do_add_errors()
        elif task_name == "STOP":
            self.do_stop_client()
        else:
            print_log(f"Command {task_name} is not supported")

    def join_dFL_topic(self, **kwargs):
        self.publish(topic="dynamicFL/join", payload=self._client_id)
        print_log(f"{self._client_id} joined dynamicFL/join of {self.broker_host}")

    def do_add_errors(self, **kwargs):
        self.publish.single(topic="dynamicFL/errors", payload=self._client_id, hostname=self.broker_host, client_id=self._client_id)

    def wait_for_model(self, **kwargs):
        msg = self.subscribe.simple("dynamicFL/model", hostname=self.broker_host)
        with open("mymodel.pt", "wb") as fo:
            fo.write(msg.payload)
        print_log(f"{self._client_id} write model to mymodel.pt")

    def handle_cmd(self, msg, **kwargs):
        print_log("wait for cmd")
        self.handle_task(msg)
 
    def handle_model(self, client, userdata, msg, **kwargs):
        print_log("receive model")
        with open("newmode.pt", "wb") as f:
            f.write(msg.payload)
        print_log("done write model")
        result = {
            "client_id": self._client_id,
            "task": "WRITE_MODEL" 
        }
        self.publish(topic="dynamicFL/res/"+self._client_id, payload=json.dumps(result))

    def handle_recall(self, msg, **kwargs):
        task_name = msg.payload.decode("utf-8")

    def start(self, **kwargs):
        self.connect(self.broker_host, port=1883, keepalive=3600)
        self.message_callback_add("dynamicFL/model/all_client", self.handle_model)
        self.loop_start()
        self.subscribe(topic="dynamicFL/model/all_client")
        self.subscribe(topic="dynamicFL/req/" + self._client_id)
        self.subscribe(topic="dynamicFL/wait/" + self._client_id)
        self.publish(topic="dynamicFL/join", payload=self._client_id)
        print_log(f"{self._client_id} joined dynamicFL/join of {self.broker_host}")

        self._thread.join()
        print_log("client exits")


#################################################
################ SERVER SIDE ####################
#################################################

class FedAvg_Server(Server):
    def __init__(self, client_fl_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311, server_config={}):
        super().__init__(client_fl_id, clean_session, userdata, protocol)

        self.NUM_ROUND = server_config['num_rounds']
        self.NUM_DEVICE = server_config['num_clients']
        self.time_between_two_round = 1
        self.round_state = "finished"
        self.n_round = 0


    def on_connect_callback(self, client, userdata, flags, rc):
        print_log("Connected with result code "+str(rc))

    def on_disconnect_callback(self, client, userdata, rc):
        print_log("Disconnected with result code "+str(rc))
        self.reconnect()

    def on_message_callback(self, client, userdata, msg):
        print(f"received msg from {msg.topic}")
        topic = msg.topic
        if topic == "dynamicFL/join": # topic is join --> handle_join
            self.handle_join(self, userdata, msg)
        elif "dynamicFL/res" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_res(this_client_id, msg)

    def on_subscribe_callback(self, mosq, obj, mid, granted_qos):
        print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

    def send_task(self, task_name, client, this_client_id):
        print(this_client_id)
        print(task_name)
        print_log("publish to " + "dynamicFL/req/"+this_client_id)
        self.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)

    def send_model(self, path, client, this_client_id):
        f = open(path, "rb")
        data = f.read()
        f.close()
        self.publish(topic="dynamicFL/model/all_client", payload=data)

    def handle_res(self, this_client_id, msg):
        data = json.loads(msg.payload)
        cmd = data["task"]
        if cmd == "EVA_CONN":
            print_log(f"{this_client_id} complete task EVA_CONN")
            self.handle_pingres(this_client_id, msg)
        elif cmd == "TRAIN":
            print_log(f"{this_client_id} complete task TRAIN")
            self.handle_trainres(this_client_id, msg)
        elif cmd == "WRITE_MODEL":
            print_log(f"{this_client_id} complete task WRITE_MODEL")
            self.handle_update_writemodel(this_client_id, msg)

    def handle_join(self, client, userdata, msg):
        this_client_id = msg.payload.decode("utf-8")
        self.client_dict[this_client_id] = {
            "state": "joined"
        }
        self.subscribe(topic="dynamicFL/res/"+this_client_id)
 
    def handle_pingres(self, this_client_id, msg):
        print(msg.topic+" "+str(msg.payload.decode()))
        ping_res = json.loads(msg.payload)
        this_client_id = ping_res["client_id"]
        if ping_res["packet_loss"] == 0.0:
            print_log(f"{this_client_id} is a good client")
            state = self.client_dict[this_client_id]["state"]
            print_log(f"state {this_client_id}: {state}, round: {self.n_round}")
            if state == "joined" or state == "trained":
                self.client_dict[this_client_id]["state"] = "eva_conn_ok"
                #send_model("saved_model/FashionMnist.pt", server, this_client_id)
                #print(client_dict)
                count_eva_conn_ok = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "eva_conn_ok")
                if(count_eva_conn_ok == self.NUM_DEVICE):
                    print_log("publish to " + "dynamicFL/model/all_client")
                    if model_config['name'] == 'LSTM':
                        self.send_model("saved_model/LSTMModel.pt", "s", this_client_id) # send LSTM model for DGA data
                    elif model_config['name'] == 'Lenet':
                        self.send_model("saved_model/Lenet_model.pt", "s", this_client_id) # send Lenet model if using Lenet model

    def handle_trainres(self, this_client_id, msg):
        payload = json.loads(msg.payload.decode())
        
        self.client_trainres_dict[this_client_id] = payload["weight"]
        state = self.client_dict[this_client_id]["state"]
        if state == "model_recv":
            self.client_dict[this_client_id]["state"] = "trained"
        
    def handle_update_writemodel(self, this_client_id, msg):
        state = self.client_dict[this_client_id]["state"]
        if state == "eva_conn_ok":
            self.client_dict[this_client_id]["state"] = "model_recv"
            self.send_task("TRAIN", self, this_client_id) # hmm
            count_model_recv = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "model_recv")
            if(count_model_recv == self.NUM_DEVICE):
                print_log(f"Waiting for training round {self.n_round} from client...")


    def start_round(self):
        self.n_round = self.n_round + 1
        print_log(f"server start round {self.n_round}")
        self.round_state = "started"
        print(self.client_dict)
        for client_i in self.client_dict:
            self.send_task("EVA_CONN", self, client_i) # hmm
        while (len(self.client_trainres_dict) != self.NUM_DEVICE):
            time.sleep(1)
        time.sleep(1)
        self.end_round()

    def do_aggregate(self):
        print_log("Do aggregate ...")
        self.aggregated_models()
        
    def handle_next_round_duration(self):
        while (len(self.client_trainres_dict) < self.NUM_DEVICE):
            time.sleep(1)

    def end_round(self):
        print_log(f"server end round {self.n_round}")
        self.round_state = "finished"

        if self.n_round < self.NUM_ROUND:
            self.handle_next_round_duration()
            self.do_aggregate()
            t = threading.Timer(self.time_between_two_round, self.start_round)
            t.start()
        else:
            self.do_aggregate()
            for c in self.client_dict:
                self.send_task("STOP", self, c)
                print_log(f"send task STOP {c}")
            self.loop_stop()
        

    def aggregated_models(self):
        sum_state_dict = OrderedDict()
        for client_id, state_dict in self.client_trainres_dict.items():
            for key, value in state_dict.items():
                if key in sum_state_dict:
                    sum_state_dict[key] = sum_state_dict[key] + torch.tensor(value, dtype=torch.float32)
                else:
                    sum_state_dict[key] = torch.tensor(value, dtype=torch.float32)

        num_models = len(self.client_trainres_dict)
        avg_state_dict = OrderedDict((key, value / num_models) for key, value in sum_state_dict.items())
        
        torch.save(avg_state_dict, f'model_round/model_round_{self.n_round}.pt')
        if model_config['name'] == 'LSTM':
            torch.save(avg_state_dict, "saved_model/LSTMModel.pt")
        elif model_config['name'] == 'Lenet':
            torch.save(avg_state_dict, "saved_model/Lenet_model.pt")
        self.client_trainres_dict.clear()