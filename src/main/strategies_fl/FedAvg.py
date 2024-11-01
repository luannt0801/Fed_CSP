from src.utils import *
from src.add_config import *

from src.model_install.run_model import trainning_model # for another dataset
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client as mqtt
import json
import torch
import time
import threading
from collections import OrderedDict
from paho.mqtt.client import Client as MqttClient

from src.add_config import *
from src.logging import *

class FedAvg_Client():
    def __init__(self, client_id, broker_host):
        self.client_id = client_id
        self.broker_name = broker_host

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe

    def on_connect(self, client, userdata, flags, rc):
        print_log(f"Connected with result code {rc}")
        self.join_dFL_topic()

    def on_disconnect(self, client, userdata, rc):
        print_log(f"Disconnected with result code {rc}")
        self.client.reconnect()

    def on_message(self, client, userdata, msg):
        print_log(f"on_message {client._client_id.decode()}")
        print_log(f"RECEIVED msg from {msg.topic}")
        topic = msg.topic
        print(topic)
        if topic == "dynamicFL/req/"+self.client_id:
            self.handle_cmd(msg)
        elif topic == "dynamicFL/model/all_client":
            self.handle_model(client, userdata, msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print_log(f"Subscribed: {mid} {granted_qos}")

    def do_evaluate_connection(self):
        print_log("doing ping")
        result = ping_host(self.broker_name)
        result["client_id"] = self.client_id
        result["task"] = "EVA_CONN"
        self.client.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))
        print_log(f"Published to topic dynamicFL/res/{self.client_id}")
        return result

    def do_train(self):
        print_log("Client start trainning . . .")
        client_id = self.client_id
        result = trainning_model()

        # Convert tensors to numpy arrays
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "TRAIN",
            "weight": result_np
        }
        self.client.publish(topic="dynamicFL/res/" + client_id, payload=json.dumps(payload))
        print_log(f"end training")

    def do_evaluate_data(self):
        pass

    def do_test(self):
        pass

    def do_update_model(self):
        pass

    def do_stop_client(self):
        print_log("stop client")
        self.client.loop_stop()

    def handle_task(self, msg):
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

    def join_dFL_topic(self):
        self.client.publish(topic="dynamicFL/join", payload=self.client_id)
        print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

    def do_add_errors(self):
        publish.single(topic="dynamicFL/errors", payload=self.client_id, hostname=self.broker_name, client_id=self.client_id)

    def wait_for_model(self):
        msg = subscribe.simple("dynamicFL/model", hostname=self.broker_name)
        with open("mymodel.pt", "wb") as fo:
            fo.write(msg.payload)
        print_log(f"{self.client_id} write model to mymodel.pt")

    def handle_cmd(self, msg):
        print_log("wait for cmd")
        self.handle_task(msg)
 
    def handle_model(self, client, userdata, msg):
        print_log("receive model")
        with open("newmode.pt", "wb") as f:
            f.write(msg.payload)
        print_log("done write model")
        result = {
            "client_id": self.client_id,
            "task": "WRITE_MODEL" 
        }
        self.client.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))

    def handle_recall(self, msg):
        print("do handle_recall")
        task_name = msg.payload.decode("utf-8")
        if task_name == "RECALL":
            self.do_recall()

    def start(self):
        self.client.connect(self.broker_name, port=1883, keepalive=3600)
        self.client.message_callback_add("dynamicFL/model/all_client", self.handle_model)
        self.client.loop_start()
        self.client.subscribe(topic="dynamicFL/model/all_client")
        self.client.subscribe(topic="dynamicFL/req/" + self.client_id)
        self.client.subscribe(topic="dynamicFL/wait/" + self.client_id)
        self.client.publish(topic="dynamicFL/join", payload=self.client_id)
        print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

        self.client._thread.join()
        print_log("client exits")


class FedAvg_Server(MqttClient):
    def __init__(self, client_fl_id,  clean_session=True, userdata=None, protocol=mqtt.MQTTv311):
        super().__init__(client_fl_id, clean_session, userdata, protocol)

        self.on_connect = self.on_connect_callback
        self.on_message = self.on_message_callback
        self.on_disconnect = self.on_disconnect_callback
        self.on_subscribe = self.on_subscribe_callback

        self.client_dict = {}
        self.client_trainres_dict = {}

        self.NUM_ROUND = 10
        self.NUM_DEVICE = 1
        self.time_between_two_round = 1
        self.round_state = "finished"
        self.n_round = 0

    # check connect to broker return result code
    def on_connect_callback(self, client, userdata, flags, rc):
        logger.info(f"Do on_connect_callback")
        print_log("Connected with result code "+str(rc))

    def on_disconnect_callback(self, client, userdata, rc):
        logger.info(f"Do on_disconnect_callback")
        print_log("Disconnected with result code "+str(rc))
        self.reconnect()

    # handle message receive from client
    def on_message_callback(self, client, userdata, msg):
        logger.info(f"Do on_message_callback")
        topic = msg.topic
        if topic == "dynamicFL/join": # topic is join --> handle_join
            self.handle_join(self, userdata, msg)
        elif "dynamicFL/res" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_res(this_client_id, msg)

    def on_subscribe_callback(self, mosq, obj, mid, granted_qos):
        logger.info(f"Do on_subscribe_callback")
        print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

    def send_task(self, task_name, client, this_client_id):
        logger.info(f"Do send_task")
        print_log("publish to " + "dynamicFL/req/"+this_client_id)
        self.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)

    def send_model(self, path, client, this_client_id):
        logger.info(f"Do send_model")
        f = open(path, "rb")
        data = f.read()
        f.close()
        self.publish(topic="dynamicFL/model/all_client", payload=data)

    def handle_res(self, this_client_id, msg):
        logger.info(f"Do handle_res")
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
        logger.info(f"Do handle_join")
        this_client_id = msg.payload.decode("utf-8")
        print_log("joined from"+" "+this_client_id)
        self.client_dict[this_client_id] = {
            "state": "joined"
        }
        self.subscribe(topic="dynamicFL/res/"+this_client_id)
 
    def handle_pingres(self, this_client_id, msg):
        logger.info(f"Do handle_pingres")
        ping_res = json.loads(msg.payload)
        this_client_id = ping_res["client_id"]
        if ping_res["packet_loss"] == 0.0:
            print_log(f"{this_client_id} is a good client")
            state = self.client_dict[this_client_id]["state"]
            print_log(f"state {this_client_id}: {state}, round: {self.n_round}")
            if state == "joined" or state == "trained":
                self.client_dict[this_client_id]["state"] = "eva_conn_ok"
                count_eva_conn_ok = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "eva_conn_ok")
                if(count_eva_conn_ok == self.NUM_DEVICE):
                    print_log("publish to " + "dynamicFL/model/all_client")
                    self.send_model("src/parameter/server_model.pt", "s", this_client_id) # send LSTM model for DGA data

    def handle_trainres(self, this_client_id, msg):
        logger.info("Do hane")
        payload = json.loads(msg.payload.decode())
        
        self.client_trainres_dict[this_client_id] = payload["weight"]
        state = self.client_dict[this_client_id]["state"]
        if state == "model_recv":
            self.client_dict[this_client_id]["state"] = "trained"
        print("done train!")
        
    def handle_update_writemodel(self, this_client_id, msg):
        logger.info(f"Do handle_update_writemodel")
        state = self.client_dict[this_client_id]["state"]
        if state == "eva_conn_ok":
            self.client_dict[this_client_id]["state"] = "model_recv"
            self.send_task("TRAIN", self, this_client_id) # hmm
            count_model_recv = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "model_recv")
            if(count_model_recv == self.NUM_DEVICE):
                print_log(f"Waiting for training round {self.n_round} from client...")


    def start_round(self):
        logger.info(f"Do start_round")
        self.n_round
        self.n_round = self.n_round + 1

        print_log(f"server start round {self.n_round}")
        self.round_state = "started"

        for client_i in self.client_dict:
            self.send_task("EVA_CONN", self, client_i)
        while (len(self.client_trainres_dict) != self.NUM_DEVICE):
            time.sleep(1)
        time.sleep(1)
        self.end_round()

    def do_aggregate(self):
        logger.info(f"Do do_aggregate")
        print_log("Do aggregate ...")
        self.aggregated_models()
        
    def handle_next_round_duration(self):
        logger.info(f"Do handle_next_round_duration")
        while (len(self.client_trainres_dict) < self.NUM_DEVICE):
            time.sleep(1)

    def end_round(self):
        logger.info(f"Do end_round")
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
        logger.info(f"Do aggregated_models")
        sum_state_dict = OrderedDict()

        for client_id, state_dict in self.client_trainres_dict.items():
            for key, value in state_dict.items():
                if key in sum_state_dict:
                    sum_state_dict[key] = sum_state_dict[key] + torch.tensor(value, dtype=torch.float32)
                else:
                    sum_state_dict[key] = torch.tensor(value, dtype=torch.float32)
        num_models = len(self.client_trainres_dict)
        avg_state_dict = OrderedDict((key, value / num_models) for key, value in sum_state_dict.items())
        torch.save(avg_state_dict, "saved_model/LSTMModel.pt")
        self.client_trainres_dict.clear()