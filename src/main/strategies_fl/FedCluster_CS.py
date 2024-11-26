import json
import threading
import time
from collections import Counter, OrderedDict

import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import torch
from paho.mqtt.client import Client as MqttClient

from src.add_config import *
from src.logging import *
from src.model_install.handle_data import *
from src.model_install.run_model import trainning_model, testing_model_server  # for another dataset
from src.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FedAvg_CS_Client:
    def __init__(self, client_id, broker_host):
        self.client_id = client_id
        self.broker_name = broker_host

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.trainloader = None
        self.testloader = None

        self.trainloader, self.testloader = self.get_dataset()

    def on_connect(self, client, userdata, flags, rc):
        # if logger_config["show"] == True:
        #     logger.debug("Do on_connect")
            # print_log(f"Connected with result code {rc}")
        self.join_dFL_topic()

    def on_disconnect(self, client, userdata, rc):
        # if logger_config["show"] == True:
        #     logger.debug("Do on_disconnect")
            # print_log(f"Disconnected with result code {rc}")
        self.client.reconnect()

    def on_message(self, client, userdata, msg):
        # if logger_config["show"] == True:
        #     logger.debug("Do on_message")
            # print_log(f"on_message {client._client_id.decode()}")
            # print_log(f"RECEIVED msg from {msg.topic}")
        topic = msg.topic
        # print(topic)
        if topic == "dynamicFL/req/" + self.client_id:
            self.handle_cmd(msg)
        elif topic == "dynamicFL/model/all_client":
            self.handle_model(client, userdata, msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        # if logger_config["show"] == True:
        #     logger.debug("Do on_subscribe")
            # print_log(f"Subscribed: {mid} {granted_qos}")
        pass
    """
        Can send data loader hear
    """

    def do_evaluate_connection(self):
        # if logger_config["show"] == True:
        #     logger.debug("Do do_evaluate_connection")
        #     print_log("doing ping")
        result = ping_host(self.broker_name)
        result["client_id"] = self.client_id
        result["task"] = "EVA_CONN"
        result["data"] = counter_dataloader(
            dataloader=self.trainloader, num_classes=data_config["num_classes"]
        )
        self.client.publish(
            topic="dynamicFL/res/" + self.client_id, payload=json.dumps(result)
        )
        # if logger_config["show"] == True:
            # print_log(f"Published to topic dynamicFL/res/{self.client_id}")
        return result

    def get_dataset(self):
        if logger_config["show"] == True:
            logger.debug("Do get_dataset")

        all_trainset, all_testset = get_Dataset(
            datasetname=data_config["dataset"],
            datapath="D:\\Project\\FedCSP\\data\\images",
        )  # include images & DGA

        all_client_trainset = split_data(
            dataset_use=all_trainset,
            dataset=data_config["dataset"],
            data_for_client=data_config["data_for_client_train"],
            num_classes=data_config["num_classes"],
            partition=data_config["partition"],
            data_volume_each_client=data_config["data_volume_each_client"],
            beta=data_config["beta"],
            rho=data_config["rho"],
            num_client=server_config["num_clients"],
        )

        all_client_testset = split_data(
            dataset_use=all_testset,
            dataset=data_config["dataset"],
            data_for_client=data_config["data_for_client_test"],
            num_classes=data_config["num_classes"],
            partition=data_config["partition"],
            data_volume_each_client=data_config["data_volume_each_client"],
            beta=data_config["beta"],
            rho=data_config["rho"],
            num_client=server_config["num_clients"],
        )

        # debug data in each client
        if logger_config["show"] == True:
            logger.info(f"{self.client_id}: \n")

            trainset_client = all_client_trainset[self.client_id]
            logger.debug(f"Train data in {self.client_id} :")
            logger.debug(trainset_client)
            logger.debug("\n")

            test_client = all_client_testset[self.client_id]
            logger.debug(f"Test data in {self.client_id} :")
            logger.debug(test_client)
            logger.debug("\n")

        trainset = all_client_trainset[self.client_id]
        trainloader = DataLoader(
            trainset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            drop_last=data_config["drop_last"],
        )

        trainset = all_client_testset[self.client_id]
        testloader = DataLoader(
            trainset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            drop_last=data_config["drop_last"],
        )

        # Assuming client_dataloader is your DataLoader for the client
        # if logger_config["show"] == True:
        all_labels = []

        for _, labels in trainloader:
            all_labels.extend(labels.tolist())  # Collect all labels into a list

        # Convert list to tensor and get unique labels with counts
        all_labels_tensor = torch.tensor(all_labels)
        labels, counts = torch.unique(all_labels_tensor, return_counts=True)

        # Print the labels and their counts for the client
        print(f"{self.client_id} - Labels and their counts in DataLoader:")
        for label, count in zip(labels, counts):
            print(f"Label {label.item()}: {count.item()} samples")

        print(f"DataLoader for client {self.client_id} is ready.")

        return trainloader, testloader

    def do_train(self):
        # if logger_config["show"] == True:
        #     logger.debug("Do do_train")
        print_log("Client start trainning . . .")
        client_id = self.client_id
        # trainloader, testloader = self.get_dataset()
        result = trainning_model(
            self.trainloader,
            self.testloader,
            model_run=model_config["model_run"],
            num_classes=data_config["num_classes"],
            epochs=client_config["num_epochs"],
            batch_size=model_config["batch_size"],
        )

        # Convert tensors to numpy arrays
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {"task": "TRAIN", "weight": result_np}
        self.client.publish(
            topic="dynamicFL/res/" + client_id, payload=json.dumps(payload)
        )
        print_log(f"end training")

    def do_evaluate_data(self):
        if logger_config["show"] == True:
            logger.debug("Do do_evaluate_data")
        pass

    def do_test(self):
        if logger_config["show"] == True:
            logger.debug("Do do_test")
        pass

    def do_update_model(self):
        if logger_config["show"] == True:
            logger.debug("Do do_update_model")
        pass

    def do_stop_client(self):
        if logger_config["show"] == True:
            logger.debug("Do do_stop_client")
        print_log("stop client")
        self.client.loop_stop()

    def handle_task(self, msg):
        if logger_config["show"] == True:
            logger.debug("Do handle_task")
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
        if logger_config["show"] == True:
            logger.debug("Do join_dFL_topic")
        self.client.publish(topic="dynamicFL/join", payload=self.client_id)
        print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

    def do_add_errors(self):
        if logger_config["show"] == True:
            logger.debug("Do do_add_errors")
        publish.single(
            topic="dynamicFL/errors",
            payload=self.client_id,
            hostname=self.broker_name,
            client_id=self.client_id,
        )

    # def wait_for_model(self):
    #     msg = subscribe.simple("dynamicFL/model", hostname=self.broker_name)
    #     with open("src/parameter/client_model.pt", "wb") as fo:
    #         fo.write(msg.payload)
    #     print_log(f"{self.client_id} write model to mymodel.pt")

    def handle_cmd(self, msg):
        if logger_config["show"] == True:
            logger.debug("Do handle_cmd")
            print_log("wait for cmd")
        self.handle_task(msg)

    def handle_model(self, client, userdata, msg):
        if logger_config["show"] == True:
            logger.debug("Do handle_model")
        print_log("receive model from Server")
        with open("src/parameter/client_model.pt", "wb") as f:
            f.write(msg.payload)
        print_log("done write model")
        result = {"client_id": self.client_id, "task": "WRITE_MODEL"}
        self.client.publish(
            topic="dynamicFL/res/" + self.client_id, payload=json.dumps(result)
        )

    def handle_recall(self, msg):
        if logger_config["show"] == True:
            logger.debug("Do handle_recall")
            print("do handle_recall")
        task_name = msg.payload.decode("utf-8")
        if task_name == "RECALL":
            self.do_recall()

    def start(self):
        if logger_config["show"] == True:
            logger.debug("Do start")
        self.client.connect(self.broker_name, port=1883, keepalive=3600)
        self.client.message_callback_add(
            "dynamicFL/model/all_client", self.handle_model
        )
        self.client.loop_start()
        self.client.subscribe(topic="dynamicFL/model/all_client")
        self.client.subscribe(topic="dynamicFL/data/" + self.client_id)
        self.client.subscribe(topic="dynamicFL/req/" + self.client_id)
        self.client.subscribe(topic="dynamicFL/wait/" + self.client_id)
        self.client.publish(topic="dynamicFL/join", payload=self.client_id)
        print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

        self.client._thread.join()
        print_log("client exits")


class FedAvg_CS_Server(MqttClient):
    def __init__(
        self, client_fl_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311
    ):
        super().__init__(client_fl_id, clean_session, userdata, protocol)

        self.on_connect = self.on_connect_callback
        self.on_message = self.on_message_callback
        self.on_disconnect = self.on_disconnect_callback
        self.on_subscribe = self.on_subscribe_callback

        self.client_dict = {}
        self.client_trainres_dict = {}

        self.NUM_ROUND = server_config["num_rounds"]
        self.NUM_DEVICE = server_config["num_clients"]
        self.time_between_two_round = 1
        self.round_state = "finished"
        self.n_round = 0

        # do for cluster
        self.client_data = {}
        self.cluster_labels_client = {}
        self.selected_client = []
        self.speeds = [340, 585, 296, 214, 676, 550, 439, 332, 440, 583, 885, 295, 429, 609, 585, 931, 674, 227, 929,
                       442, 807, 995, 343, 377, 514, 918, 691, 323, 549, 705]

    # check connect to broker return result code
    def on_connect_callback(self, client, userdata, flags, rc):
        # if logger_config["show"] == True:
        #     logger.info(f"Do on_connect_callback")
        # print_log("Connected with result code " + str(rc))
        pass

    def on_disconnect_callback(self, client, userdata, rc):
        # if logger_config["show"] == True:
        #     logger.info(f"Do on_disconnect_callback")
        #     print_log("Disconnected with result code " + str(rc))
        self.reconnect()

    # handle message receive from client
    def on_message_callback(self, client, userdata, msg):
        # if logger_config["show"] == True:
        #     logger.info(f"Do on_message_callback")
        topic = msg.topic
        if topic == "dynamicFL/join":  # topic is join --> handle_join
            self.handle_join(self, userdata, msg)
        elif "dynamicFL/res" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_res(this_client_id, msg)

    def on_subscribe_callback(self, mosq, obj, mid, granted_qos):
        if logger_config["show"] == True:
            logger.info(f"Do on_subscribe_callback")
        # print_log("Subscribed: " + str(mid) + " " + str(granted_qos))
        pass

    def send_task(self, task_name, client, this_client_id):
        # if logger_config["show"] == True:
        #     logger.info(f"Do send_task")
        # print_log("publish to " + "dynamicFL/req/" + this_client_id)
        self.publish(topic="dynamicFL/req/" + this_client_id, payload=task_name)

    def send_model(self, path, client, this_client_id):
        if logger_config["show"] == True:
            logger.info(f"Do send_model")
        f = open(path, "rb")
        data = f.read()
        f.close()
        self.publish(topic="dynamicFL/model/all_client", payload=data)

    def handle_res(self, this_client_id, msg):
        if logger_config["show"] == True:
            logger.info(f"Do handle_res")
        data = json.loads(msg.payload)
        cmd = data["task"]
        if cmd == "EVA_CONN":
            # print_log(f"{this_client_id} complete task EVA_CONN")
            self.handle_pingres(this_client_id, msg)
        elif cmd == "TRAIN":
            # print_log(f"{this_client_id} complete task TRAIN")
            self.handle_trainres(this_client_id, msg)
        elif cmd == "WRITE_MODEL":
            # print_log(f"{this_client_id} complete task WRITE_MODEL")
            self.handle_update_writemodel(this_client_id, msg)

    def handle_join(self, client, userdata, msg):
        # if logger_config["show"] == True:
        #     logger.info(f"Do handle_join")
        this_client_id = msg.payload.decode("utf-8")
        # print_log("joined from" + " " + this_client_id)
        self.client_dict[this_client_id] = {"state": "joined"}
        self.subscribe(topic="dynamicFL/res/" + this_client_id)

    """
        Do cluster and Client selection
    """

    def ClusterClient_Before(self):
        client_data = (
            self.client_data
        )  # client_data = self.client_data = {client_1: [12,4,23,54, ..., 130]} examples
        distribution = []
        client_ids = list(client_data.keys())
        for client_id, data_distribution in client_data.items():
            distribution.append(data_distribution)
        data_matrix = np.array(distribution)

        cluster_labels = apply_clustering(
            data=data_matrix, method_name="AffinityPropagation"
        )

        num_clusters = len(np.unique(cluster_labels))

        cluster_result = {
            client_ids[i]: cluster_labels[i] for i in range(len(client_ids))
        }

        print_log(f"Clustering result: {cluster_result}")

        return cluster_result

    def ClusterClient_After(self):
        client_data = self.client_trainres_dict

        client_ids = list(client_data.keys())
        distribution = []

        for client_id, model_param in client_data.items():
            flattened_param = np.concatenate(
                [np.array(param).flatten() for param in model_param.values()]
            )
            distribution.append(flattened_param)

        data_matrix = np.array(distribution)

        cluster_labels = apply_clustering(
            data=data_matrix, method_name="AffinityPropagation"
        )

        num_clusters = len(np.unique(cluster_labels))

        cluster_result = {
            client_ids[i]: cluster_labels[i] for i in range(len(client_ids))
        }

        print_log(f"\n \n Clustering result: {cluster_result} \n \n")

        return cluster_result

    # def selection_client(self):
    #     """
    #         self.cluster_labels_client = {"client_0": label, ...}
    #         self.speeds = [340, 585, 296, 214, 676, 550, 439, 332, 440, 583, 885, 295, 429, 609, 585, 931, 674, 227, 929,
    #                    442, 807, 995, 343, 377, 514, 918, 691, 323, 549, 705]
    #     """
    #     self.list_clients = list(self.cluster_labels_client.keys())

    #     self.label_counts = [np.sum(counts) for counts in self.cluster_labels_client.values()]

    #     local_speeds = self.speeds[: len(self.list_clients)]

    #     total_training_time = np.array(self.label_counts) / np.array(local_speeds)

    #     if server_config["selection"]:
    #         if server_config["cluster_mode"]:
    #             num_cluster = len(set(self.cluster_labels_client))
    #             labels = list(set(self.cluster_labels_client))
    #             for i in range(num_cluster):
    #                 cluster_client = [
    #                     index for index, label in enumerate(labels) if label == i
    #                 ]
    #                 self.selected_client += client_selection_algorithm(
    #                     cluster_client, local_speeds, self.label_counts
    #                 )
    #         else:
    #             self.selected_client = client_selection_algorithm(
    #                 [i for i in range(len(self.list_clients))],
    #                 local_speeds,
    #                 self.label_counts,
    #             )
    #     else:
    #         self.selected_client = [i for i in range(len(self.list_clients))]

    #     training_time = np.max([total_training_time[i] for i in self.selected_client])
    #     logger.info(
    #         f"Active with {len(self.selected_client)} clients: {self.selected_client}"
    #     )
    #     logger.info(f"Total training time round = {training_time}")

    def selection_client(self):
        """
            self.cluster_labels_client = {"client_0": label, ...}
            self.speeds = [340, 585, 296, 214, 676, 550, 439, 332, 440, 583, 885, 295, 429, 609, 585, 931, 674, 227, 929,
                        442, 807, 995, 343, 377, 514, 918, 691, 323, 549, 705]
        """
        self.list_clients = list(self.cluster_labels_client.keys())

        self.label_counts = [np.sum(counts) for counts in self.cluster_labels_client.values()]

        local_speeds = self.speeds[: len(self.list_clients)]

        total_training_time = np.array(self.label_counts) / np.array(local_speeds)

        # Khởi tạo `self.selected_client` là một danh sách trống
        # self.selected_client = []

        if server_config["selection"]:
            if server_config["cluster_mode"]:
                num_cluster = len(set(self.cluster_labels_client.values()))  # Sửa: thêm `.values()`
                labels = list(set(self.cluster_labels_client.values()))  # Sửa: thêm `.values()`
                for i in range(num_cluster):
                    cluster_client = [
                        index
                        for index, client in enumerate(self.list_clients)
                        if self.cluster_labels_client[client] == labels[i]
                    ]
                    self.selected_client += client_selection_algorithm(
                        cluster_client, local_speeds, self.label_counts
                    )
            else:
                self.selected_client = client_selection_algorithm(
                    [i for i in range(len(self.list_clients))],
                    local_speeds,
                    self.label_counts,
                )
        else:
            self.selected_client = [i for i in range(len(self.list_clients))]

        training_time = np.max([total_training_time[i] for i in self.selected_client])
        logger.info(
            f"Active with {len(self.selected_client)} clients: {self.selected_client}"
        )
        logger.info(f"Total training time round = {training_time}")


    def handle_pingres(self, this_client_id, msg):
        if logger_config["show"] == True:
            logger.info(f"Do handle_pingres")
        ping_res = json.loads(msg.payload)
        this_client_id = ping_res["client_id"]

        """
            Save the data in client
        """
        # if server_config["point_cluster"] == "before_trainning":
        self.client_data[this_client_id] = ping_res["data"]
        print(
            f"\n \n --------------------------- \n print the data receive to cluster: \n {self.client_data} \n"
        )

        if len(self.client_data) == self.NUM_DEVICE:
            print(f"\n \n Collected all data in clients distribution \n")
            logger.info(
            f"\n \n --------------------------- \n print the data receive to cluster: \n {self.client_data} \n"
            )
            self.cluster_labels_client = self.ClusterClient_Before()

        """
            Only cluster before trainning
            -END-
        """

        if ping_res["packet_loss"] == 0.0:
            # print_log(f"{this_client_id} is a good client")
            state = self.client_dict[this_client_id]["state"]
            # print_log(f"state {this_client_id}: {state}, round: {self.n_round}")
            if state == "joined" or state == "trained":
                self.client_dict[this_client_id]["state"] = "eva_conn_ok"
                count_eva_conn_ok = sum(
                    1
                    for client_info in self.client_dict.values()
                    if client_info["state"] == "eva_conn_ok"
                )
                if count_eva_conn_ok == self.NUM_DEVICE:
                    # print_log("publish to " + "dynamicFL/model/all_client")
                    # check model using to send
                    if self.n_round == 1:
                        # print_log("Initial model in server . . .")
                        if "model_run" not in model_config:
                            model = LSTMModel(
                                model_config["max_features"],
                                model_config["embed_size"],
                                model_config["hidden_size"],
                                model_config["n_layers"],
                            ).to(device)
                            warnings.warn(
                                f"Not input model. Model {client_config['model']} is being used for trainning . . ."
                            )
                        else:
                            model_use = model_config["model_run"]

                            if model_use == "LSTMModel":
                                model = LSTMModel(
                                    model_config["max_features"],
                                    model_config["embed_size"],
                                    model_config["hidden_size"],
                                    model_config["n_layers"],
                                    model_config["num_classes"],
                                ).to(device)
                            elif model_use == "Lenet":
                                model = LeNet(
                                    num_classes=data_config["num_classes"]
                                ).to(device)

                        torch.save(model.state_dict(), "src/parameter/server_model.pt")
                    self.send_model(
                        "src/parameter/server_model.pt", "s", this_client_id
                    )

    def handle_trainres(self, this_client_id, msg):
        if logger_config["show"] == True:
            logger.info("Do handle_trainres")
        payload = json.loads(msg.payload.decode())

        self.client_trainres_dict[this_client_id] = payload["weight"]
        state = self.client_dict[this_client_id]["state"]
        if state == "model_recv":
            self.client_dict[this_client_id]["state"] = "trained"
        print("done train!")

    def handle_update_writemodel(self, this_client_id, msg):
        if logger_config["show"] == True:
            logger.info(f"Do handle_update_writemodel")
        state = self.client_dict[this_client_id]["state"]
        if state == "eva_conn_ok":
            self.client_dict[this_client_id]["state"] = "model_recv"
            self.send_task("TRAIN", self, this_client_id)  # hmm
            count_model_recv = sum(
                1
                for client_info in self.client_dict.values()
                if client_info["state"] == "model_recv"
            )
            if count_model_recv == self.NUM_DEVICE:
                print_log(f"Waiting for training round {self.n_round} from client...")

    def start_round(self):
        if logger_config["show"] == True:
            logger.info(f"Do start_round")
        self.n_round
        self.n_round = self.n_round + 1

        print_log(f"server start round {self.n_round}")
        self.round_state = "started"

        # logger.info("1st: Server send task EVACONN")
        for client_i in self.client_dict:
            self.send_task("EVA_CONN", self, client_i)
        while len(self.client_trainres_dict) != self.NUM_DEVICE:
            time.sleep(1)
        time.sleep(1)

        """
            Save the data in client
        """
        if server_config["point_cluster"] == "after_trainning":
            if len(self.client_trainres_dict) == self.NUM_DEVICE:
                print(f"\n \n Cluster Client by model \n")
                self.cluster_labels_client = self.ClusterClient_After()

        """
            Only cluster before trainning
            -END-
        """

        self.end_round()

    def do_aggregate(self):
        if logger_config["show"] == True:
            logger.info(f"Do do_aggregate")
        print_log("Do aggregate ...")
        self.aggregated_models()

    def handle_next_round_duration(self):
        if logger_config["show"] == True:
            logger.info(f"Do handle_next_round_duration")
        while len(self.client_trainres_dict) < self.NUM_DEVICE:
            time.sleep(1)

    def end_round(self):
        if logger_config["show"] == True:
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
        if logger_config["show"] == True:
            logger.info(f"Do aggregated_models")
        sum_state_dict = OrderedDict()

        print(f"\n \n Do selection client \n \n")
        self.selection_client()
        print(f"\n \n Selection Clients: {self.selected_client} \n \n")

        for client_id, state_dict in self.client_trainres_dict.items():
            # Chỉ xử lý các client được chọn
            if int(client_id.split("_")[1]) in self.selected_client:
                for key, value in state_dict.items():
                    if key in sum_state_dict:
                        sum_state_dict[key] = sum_state_dict[key] + torch.tensor(
                            value, dtype=torch.float32
                        )
                    else:
                        sum_state_dict[key] = torch.tensor(value, dtype=torch.float32)

        # Tổng số client được chọn
        num_models = len(self.selected_client)
        avg_state_dict = OrderedDict(
            (key, value / num_models) for key, value in sum_state_dict.items()
        )

        torch.save(avg_state_dict, "src/parameter/server_model.pt")


        self.client_trainres_dict.clear()
        self.client_data.clear()
        self.cluster_labels_client.clear()
        self.selected_client.clear()


        if model_config['model_run'] == "LSTMModel":
            model_server = LSTMModel(
                model_config["max_features"],
                model_config["embed_size"],
                model_config["hidden_size"],
                model_config["n_layers"],
                model_config["num_classes"],
            ).to(device)
        elif model_config['model_run'] == "Lenet":
            model_server = LeNet(
                num_classes=data_config["num_classes"]
            ).to(device)
        model_server.load_state_dict(
            torch.load("src/parameter/server_model.pt", map_location=device)
        )
        _, testset =get_Dataset(
            datasetname=data_config["dataset"],
            datapath="D:\\Project\\FedCSP\\data\\images",
        )
        logger.info(f'\n \n ---------------Server Testing model--------------- \n \n')
        test_serverloader = DataLoader(testset, batch_size=model_config['batch_size'], shuffle=True)
        testing_model_server(model_input=model_server,
                             testloader=test_serverloader,
                             model_run=model_config["model_run"],
                            num_classes=data_config["num_classes"],
                            epochs=client_config["num_epochs"],
                            batch_size=model_config["batch_size"],)
