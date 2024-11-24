from src.utils import *
from src.add_config import *
from src.model_install.handle_data import *
from src.logging import *

from src.model_install.run_model import trainning_model # for another dataset
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client as mqtt
import json
import torch
import time
import threading
import pandas as pd
from collections import OrderedDict
from paho.mqtt.client import Client as MqttClient, MQTTv311
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.cluster import AffinityPropagation

from src.main.strategies_fl.FedAvg import FedAvg_Server, FedAvg_Client

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedCluster_Server(FedAvg_Server):
    def __init__(self, client_fl_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311):
        super().__init__(client_fl_id, clean_session, userdata, protocol)
        
        self.client_prototypes = {}
        self.prototypes = []
        self.state = ""
        self.server_labels = {}
        self.proto_df = pd.DataFrame(columns=['label_Server', 'client_id', 'label_client', 'protos'])

    """
        Server connect to all client, send model.
        In a round:
            Client collect data_round -> do unsupervised learning -> return a surrogate labels
            Client calculates prototype for each surrogate labels
            -> upload to servers

            Server cluster prototypes of surrogate labels
            -> Send surrogate labels ending to client = num_classes

            Client do train
    """

    def start_round(self):
        logger.info(f"Do start_round")
        self.n_round
        self.n_round = self.n_round + 1

        print_log(f"server start round {self.n_round}")
        self.round_state = "started"

        # logger.info("1st: Server send task EVACONN")
        for client_i in self.client_dict:
            self.send_task("EVA_CONN", self, client_i)

        while (len(self.client_trainres_dict) != self.NUM_DEVICE):
            self.do_label_synthesis()

        while (len(self.client_trainres_dict) != self.NUM_DEVICE):
            time.sleep(1)
        time.sleep(1)
        self.end_round()
    
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

    def do_label_synthesis(self):

        """
            self.client_prototypes = {idx: self.psl}
            self.psl = {sl: p}

            return server_labels = {label_in_server: {idx:{sl:p}}}
        """

        for idx, psl in self.client_prototypes.items():
            for sl, p in psl.items():
                self.prototypes.append(p)
            
        model = AffinityPropagation(random_state=2024)
        label_in_server = model.fit(self.prototypes)

        unique_labels = np.unique(label_in_server)

        for client_id, protos in self.client_prototypes.items():
            for label in unique_labels:
                pass
        
        for proto in self.prototypes:
            pass
        for label in unique_labels:
            self.server_labels[label] = self.prototypes[label_in_server == label]
        
            domains_with_type = [self.server_labels[label], client_id, protos['p'] self.prototypes[label_in_server == label],]
            appending_df = pd.DataFrame(domains_with_type, columns=['label_Server', 'client_id', 'label_client', 'protos'])
            my_df = pd.concat([my_df, appending_df], ignore_index=True)

        """
            server_labels = {labels in server : proto}
        """    

            

        # for idx, label in enumerate(label_in_server):
        #     sl = next(iter(self.client_prototypes[idx].keys()))  # Lấy surrogate label đầu tiên từ client_prototypes
        #     if label not in self.server_labels:
        #         self.server_labels[label] = {}
        #     self.server_labels[label][idx] = {sl: self.prototypes[idx]}

        return self.server_labels




        for this_client_id in self.client_dict:  # Lặp qua từng client trong client_dict
            self.send_task("TRAIN", self, this_client_id)  # Gửi nhiệm vụ "TRAIN"

        count_model_recv = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "model_recv")
        if count_model_recv == self.NUM_DEVICE:
            print_log(f"Waiting for training round {self.n_round} from client...")


    def handle_update_writemodel(self, this_client_id, msg):
        logger.info(f"Do handle_update_writemodel")
        self.state = self.client_dict[this_client_id]["state"]
        if self.state  == "eva_conn_ok":
            self.client_dict[this_client_id]["state"] = "model_recv"
        data = json.loads(msg.payload)
        self.client_prototypes['this_client_id'] = data['prototype_surrogate_label']

class FedCluster_Client():

    def get_dataset(self):
        logger.debug("Do get_dataset")
        all_trainset, all_testset = get_Dataset(datasetname=client_config['dataset'],datapath= "D:\\Project\\FedCSP\\data\\images") #include images & DGA

        all_client_trainset = split_data(dataset_use=all_trainset, dataset = client_config['dataset'],
                                  data_for_client = client_config['data_for_client_train'], num_classes=client_config['num_classes'],
                                  partition=client_config['partition'], data_volume_each_client = client_config['data_volume_each_client'],
                                  beta = client_config['beta'], rho = client_config['rho'], num_client = server_config['num_clients'])
        
        all_client_testset = split_data(dataset_use=all_testset, dataset = client_config['dataset'],
                                  data_for_client = client_config['data_for_client_test'], num_classes=client_config['num_classes'],
                                  partition=client_config['partition'], data_volume_each_client = client_config['data_volume_each_client'],
                                  beta = client_config['beta'], rho = client_config['rho'], num_client = server_config['num_clients'])

        # debug data in each client

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
        trainloader =  DataLoader(trainset, batch_size=client_config['batch_size'], shuffle=True,drop_last=client_config['drop_last'])

        trainset = all_client_testset[self.client_id]
        testloader =  DataLoader(trainset, batch_size=client_config['batch_size'], shuffle=True, drop_last=client_config['drop_last'])

        # Assuming client_dataloader is your DataLoader for the client
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

    def create_surogate_labels(self):
        
        """
            {client_id: prototype_surrogate_label}
            prototype_surrogate_label = {labels: prototype}

            in server:
                prototype (client_id, labels)
        """

        trainloader, testloader = self.get_dataset()
        datset = ConcatDataset([trainloader.dataset, testloader.dataset])
        dataloader = DataLoader(datset, batch_size=trainloader.batch_size, shuffle=True)

        model = AffinityPropagation(random_state=2024)
        raw_data = []
        for inputs, _ in dataloader:
            raw_data.append(inputs.numpy()) # .detach().cpu().numpy() if use GPU
        
        labels = model.fit(raw_data)
        unique_labels = np.unique(labels)
        logger.debug(f"All {unique_labels}")

        prototype_surrogate_label = {}
        
        for label in unique_labels:
            inputs_for_label = raw_data[labels == label]
            prototype_surrogate_label[label] = np.mean(inputs_for_label, axis=0)

        for label, avg in prototype_surrogate_label.items():
            logger.debug(f"Average for label {label}: {avg}")

        return prototype_surrogate_label
        
    def handle_model(self, client, userdata, msg):
        logger.debug("Do handle_model")
        print_log("receive model from Server")
        with open("src/parameter/client_model.pt", "wb") as f:
            f.write(msg.payload)
        print_log("done write model")

        prototype_surrogate_label = self.create_surogate_labels()

        result = {
            "client_id": self.client_id,
            "task": "WRITE_MODEL",
            "prototype_surrogate_label": prototype_surrogate_label
        }                                                                                         
        self.client.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))

    


