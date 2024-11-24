import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from src.add_config import *
from src.logging import *
from src.main.strategies_fl.local_strategy.cluster_fuc import (
    apply_clustering,
    do_silhouette_score,
)
from src.model_install.run_model import *

pd.set_option("display.max_rows", None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)


def calculate_labels_client(trainset):  # this is return_sl_cluster

    raw_data_client = []
    reality_label = []

    data_handle = []
    surrogate_labels = []

    proto_data = {}
    proto_label = {}

    for inputs, labels in trainset:
        raw_data_client.append(inputs)
        reality_label.append(labels)

    # print(f"raw_data_client: \n {len(raw_data_client)} \n ----------------------------")
    # print(f"reality_label: \n {len(reality_label)}")

    for idx in range(len(raw_data_client)):
        data = raw_data_client[idx].view(-1)
        # data = torch.cat(data)
        data = data.numpy()
        data_handle.append(data)

    # print(f"len data_handle: {len(data_handle)}")
    # print(f"check 1 element shape: {data_handle[1].shape}")
    # print(f"check 1 element: {data_handle[1]}")

    # optimal_cluster = do_silhouette_score(k_range=11, data_samples=data_handle)
    test = optimal_cluster = 10

    kmeans = KMeans(n_clusters=optimal_cluster, random_state=0)

    surrogate_labels = kmeans.fit_predict(data_handle)

    # print(f"len data_handle: {len(surrogate_labels)}")

    for idx, label in enumerate(surrogate_labels):
        if label not in proto_data:
            proto_data[label] = []
        proto_data[label].append(data_handle[idx])

    for label, data in proto_data.items():
        proto_label[label] = np.mean(data, axis=0)

    # print(f"proto_label: {proto_label}")
    # for key in proto_label.keys():
    # print(f"key proto_label: {key}")

    return proto_label, reality_label


def cluster_by_data(all_client_trainset, all_client_testset, num_clients):
    # client
    client_data_dict = {}  # client_data_dict = {client_id: reality_label}

    # server
    server_df = pd.DataFrame(columns=["Client_id", "c_labels", "c_protos", "s_labels"])
    data_list = []

    for client_id in range(num_clients):
        trainset = all_client_trainset[f"client_{client_id}"]
        testset = all_client_testset[f"client_{client_id}"]

        # print(f"Client_{client_id+1} run")
        proto_label, reality_label = calculate_labels_client(trainset)

        client_data_dict[f"client_{client_id+1}"] = reality_label

        for labels, proto in proto_label.items():
            data_list.append(
                {
                    "Client_id": f"client_{client_id+1}",
                    "c_labels": f"c_{client_id+1}_{labels}",
                    "c_protos": proto,
                    "s_labels": None,
                }
            )

    ### server run

    server_df = pd.DataFrame(data_list)
    # print(f"server_df: \n {server_df}")

    protos_collect = server_df["c_protos"].tolist()
    s_labels = apply_clustering(protos_collect, method_name="AffinityPropagation")
    # print(f"s_labels: {s_labels}")

    server_df["s_labels"] = s_labels
    print_log(f"server_df: \n {server_df}")

    update_labels = {}

    for index, row in server_df.iterrows():
        c_labels = row["c_labels"]
        s_labels = row["s_labels"]
        client_id = row["Client_id"]
        combined_labels = f"{c_labels}_{s_labels}"
        if client_id not in update_labels:
            update_labels[client_id] = []
        update_labels[client_id].append(combined_labels)

        # print(f"Combined labels for {row['Client_id']}: {combined_labels}")

    # for client_id, labels in update_labels.items():
    # print(f"{client_id}: {labels}")

    client_mapping_labels = {}
    for sid, labels in update_labels.items():
        # print(f"labels: {labels}")
        mappings = []
        for idx in range(len(labels)):
            # print(f"labels[idx]: {labels[idx]}")
            # print(f"labels[idx].split[2]: {labels[idx].split("_")[2]}")
            # print(f"labels[idx].split[3]: {labels[idx].split("_")[3]}")
            mappings.append({labels[idx].split("_")[2]: labels[idx].split("_")[3]})

        client_mapping_labels[sid] = mappings
        # mappings.clear()

    # print(f"client_mapping_labels: {client_mapping_labels}")
    # print(f"client_data_dict: {client_data_dict}")

    # for key, value in client_data_dict.items():
    #     print(f"key in client_data_dict: {key}")
    #     print(f"value in client_data_dict: {len(value)}")

    final_mapping = {}
    for client_id, reality_labels in client_data_dict.items():
        # print(f"client_mapping_labels.get({client_id}): {client_mapping_labels.get(client_id)}")
        labels_mapping_for_clients = client_mapping_labels.get(client_id)
        # print(f"labels_mapping_for_clients: {labels_mapping_for_clients}")

        if labels_mapping_for_clients:
            # sc_labels = []
            # for label in reality_labels:
            #     for mapping in labels_mapping_for_clients:
            #         if str(label) in mapping:
            #             print_log(f"str(label): {str(label)}")
            #             print_log(f"mapping: {mapping}")
            #             sc_labels.append(mapping[str(label)])
            #             break
            # print_log(f"sc_labels: {sc_labels}")
            sc_labels = []

            for label in reality_labels:
                mapped_label = None

                for mapping in labels_mapping_for_clients:
                    if str(label.item()) in mapping:
                        mapped_label = mapping[str(label.item())]
                        break

                if mapped_label is None:
                    raise ValueError(f"No mapping found for label {label.item()}")
                else:
                    sc_labels.append(mapped_label)

                # Thêm nhãn mới vào sc_labels
            final_mapping[client_id] = sc_labels

    return final_mapping

    # final_mapping
    #


class CustomDatasetWithUpdatedLabels(Dataset):
    """
    Custom Dataset class, replaces original labels with new labels.
    """

    def __init__(self, data, updated_labels):
        """
        :param data: Input data (eg image)
        :param updated_labels: New labels have been replaced
        """
        self.data = data
        self.updated_labels = updated_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.updated_labels[idx]


def update_trainset_with_new_labels(all_client_trainset, final_mapping):
    updated_client_trainset = {}

    for client_id, trainset in all_client_trainset.items():
        # print_log(f"client_id: {client_id}")
        client_number = int(client_id.split("_")[1])
        client_id = f"client_{client_number + 1}"
        new_labels = final_mapping.get(client_id, [])
        new_labels = torch.tensor([int(label) for label in new_labels])

        raw_data_client = []
        reality_labels = []

        for inputs, labels in trainset:
            raw_data_client.append(inputs)
            reality_labels.append(labels)

        if len(new_labels) != len(reality_labels):
            raise ValueError(
                f"The number of new labels does not match the number of original client {client_id} labels."
            )

        updated_trainset = CustomDatasetWithUpdatedLabels(raw_data_client, new_labels)

        updated_client_trainset[client_id] = updated_trainset

    return updated_client_trainset


def local_fedscp(num_clients, all_client_trainset, all_client_testset, client_res_dict):
    """
    return client_res_dict
    """
    finnal_mapping = cluster_by_data(
        all_client_trainset, all_client_testset, num_clients
    )

    # for key, value in finnal_mapping.items():
    #     print(f"key in final_mapping: {key}")
    #     print(f"value in final_mapping: {len(value)}")

    updata_all_trainset_client = update_trainset_with_new_labels(
        all_client_trainset=all_client_trainset, final_mapping=finnal_mapping
    )

    for client_id in range(num_clients):
        # logger.info(F'Client {client_id+1} is trainning . . .')
        # print_log(client_id+1)

        trainset = updata_all_trainset_client[f"client_{client_id+1}"]
        testset = all_client_testset[f"client_{client_id}"]

        trainloader = DataLoader(
            trainset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            drop_last=data_config["drop_last"],
        )
        testloader = DataLoader(
            testset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            drop_last=data_config["drop_last"],
        )

        parameter_client = trainning_model(
            trainloader,
            testloader,
            model_run=model_config["model_run"],
            num_classes=data_config["num_classes"],
            epochs=client_config["num_epochs"],
            batch_size=model_config["batch_size"],
        )

        client_res_dict[f"{client_id}"] = parameter_client

    return client_res_dict
