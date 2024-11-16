from src.add_config import *
from src.model_install.run_model import *
from src.logging import *
from src.model_install.setup import print_dataset
from src.model_install.handle_data import get_Dataset, split_data
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.metrics import silhouette_score
from src.main.strategies_fl.local_strategy.fedavg import local_fedavg
from src.main.strategies_fl.local_strategy.fedscp import local_fedscp

import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, AffinityPropagation, MeanShift, KMeans
from sklearn.cluster import DBSCAN

def get_dataset():
    # logger.debug("Do get_dataset")
    all_trainset, all_testset = get_Dataset(datasetname=data_config['dataset'],datapath= "D:\\Project\\FedCSP\\data\\images") #include images & DGA

    all_client_trainset = split_data(dataset_use=all_trainset, dataset = data_config['dataset'],
                                data_for_client = data_config['data_for_client_train'], num_classes=data_config['num_classes'],
                                partition=data_config['partition'], data_volume_each_client = data_config['data_volume_each_client'],
                                beta = data_config['beta'], rho = data_config['rho'], num_client = server_config['num_clients'])
    
    all_client_testset = split_data(dataset_use=all_testset, dataset = data_config['dataset'],
                                data_for_client = data_config['data_for_client_test'], num_classes=data_config['num_classes'],
                                partition=data_config['partition'], data_volume_each_client = data_config['data_volume_each_client'],
                                beta = data_config['beta'], rho = data_config['rho'], num_client = server_config['num_clients'])
    
    if logger_config['show'] == True:
        logger.info(all_client_trainset)
        print_dataset(all_client_trainset['client_0'])

    return all_client_trainset, all_client_testset

def server_aggregation(num_clients, num_rounds, round):

    client_res_dict = {}
    sum_parameter = OrderedDict()

    all_client_trainset, all_client_testset = get_dataset()
    
    if server_config['method'] == 'FedAvg':
        client_res_dict = local_fedavg(num_clients = num_clients, all_client_trainset = all_client_trainset,
                                        all_client_testset = all_client_testset, client_res_dict=client_res_dict)

    elif server_config['method'] == 'FedSCP':
        client_res_dict = local_fedscp(num_clients = num_clients, all_client_trainset = all_client_trainset,
                                        all_client_testset = all_client_testset, client_res_dict=client_res_dict)

    # Aggregation step: Average client parameters
    for client_id, parameter in client_res_dict.items():
        for key, value in parameter.items():
            if key in sum_parameter:
                # sum_parameter[key] += torch.tensor(value, dtype=torch.float32)
                sum_parameter[key] += value.clone().detach().float()

            else:
                # sum_parameter[key] = value.clone().detach().to(torch.float32)
                sum_parameter[key] = value.clone().detach().float()


    # Compute average parameters for all clients
    num_models = len(client_res_dict)
    avg_state_dict = OrderedDict((key, value / num_models) for key, value in sum_parameter.items())
    torch.save(avg_state_dict, "src/parameter/local_client.pt")
    client_res_dict.clear()

 
def local_running(num_clients, num_rounds):
    """
    Server call trainning:
        Input:
            n rounds | Strategy | num_clients | 

    Client trainning:
        Input:
            model | batch_size | lr | drop_last | num_epochs
            dataset | num_classes | partitition | data_volume_each_client | beta |rho

    """
    # print(all_client_trainset)

    # start
    for round in range(num_rounds):
        if round == 0:
            print_log(f"Round {round}: \n")
            if logger_config['show'] == True:
                logger.info("Server install model and save to local_client.pt!")
            
            if model_config['model_run'] == 'LSTMModel':
                model = LSTMModel(max_features, embed_size, hidden_size, n_layers, num_classes=data_config['num_classes']).to(device)
                torch.save(model.state_dict(), "src/parameter/local_client.pt")
            elif model_config['model_run'] == 'Lenet':
                model = LeNet(num_classes=data_config['num_classes']).to(device)
                torch.save(model.state_dict(), "src/parameter/local_client.pt")

            server_aggregation(num_clients, num_rounds, round)

            if model_config['model_run'] == 'LSTMModel':
                test_model = LSTMModel(max_features, embed_size, hidden_size, n_layers, num_classes=data_config['num_classes']).to(device)
            elif model_config['model_run'] == 'Lenet':
                test_model = LeNet(num_classes=data_config['num_classes']).to(device)

            _, test_set = get_Dataset(datasetname=data_config['dataset'],datapath= "D:\\Project\\FedCSP\\data\\images")

            test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=True)

            testing_model_server(model_input=test_model, testloader=test_loader, model_run = model_config['model_run'],
                num_classes = data_config['num_classes'], epochs = client_config['num_epochs'],
                batch_size = model_config['batch_size'])
        else:
            print_log(f"Round {round}: \n")
            server_aggregation(num_clients, num_rounds, round)

            if model_config['model_run'] == 'LSTMModel':
                test_model = LSTMModel(max_features, embed_size, hidden_size, n_layers, num_classes=data_config['num_classes']).to(device)
            elif model_config['model_run'] == 'Lenet':
                test_model = LeNet(num_classes=data_config['num_classes']).to(device)
            
            test_model.load_state_dict(torch.load("src/parameter/local_client.pt", map_location=device))

            _, test_set = get_Dataset(datasetname=data_config['dataset'],datapath= "D:\\Project\\FedCSP\\data\\images")

            test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=True)

            testing_model_server(model_input=test_model, testloader=test_loader, model_run = model_config['model_run'],
                num_classes = data_config['num_classes'], epochs = client_config['num_epochs'],
                batch_size = model_config['batch_size'])

# if __name__ == "__main__":
#     local_running()