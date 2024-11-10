from src.add_config import *
from src.model_install.run_model import *
from src.logging import *
from src.model_install.handle_data import get_Dataset, split_data

from torch.utils.data import DataLoader
from collections import OrderedDict

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

    # # debug data in each client
    # print_log(f"{client_id}: \n", color_='red')

    # print_log("show client_id is: ", client_id)
    # print_log("show debug all client trainset: ",all_client_trainset)
    """
    trainset_client = all_client_trainset[client_id]
    trainset = all_client_trainset[client_id]
    """
    # logger.debug(f"Train data in {client_id} :")
    # logger.debug(trainset_client)
    # logger.debug("\n")
    
    # test_client = all_client_testset[client_id]
    # logger.debug(f"Test data in {client_id} :")
    # logger.debug(test_client)
    # logger.debug("\n")
    """
    trainloader =  DataLoader(trainset, batch_size=client_config['batch_size'], shuffle=True,drop_last=client_config['drop_last'])

    trainset = all_client_testset[client_id]
    testloader =  DataLoader(trainset, batch_size=client_config['batch_size'], shuffle=True, drop_last=client_config['drop_last'])
    """
    
    # # Assuming client_dataloader is your DataLoader for the client
    # all_labels = []

    # for _, labels in trainloader:
    #     all_labels.extend(labels.tolist())  # Collect all labels into a list

    # # Convert list to tensor and get unique labels with counts
    # all_labels_tensor = torch.tensor(all_labels)
    # labels, counts = torch.unique(all_labels_tensor, return_counts=True)

    # # Print the labels and their counts for the client
    # print(f"{client_id} - Labels and their counts in DataLoader:")
    # for label, count in zip(labels, counts):
    #     print(f"Label {label.item()}: {count.item()} samples")

    # print(f"DataLoader for client {client_id} is ready.")

    # logger.debug(f"Number of clients in all_client_trainset: {len(all_client_trainset)}")
    # logger.debug(f"Expected number of clients: {server_config['num_clients']}")


    return all_client_trainset, all_client_testset

def server_aggregation(num_clients, num_rounds):

    client_res_dict = {}
    sum_parameter = OrderedDict()

    all_client_trainset, all_client_testset = get_dataset()
    
    # print(all_client_trainset)

    # start
    logger.debug(f"{num_clients}")

    for round in range(num_rounds):
        print_log(f"Round {round}: \n")
        for client_id in range(num_clients):
            logger.info(F'Client {client_id+1} is trainning . . .')
            print_log(client_id+1)

            trainset = all_client_trainset[f'client_{client_id+1}']
            testset = all_client_testset[f'client_{client_id+1}']

            trainloader = DataLoader(trainset, batch_size=model_config['batch_size'], shuffle=True,drop_last=data_config['drop_last'])
            testloader = DataLoader(testset, batch_size=model_config['batch_size'], shuffle=True,drop_last=data_config['drop_last'])

            parameter_client = trainning_model(trainloader, testloader, model_use = model_config['model'], num_classes = data_config['num_classes'],
                                epochs = client_config['num_epochs'], batch_size = model_config['batch_size'])
            client_res_dict[f'{client_id+1}'] = parameter_client
        # do aggregation
        for client_id, parameter in client_res_dict.items():
            for key, value in parameter.items():
                    if key in sum_parameter:
                        sum_parameter[key] = sum_parameter[key] + torch.tensor(value, dtype=torch.float32)
                    else:
                        sum_parameter[key] = torch.tensor(value, dtype=torch.float32)
            num_models = len(client_res_dict)
            avg_state_dict = OrderedDict((key, value / num_models) for key, value in sum_parameter.items())
            torch.save(avg_state_dict, "src/parameter/server_model.pt")
            client_res_dict.clear()
    # end

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

    server_aggregation(num_clients, num_rounds)


# if __name__ == "__main__":
#     local_running()