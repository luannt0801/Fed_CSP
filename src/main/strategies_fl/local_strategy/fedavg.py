from src.add_config import *
from src.model_install.run_model import *
from src.logging import *
from torch.utils.data import DataLoader



def trainning_fedavg(trainloader, testloader):
    parameter_client = trainning_model(trainloader, testloader, model_run = model_config['model_run'],
                                        num_classes = data_config['num_classes'], epochs = client_config['num_epochs'],
                                        batch_size = model_config['batch_size'])
    return parameter_client

def local_fedavg(num_clients, all_client_trainset, all_client_testset, client_res_dict):
    for client_id in range(num_clients):
        logger.info(F'Client {client_id+1} is trainning . . .')
        print_log(client_id+1)

        trainset = all_client_trainset[f'client_{client_id}']
        testset = all_client_testset[f'client_{client_id}']

        trainloader = DataLoader(trainset, batch_size=model_config['batch_size'], shuffle=True,drop_last=data_config['drop_last'])
        testloader = DataLoader(testset, batch_size=model_config['batch_size'], shuffle=True,drop_last=data_config['drop_last'])
        parameter_client = trainning_fedavg(trainloader=trainloader, testloader=testloader)
        client_res_dict[f'{client_id}'] = parameter_client
    return client_res_dict