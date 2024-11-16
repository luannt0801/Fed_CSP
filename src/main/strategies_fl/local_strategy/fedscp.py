from src.add_config import *
from src.model_install.run_model import *
from src.logging import *
from src.model_install.setup import print_dataset
from src.model_install.handle_data import get_Dataset, split_data
from sklearn.metrics import silhouette_score
from src.main.strategies_fl.local_strategy.fedavg import local_fedavg
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, AffinityPropagation, MeanShift, KMeans
from sklearn.cluster import DBSCAN

def apply_clustering(data, method_name='KMeans', n_clusters=10):
    if method_name == 'KMeans':
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=2024, batch_size=100)
    elif method_name == 'AffinityPropagation':
        model = AffinityPropagation(random_state=2024)
    elif method_name == 'MeanShift':
        model = MeanShift()
    elif method_name == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Clustering method {method_name} not supported!")

    labels = model.fit_predict(data)
    return labels

def calculate_proto(cluster_data):
    return torch.mean(cluster_data, dim=0)

def do_silhouette_score (k_range, data_samples):
    scores = []
    
    for k in range(10, k_range):
        print_log(f"this k={k} is calculating!")
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data_samples)
        
        if len(set(model.labels_)) > 1: 
            score = silhouette_score(data_samples, model.labels_)
            print_log(f"Print the score = {score}")
            scores.append(score)
        else:
            scores.append(float('-inf'))
    # Chọn k với silhouette score cao nhất
    best_k = scores.index(max(scores)) + 2

    return best_k

def elbow_method(data, max_k=30):
    """
    Elbow method.
    """
    distortions = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    # Vẽ đồ thị Elbow Method
    plt.plot(K_range, distortions, marker='o')
    plt.xlabel('Số cụm (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()
    
    # Trả về số cụm tối ưu (k) dựa trên Elbow method
    optimal_k = np.argmin(np.diff(distortions)) + 2  # Chọn k tại điểm gãy
    return optimal_k

def return_sl_client(dataloader):
    """
    use afinity propogation to take surrogate_label -> upload to server
    server use dbscan, kmeans to -> real label
    update -> client -> cluster
    """
    if logger_config['show'] == True:
        logger.info(f"\n calculate surrogate_labels -> proto_labels")

    surrogate_labels = []
    client_sl = {}  # {sl: data}
    data_samples = []

    for batch in dataloader:
        inputs, _ = batch 
        inputs = inputs.view(inputs.size(0), -1) # change data 4 dimensions to 1 dimensions
        data_samples.append(inputs)
    
    data_samples = torch.cat(data_samples, dim=0)

    # for AffinityPropagation
    # surrogate_labels = apply_clustering(data_samples, method_name='AffinityPropagation')
    data_samples = data_samples.numpy()

    # optimal_k = elbow_method(data_samples)
    # print(f"Best number of cluster (k): {optimal_k}")

    optimal_k = do_silhouette_score(k_range=16, data_samples=data_samples)

    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    surrogate_labels = kmeans.fit_predict(data_samples)

    for idx, label in enumerate(surrogate_labels):
        if label not in client_sl:
            client_sl[label] = []
        client_sl[label].append(data_samples[idx])
    
    for label in client_sl:
        # client_sl[label] = torch.stack(client_sl[label])
        client_sl[label] = torch.tensor(client_sl[label], dtype=torch.float32)

    proto_labels = {label: calculate_proto(data) for label, data in client_sl.items()}

    return surrogate_labels, proto_labels

def trainning_fedscp(trainloader, testloader):
    parameter_client = trainning_model(trainloader, testloader, model_run = model_config['model_run'],
                                        num_classes = data_config['num_classes'], epochs = client_config['num_epochs'],
                                        batch_size = model_config['batch_size'])
    return parameter_client

def local_fedscp(num_clients, all_client_trainset, all_client_testset, client_res_dict):
    sl = {} # sl = {client: [sl1, sl2, ...]}
    sl_data = {} # s1_data = {sli: [data_1, data_2. ...]}

    #in server
    server_proto_data = []
    for client_id in range(num_clients):
        logger.info(f'\n Client {client_id+1} is clustering . . .')

        trainset = all_client_trainset[f'client_{client_id}']
        testset = all_client_testset[f'client_{client_id}']

        trainloader = DataLoader(trainset, batch_size=model_config['batch_size'], shuffle=True,drop_last=data_config['drop_last'])
        testloader = DataLoader(testset, batch_size=model_config['batch_size'], shuffle=True,drop_last=data_config['drop_last'])

        sl[f'client_{client_id+1}'], sl_data[f'client_{client_id+1}'] = return_sl_client(trainloader)

        if logger_config['show'] == True:
            logger.info(f"show sl labels in {client_id+1}: \n {np.unique(sl[f'client_{client_id+1}'])} \n")
            logger.info(f"show sl data in each sl label in {client_id+1}: \n {sl[f'client_{client_id+1}']} \n")
        
        # server proto collect
        server_proto_data.extend(sl_data[f'client_{client_id+1}'].values())
    
    server_proto_data_tensor = torch.stack(server_proto_data)
    
    server_sl_labels = apply_clustering(server_proto_data_tensor, method_name='AffinityPropagation')

    server_proto_clusters = {}

    for label, proto in zip(server_sl_labels, server_proto_data):
        if label not in server_proto_clusters:
            server_proto_clusters[label] = []
        server_proto_clusters[label].append(proto)
    
    server_proto_representatives = {label: calculate_proto(torch.stack(protos)) for label, protos in server_proto_clusters.items()}

    for client_id in range(num_clients):
        logger.info(f'\n Client {client_id+1} is trainning . . .')
        client_trainset = all_client_trainset[f'client_{client_id}']
        client_trainloader = DataLoader(client_trainset, batch_size=model_config['batch_size'], shuffle=True, drop_last=data_config['drop_last'])

        client_sl_labels, client_protos = return_sl_client(client_trainloader)
        
        
        aligned_data = {label: [] for label in set(server_sl_labels)}

        for data, sl_label in zip(client_trainloader, client_sl_labels):
            try:
                server_label = server_sl_labels[sl_label]  
                aligned_data[server_label].append(data)
            except IndexError:
                print(f"Warning: No corresponding server label found for surrogate label {sl_label}")

        
        # Perform local training after aligning data
        parameter_client = local_fedavg(aligned_data, testloader,)
        client_res_dict[f'{client_id}'] = parameter_client

        return client_res_dict