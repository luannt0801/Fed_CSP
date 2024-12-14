<!---
git add . ':(exclude)data/*'
git add -- . ':(exclude)data'
-->

# Federated Learning Client and Selection Prototype

Running in Server side

```
python server.py --strategy FedAvg --num_rounds 3 --num_clients 1 --dataset Cifar10 --num_classes 10 --partition iid_equal_size --data_volume_each_client equal
```

Running in Client side

```
python client.py --ID 1 --host 192.168.1.119 --strategy FedAvg --model LSTMModel --epochs 1
```

-------
Build
1. server
```
docker build -t myproject:server --build-arg ENVIRONMENT=server .
```

2. client
```
docker build -t myproject:client --build-arg ENVIRONMENT=client --build-arg CLIENT_ID=1 .

```

-------
Run
1. server
```
docker run -it --rm myproject:server
```

2. client
```
docker run -it --rm myproject:client
```

hehe