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
python client.py 1
```