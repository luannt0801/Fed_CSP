import yaml

path = "config.yaml"
with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

server_config = config['server']
client_config = config['client']
data_config = config['dataset']
model_config = config['model_CNN']