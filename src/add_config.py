import yaml

path = "Fl_mqtt_Config.yaml"
with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

server_config = config['Server']
client_config = config['Client']
data_config = config['Dataset']
model_config = config['Model_CNN']