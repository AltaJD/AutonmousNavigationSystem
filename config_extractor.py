from yaml import safe_load

CONFIG_PATH = 'data_storage/config.yaml'


def get(prop: str):
    global CONFIG_PATH
    with open(CONFIG_PATH, 'r') as stream:
        configuration = safe_load(stream)
        # return the value of the config by its propepty
        return configuration[prop]
