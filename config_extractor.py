from yaml import safe_load
from pprint import pprint

CONFIG_PATH = 'data_storage/config.yaml'


def get(prop: str):
    global CONFIG_PATH
    with open(CONFIG_PATH, 'r') as stream:
        configuration = safe_load(stream)
        # return the value of the config by its propepty
        value = configuration[prop]
        stream.close()
        return value


def show_all() -> None:
    """ Show all the properties and values of the configuration file
    It allows to determine whether the file has been read correctly or any values are missing
    """
    global CONFIG_PATH
    with open(CONFIG_PATH) as stream:
        configuration = safe_load(stream)
        print('='*5+f'CONFIGURATION'+'='*5)
        pprint(configuration)
        stream.close()
