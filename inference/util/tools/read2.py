import json
import yaml

class AC_read:
    def __init__(self):
        self.server_config = read_server("machine_info.yaml")
        self.version = version_read()

def version_read():
    file_path = "__version__.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["AC Bench"]["version"]

def check_for_none(data, path):
    if isinstance(data, dict):
        for key, value in data.items():
            if value == "None":
                raise ValueError(f"空值发现在: {' -> '.join(path + [key])}")
            check_for_none(value, path + [key])
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            check_for_none(item, path + [str(idx)])

def read_server(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            check_for_none(data, [])
            return data
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
        except ValueError as ve:
            print(ve)
            exit(1)