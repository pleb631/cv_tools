import os.path as osp

import yaml

here = osp.dirname(osp.abspath(__file__))


def update_dict(target_dict, new_dict):
    for key, value in new_dict.items():
        if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value)
        else:
            target_dict[key] = value


# -----------------------------------------------------------------------------


def get_default_config():
    config_file = osp.join(here, "default_config.yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config




def get_config(config_file_or_yaml=None, config_from_args=None):
    # 1. default config
    config = get_default_config()

    # 2. specified as file or yaml
    if config_file_or_yaml is not None:
        config_from_yaml = yaml.safe_load(config_file_or_yaml)
        if not isinstance(config_from_yaml, dict):
            with open(config_from_yaml) as f:
                config_from_yaml = yaml.safe_load(f)
        update_dict(
            config, config_from_yaml)

    # 3. command line argument or specified config file
    if config_from_args is not None:
        update_dict(
            config, config_from_args)

    return config
