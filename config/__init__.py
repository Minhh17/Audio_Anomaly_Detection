from .default import get_cfg_defaults
from .config_manager import ConfigManager

# __all__ = ['ConfigManager']

def update_config(cfg, yaml_file):
    '''
        This is a function that updates config from yaml file
    '''
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    return cfg
