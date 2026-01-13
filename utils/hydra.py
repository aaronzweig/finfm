import sys
from hydra import initialize, compose
from omegaconf import OmegaConf

def load_config(config_name="config", overrides=None):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg