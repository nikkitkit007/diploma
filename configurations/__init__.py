import yaml
from typing import NamedTuple


config_file = "/home/nikita/Desktop/one-pixel-attack-master/configurations/config.yaml"
CONFIG = yaml.safe_load(open(config_file))


class Datasets(NamedTuple):
    origin = "/home/nikita/Desktop/one-pixel-attack-master/" + CONFIG["input_dir"]
    broken = "/home/nikita/Desktop/one-pixel-attack-master/" + CONFIG["output_dir"]
    test = "/home/nikita/Desktop/one-pixel-attack-master/" + CONFIG["test_output_dir"]
