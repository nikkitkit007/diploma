import yaml

config_file = "configurations/config.yaml"
CONFIG = yaml.safe_load(open(config_file))

origin_dataset_path = "/home/nikita/Desktop/one-pixel-attack-master/"+CONFIG["input_dir"]
broken_dataset_path = "/home/nikita/Desktop/one-pixel-attack-master/"+CONFIG["output_dir"]
