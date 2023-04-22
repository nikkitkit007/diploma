import yaml

config_file = "/home/nikita/Desktop/one-pixel-attack-master/configurations/config.yaml"
CONFIG = yaml.safe_load(open(config_file))

origin_dataset_path = "/home/nikita/Desktop/one-pixel-attack-master/"+CONFIG["input_dir"]
broken_dataset_path = "/home/nikita/Desktop/one-pixel-attack-master/"+CONFIG["output_dir"]
test_dataset_path = "/home/nikita/Desktop/one-pixel-attack-master/"+CONFIG["test_output_dir"]
