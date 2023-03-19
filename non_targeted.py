import argparse
import logging
import os
import sys

import imageio
import numpy as np
import yaml
from IPython import embed

from one_pixel_attack.common import get_image_array, get_probability_for_class, get_perturbed_images
from one_pixel_attack.differential_evolution import init_population, gen_children
from models.base import get_model_from_name

config_file = "config.yaml"
CONFIG = yaml.safe_load(open(config_file))
model = get_model_from_name(CONFIG["model"])
CONFIG["img_x"], CONFIG["img_y"], CONFIG["img_channels"] = model.input_size

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def fitness_function(prediction, true_class=None):
    """
    For non-targeted attacks, the fitness function is the negative probability of true class
    """
    return -get_probability_for_class(prediction, true_class)


def get_fit_population(fathers, children, fathers_predictions, children_predictions, true_class):
    final_population = list()
    for i in range(len(fathers_predictions)):
        father_fitness = fitness_function(fathers_predictions[i], true_class)
        child_fitness = fitness_function(children_predictions[i], true_class)
        if father_fitness < child_fitness:
            final_population.append(children[i])
        else:
            final_population.append(fathers[i])
    return np.array(final_population)


def find_adversary_image(image, model, name):
    original_predictions = model.predict(np.copy(image))
    true_label = original_predictions[0][0][1]
    true_label_probability = original_predictions[0][0][2]
    # logging.info("True label: {}, Probability: {}".format(true_label, true_label_probability))
    population = init_population(CONFIG)
    for i in range(CONFIG["num_iterations"]):
        # logging.info("Iteration: {}".format(i))
        perturbed_images = get_perturbed_images(image, population)
        perturbed_predictions = model.predict(np.copy(perturbed_images), top=model.num_classes)

        true_class_probabilities = list(map(lambda p: get_probability_for_class(p, true_label), perturbed_predictions))
        # logging.info("Probabilites for true class: Min={}, Max={}".format(min(true_class_probabilities),
        #                                                                   max(true_class_probabilities)))
        if max(true_class_probabilities) < CONFIG["confidence"]:
            logging.info(os.path.abspath(CONFIG["output_dir"]+'{}'.format(name)))
            return imageio.imwrite(CONFIG["output_dir"]+'{}'.format(name),
                            perturbed_images[true_class_probabilities.index(min(true_class_probabilities))])

        population_children = gen_children(population, CONFIG)
        perturbed_images_children = get_perturbed_images(image, population_children)
        perturbed_predictions_children = model.predict(np.copy(perturbed_images_children), top=model.num_classes)

        population = get_fit_population(population, population_children,
                                        perturbed_predictions,
                                        perturbed_predictions_children,
                                        true_class=true_label)


# def start(input_image: str) -> None:
#     # sys.stdout = open("logs/"+str(os.getpid()) + ".out", "a")
#     # sys.stderr = open("logs/"+str(os.getpid()) + "_error.out", "a")
#     image_arr = get_image_array(input_image, config=CONFIG)
#     name = input_image.split('/')[-1]
#     return find_adversary_image(image_arr, model, name)
#
#
# if __name__ == "__main__":
#     config_file = "config.yaml"
#     CONFIG = yaml.safe_load(open(config_file))
#     model = get_model_from_name(CONFIG["model"])
#     CONFIG["img_x"], CONFIG["img_y"], CONFIG["img_channels"] = model.input_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file', help='config file')
    parser.add_argument('--input', '-i', dest='input_image', help='input image file')
    args = parser.parse_args()

    CONFIG = yaml.safe_load(open(args.config_file))
    model = get_model_from_name(CONFIG["model"])
    CONFIG["img_x"], CONFIG["img_y"], CONFIG["img_channels"] = model.input_size
    print('\n------->>>>', args.input_image)
    image_arr = get_image_array(args.input_image, config=CONFIG)
    name = args.input_image.split('/')[-1]

    find_adversary_image(image_arr, model, name)
