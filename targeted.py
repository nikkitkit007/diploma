import argparse
import logging

import imageio
import numpy as np
import yaml
from IPython import embed

from common import get_image_array, get_probability_for_class, get_perturbed_images
from differential_evolution import init_population, gen_children
from models.base import get_model_from_name

CONFIG = None
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def fitness_function(prediction, target_class):
    """
    For targeted attacks, the fitness function is the probability of target class
    """
    return get_probability_for_class(prediction, target_class)


def get_fit_population(fathers, children, fathers_predictions, children_predictions, target_class):
    final_population = list()
    for i in range(len(fathers_predictions)):
        father_fitness = fitness_function(fathers_predictions[i], target_class)
        child_fitness = fitness_function(children_predictions[i], target_class)
        if father_fitness < child_fitness:
            final_population.append(children[i])
        else:
            final_population.append(fathers[i])
    return np.array(final_population)


def find_adversary_image(image, model, target_label):
    original_predictions = model.predict(np.copy(image))
    true_label = original_predictions[0][0][1]
    true_label_probability = original_predictions[0][0][2]
    logging.info("True label: {}, Probability: {}".format(true_label, true_label_probability))

    target_label_probability = get_probability_for_class(original_predictions[0], target_label)
    logging.info("Target label: {}, Probability: {}".format(target_label, target_label_probability))

    imageio.imwrite('output/original.jpg', image[0])

    population = init_population(CONFIG)
    for i in range(CONFIG["num_iterations"]):
        logging.info("Iteration: {}".format(i))
        perturbed_images = get_perturbed_images(image, population)
        perturbed_predictions = model.predict(np.copy(perturbed_images), top=model.num_classes)

        target_class_probabilities = map(lambda p: get_probability_for_class(p, target_label), perturbed_predictions)
        logging.info("Probabilites for target class: Min={}, Max={}".format(min(target_class_probabilities),
                                                                            max(target_class_probabilities)))
        if i % 10 == 0:
            imageio.imwrite('output/{}.jpg'.format(i),
                            perturbed_images[target_class_probabilities.index(max(target_class_probabilities))])

        population_children = gen_children(population, CONFIG)
        perturbed_images_children = get_perturbed_images(image, population_children)
        perturbed_predictions_children = model.predict(np.copy(perturbed_images_children), top=model.num_classes)

        population = get_fit_population(population, population_children,
                                        perturbed_predictions,
                                        perturbed_predictions_children,
                                        target_class=target_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file', help='config file')
    parser.add_argument('--input', '-i', dest='input_image', help='input image file')
    parser.add_argument('--target', '-t', dest='target_class', help='target class name')
    args = parser.parse_args()

    CONFIG = yaml.safe_load(open(args.config_file))
    model = get_model_from_name(CONFIG["model"])
    target_label = args.target_class
    CONFIG["img_x"], CONFIG["img_y"], CONFIG["img_channels"] = model.input_size
    image_arr = get_image_array(args.input_image, config=CONFIG)
    find_adversary_image(image_arr, model, target_label)
