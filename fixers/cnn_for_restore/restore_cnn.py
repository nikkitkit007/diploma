import numpy as np
import cv2
import os

from keras.models import Model
from keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, Flatten, concatenate, Input
from sklearn.model_selection import train_test_split

from utils.img_worker import get_image_names, calc_dist

from configurations import test_dataset_path, origin_dataset_path


num_channels = 3
num_epochs = 100
batch_size = 10


def compare(img1, img2):
    diff = cv2.subtract(img1, img2)
    return diff


def create_nn_model() -> Model:

    # Создание архитектуры сети
    input_image = Input(shape=(32, 32, num_channels), name='input_image')
    input_coords = Input(shape=(2,), name='input_coords')

    # Сверточные слои
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_image)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Полносвязные слои
    flatten = Flatten()(pool2)
    dense1 = Dense(128, activation='relu')(flatten)

    # Конкатенация координат искаженного пикселя
    concat = concatenate([dense1, input_coords])

    # Выходной слой
    output = Dense(3, activation='linear', name='output')(concat)

    # Создание и компиляция модели
    model = Model(inputs=[input_image, input_coords], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model


def lean_nn_model(images: list, coords: list, pixel_values: list, model: Model):
    # Обучение модели
    images = np.array(images)
    coords = np.array(coords)
    pixel_values = np.array(pixel_values)

    # Разделение датасета на обучающую и тестовую выборки
    train_images, test_images, train_coords, test_coords, train_pixel_values, test_pixel_values = train_test_split(images, coords, pixel_values, test_size=0.2, random_state=42)

    # Нормализация изображений
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    history = model.fit(
        {'input_image': train_images, 'input_coords': train_coords},
        {'output': train_pixel_values},
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=({'input_image': test_images, 'input_coords': test_coords}, {'output': test_pixel_values})
    )

    # Оценка модели
    model.evaluate({'input_image': test_images, 'input_coords': test_coords}, {'output': test_pixel_values})

    predicted_pixel_values = model.predict({'input_image': test_images, 'input_coords': test_coords})

    dist = []
    for i, pr in enumerate(predicted_pixel_values):
        dist.append(calc_dist(pr, test_pixel_values[i]))
    print(sum(dist)/len(dist))


if __name__ == '__main__':
    img_names = get_image_names(test_dataset_path)
    images = []
    coords = []
    pixel_values = []
    for img_name in img_names:
        img_origin = cv2.imread(origin_dataset_path+img_name)
        img_broken = cv2.imread(test_dataset_path+img_name)
        compared_pictures = compare(img_origin, img_broken)
        x, y = np.unravel_index(compared_pictures.argmax(), compared_pictures.shape)[:2]

        images.append(img_broken)
        coords.append((x, y))
        pixel_values.append((img_origin[x, y]))

    # print(images)
    # print(coords)
    # print(pixel_values)

    model = create_nn_model()
    lean_nn_model(images, coords, pixel_values, model)

