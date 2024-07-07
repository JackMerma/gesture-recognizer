import cv2
import sys
import os
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def load_data(path, categories, img_width, img_height):

    images = []
    labels = []
    
    for category in range(len(categories)):
        category_dir = os.path.join(path, categories[category])

        if not os.path.exists(category_dir):
            raise Exception("You need to load data. Use '--help' for more information.")

        # Processing files
        for file in os.listdir(category_dir):
            filepath = os.path.join(category_dir, file)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_width, img_height))
            images.append(np.array(image))
            labels.append(category)

    return images, labels


def get_model(shape, categories):

    model = Sequential([
        Conv2D(
            32, (2, 2), activation="relu", input_shape=shape
            ),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (2, 2), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (2, 2), activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(len(categories), activation="sigmoid")
        ])


    model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )

    return model


def get_prediction(prediction):

    if prediction[0] > prediction[1]:
        return 0, prediction[0]

    return 1, prediction[1]


def predict(frame, model):

    image = np.expand_dims(frame, axis=(0, -1))
    prediction = model.predict(image)
    
    class_prediction, acc = get_prediction(prediction[0])
    print(class_prediction)
    print(prediction)

