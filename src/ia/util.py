import cv2
import sys
import os
import tensorflow as tf
import numpy as np
from config import *

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
        Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=shape),
        BatchNormalization(),

        Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        Dropout(0.2),
        BatchNormalization(),

        Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        BatchNormalization(),

        Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        Dropout(0.2),
        BatchNormalization(),

        Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        BatchNormalization(),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="sigmoid")
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


def put_label(label, percent, frame):

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    font_thickness = 3
    font_color = (229, 232, 232)

    # Getting text shape
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Calculating position
    x = (frame.shape[1] - text_width) // 2
    y = frame.shape[0] - baseline

    cv2.putText(frame, label, (x, y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)


def predict(base_frame, frame, model):

    image = np.expand_dims(base_frame, axis=(0, -1))
    prediction = model.predict(image)
    
    class_prediction, percent = get_prediction(prediction[0])

    # Adding prediction info into frame
    put_label(CATEGORIES[class_prediction], percent, frame)
