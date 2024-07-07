import cv2
import sys
import os
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_data(path, categories, img_width, img_height):

    images = []
    labels = []
    
    for category in range(len(categories)):
        category_dir = os.path.join(path, categories[category])
        print("DEBUG: ", category_dir)

        # Processing files
        for file in os.listdir(category_dir):
            filepath = os.path.join(category_dir, file)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_width, img_height))
            images.append(np.array(image))
            labels.append(category)

    return images, labels
