import argparse

DATA_FOLDER = "data"
CATEGORIES = ["open", "close"]
CLASS_NAME = ["OPEN", "CLOSE"]
EXTENSION = "jpg"
MODEL_FOLDER = "models"
IMAGE_WIDTH = 50
IMAGE_HEIGTH = 50


def load_parser():
    parser = argparse.ArgumentParser(description="Gesture Recongnizer desc...")

    # Adding load data arguments
    parser.add_argument('-l', '--load', action='store_true', help="capture the hand data using the camera.")
    parser.add_argument('-o', '--open', action='store_true', help="activate open hand capture.")
    parser.add_argument('-c', '--close', action='store_true', help="activate close hand capture.")

    # Adding IA arguments
    parser.add_argument('-t', '--train', action='store_true', help="train model with loaded data.")

    return parser.parse_args()
