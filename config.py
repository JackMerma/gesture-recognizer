import argparse

CAMERA_INDEX = 0
DATA_FOLDER = "data"
CATEGORIES = ["open", "close"]
CLASS_NAME = ["OPEN", "CLOSE"]
EXTENSION = "jpg"
MODEL_FOLDER = "models"
MODEL_NAME = "model"
MODEL_EXTENSION = "keras"
IMAGE_WIDTH = 100
IMAGE_HEIGTH = 100
TEST_SIZE = 0.4
EPOCHS = 10
RESOURCES_PATH = "assets/images/"
GAME_FILE = "resources.png"


def load_parser():
    parser = argparse.ArgumentParser(description="Gesture Recongnizer desc...")

    # Adding load data arguments
    parser.add_argument('-l', '--load', action='store_true', help="capture the hand data using the camera.")
    parser.add_argument('-o', '--open', action='store_true', help="activate open hand capture.")
    parser.add_argument('-c', '--close', action='store_true', help="activate close hand capture.")

    # Adding IA arguments
    parser.add_argument('-t', '--train', action='store_true', help="train model with loaded data.")
    parser.add_argument('-n', '--name', type=str, required=False, help="saved model name.")
    parser.add_argument('-p', '--play', action="store_true", help="play real time app predictor.")

    return parser.parse_args()
