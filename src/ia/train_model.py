from src.ia.util import *
from config import *


def train():

    # Loading data
    images, labels = load_data(DATA_FOLDER, CATEGORIES, IMAGE_WIDTH, IMAGE_HEIGTH)
