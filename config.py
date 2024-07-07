import argparse

PATH_CLASS  = ["data/open_hand", "data/close_hand"]
CLASS_NAME = ["OPEN", "CLOSE"]
EXTENSION = "jpg"

def load_parser():
    parser = argparse.ArgumentParser(description="Gesture Recongnizer desc...")

    # Adding arguments
    parser.add_argument('-l', '--load', action='store_true', help="capture the hand data using the camera.")
    parser.add_argument('-o', '--open', action='store_true', help="activate open hand capture.")
    parser.add_argument('-c', '--close', action='store_true', help="activate close hand capture.")

    return parser.parse_args()
