from src.capture.capture_data import *
from config import *


def main():

    # Getting parser
    args = load_parser()

    if args.load:
        if args.open:
            capture(0)
        elif args.close:
            capture(1)



if __name__ == "__main__":
    main()
