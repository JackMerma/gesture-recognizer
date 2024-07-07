import cv2
import os
import shutil
from config import *


def get_frame(cap):
    return cap.read()


def show_frame(frame):
    cv2.imshow("Camera feed", frame)


def is_pressing_killing_key():
    return cv2.waitKey(1) & 0xFF == ord('q')


def save_frame(frame, file_path):
    cv2.imwrite(file_path, frame)


def check(path):

    # Deleting all path data
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Can't delete {path} directory")

    # Creating if does not exist
    os.makedirs(path, exist_ok=True)


def capture(data_class):

    # Getting path
    path = os.path.join(DATA_FOLDER, CATEGORIES[data_class])

    # Assert that path exist
    check(path)

    # Capturing data using the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Can't open the camera")

    frame_count = 0

    # Real time capture
    while True:
        ret, frame = get_frame(cap)

        if not ret:
            raise Exception("Can't load the frame correctly")

        frame_count += 1
        print(f"Frame: {frame_count}")

        # Saving readed frame
        file_name = f"{CLASS_NAME[data_class]}{frame_count}.{EXTENSION}"
        file_path = os.path.join(path, file_name)
        save_frame(frame, file_path)

        # Showing the frame
        show_frame(frame)

        # Exit from the loop
        if is_pressing_killing_key():
            break

    cap.release()
    cv2.destroyAllWindows()
