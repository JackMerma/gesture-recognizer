import cv2


def get_frame(cap):
    return cap.read()


def show_frame(frame):
    cv2.imshow("Camera feed", frame)


def is_pressing_killing_key():
    return cv2.waitKey(1) & 0xFF == ord('q')
