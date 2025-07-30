from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import GestureRecognizerOptions, RunningMode, GestureRecognizer, \
    GestureRecognizerResult
from mediapipe.tasks.python.components.processors import classifier_options
import cv2
import time
import threading

current_image = None
camera_frame = None
camera_on = True


def read_camera() -> None:
    """
    Reads frame from webcam and stores it in global variable
    """

    global camera_frame, camera_on

    camera = cv2.VideoCapture(0)

    success, frame = camera.read()
    while success and camera_on:
        camera_frame = frame.copy()
        time.sleep(1 / 30)
        success, frame = camera.read()

    camera.release()


def show_window() -> None:
    """
    Shows window with image to review and webcam frame in top left corner of the window
    """

    global camera_frame, current_image

    while camera_on:
        time.sleep(1 / 30)
        if current_image is not None:
            img = cv2.resize(current_image, (960, 720))

            if camera_frame is not None:
                frame = cv2.resize(camera_frame, (128, 96))
                img[:frame.shape[0], :frame.shape[1], :] = frame

            cv2.imshow('Image approve mode', img)
            cv2.waitKey(1)

    cv2.destroyWindow('Image approve mode')


def is_approved(recognizer) -> bool:
    """
    Checks hand gesture in webcam frame until it recognizes thumb down and up gesture.
    Return True on thumb down and False on thumb up.
    """

    global camera_frame

    # give time for user to make action
    time.sleep(1.5)
    # then check gestures
    while camera_on:
        time.sleep(1 / 30)
        if camera_frame is None:
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=camera_frame.copy())
        result: GestureRecognizerResult = recognizer.recognize(mp_image)

        # check if any gesture recognized, then check if it's thumb down or up
        if result.gestures:
            if result.gestures[0][0].category_name == 'Thumb_Down':
                return False
            if result.gestures[0][0].category_name == 'Thumb_Up':
                return True


def main(model_path: str, images_folder: str, extension_patter: str) -> None:
    """
    Set's up recognizer options and runs

    :param model_path: path to model to use to recognize hand gestures
    :param images_folder: path to folder with images
    :param extension_patter: file extension pattern
    """

    options = GestureRecognizerOptions(base_options=python.BaseOptions(model_asset_path=Path(model_path).resolve()),
                                       running_mode=RunningMode.IMAGE,
                                       canned_gesture_classifier_options=classifier_options.ClassifierOptions(
                                           score_threshold=0.4,
                                           category_allowlist=['Thumb_Down', 'Thumb_Up']))

    with GestureRecognizer.create_from_options(options) as recognizer:
        # start thread to read webcam frame
        camera_thread = threading.Thread(target=read_camera)
        camera_thread.start()

        # start thread to show image and webcam frame in window
        window_thread = threading.Thread(target=show_window)
        window_thread.start()

        global current_image, camera_on

        # list files from folder and apply actions related to recognized gesture
        for img in Path(images_folder).glob(extension_patter):
            current_image = cv2.imread(str(img.resolve()))

            if is_approved(recognizer):
                print(f"Approved: {img}")
            else:
                print(f"Disapproved: {img}")

        camera_on = False
        camera_thread.join()
        window_thread.join()


if __name__ == '__main__':
    ROOT_PATH = Path('../')
    RESOURCES_PATH = ROOT_PATH.joinpath('resources')
    MODEL_PATH = RESOURCES_PATH.joinpath('models/mediapipe/gesture_recognizer.task')

    main(model_path=str(MODEL_PATH.resolve()),
         images_folder=str(ROOT_PATH.joinpath('DATA/review_images/').resolve()),
         extension_patter="*.[jJ][pP][gG]")
