import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import GestureRecognizerOptions, RunningMode, GestureRecognizer, \
    GestureRecognizerResult
from pathlib import Path
from datetime import datetime

ROOT_PATH = Path('../../../')
RESOURCES_PATH = ROOT_PATH.joinpath('resources')
MODEL_PATH = RESOURCES_PATH.joinpath('models/mediapipe/customization/rock_paper_scissors/gesture_recognizer.task')


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))


base_options = python.BaseOptions(model_asset_path=MODEL_PATH.resolve())
options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    WINDOW_NAME = "LIVE WEBCAM"
    cv2.namedWindow(WINDOW_NAME)
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    while success and cv2.waitKey(30) == -1:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.copy())
        recognizer.recognize_async(mp_image, int(datetime.now().timestamp() * 1_000))
        frame = np.fliplr(frame)
        cv2.imshow(WINDOW_NAME, frame)
        success, frame = camera.read()

    cv2.destroyAllWindows()
    camera.release()
