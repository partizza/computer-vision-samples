from pathlib import Path
from mediapipe_model_maker import gesture_recognizer

# Script to train model with new hand gesture - example from official mediapipe documentation

ROOT = Path("../../../")
DATASET_PATH = ROOT.joinpath("DATA/rps_data_sample/")
EXPORT_PATH = ROOT.joinpath("resources/models/mediapipe/customization/rock_paper_scissors")

# load dataset
print(f"Resolved dataset path: {DATASET_PATH.resolve()}")
data = gesture_recognizer.Dataset.from_folder(
    dirname=str(DATASET_PATH.resolve()),
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# traine model
hparams = gesture_recognizer.HParams(export_dir=str(EXPORT_PATH.resolve()))
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# evaluate model
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

#export model
model.export_model()