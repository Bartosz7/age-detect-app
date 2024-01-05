from face_detection.detection import *
import time

if __name__ == "__main__":
    # results = run_detection(
    #     detector=Detectors.OPEN_CV,
    #     input_path="data/train_data",
    #     output_path="data/output",
    # )
    results = run_detection(
        detector=Detectors.MEDIAPIPE,
        input_path="data/train_data",
        output_path="data/output",
    )
