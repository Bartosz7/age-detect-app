from src.face_detection.detection import *

if __name__ == "__main__":
    results = run_detection(
        detector=Detectors.OPEN_CV,
        input_path="data/train_data",
        output_path="data/output",
    )
