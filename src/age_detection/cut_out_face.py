import cv2
import mediapipe as mp
import os
import sys
import glob
from tqdm import tqdm
from multiprocessing import Pool

def cutout_face(args) -> None:
    file, save_folder = args
    
    """Detect face in an image and save the face"""

    # Using the `with` statement ensures that the mediapipe solution is properly closed
    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        # Read image
        image = cv2.imread(file)
        height, width, _ = image.shape

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image and get the results
        results = face_detection.process(image_rgb)

        if results.detections and len(results.detections) == 1:
            # Assuming the first detection is the most prominent face
            face = results.detections[0].location_data.relative_bounding_box
            
            face = image[
                int(face.ymin * height):int((face.ymin+face.height) * height),
                int(face.xmin * width):int((face.xmin+face.width) * width),
            ]
            dir_name = os.path.basename(os.path.dirname(file))
            os.makedirs(os.path.join(save_folder, dir_name), exist_ok=True)
            if face.shape[0] > 50 and face.shape[1] > 50:
                cv2.imwrite(os.path.join(save_folder, dir_name, os.path.basename(file)), face)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cut_out_face.py <image_folder> <save_folder>")
        sys.exit(1)

    images = glob.glob(os.path.join(sys.argv[1], "**", "*.jpg"), recursive=True)

    # Adjust the number of processes based on your system capabilities. Here, I've used 4 processes.
    with Pool(processes=4) as pool:
        list(tqdm(pool.imap(cutout_face, [(image, sys.argv[2]) for image in images]), total=len(images)))
