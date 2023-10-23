import cv2
import matplotlib.pyplot as plt

import os

# shira
# imagePath = "face_images/20230527_224925.jpg"

# ho lee shit
# imagePath = "face_images/2023_0528_023416.jpg"

# ja w lesie

filenames = next(os.walk("face_detection/face_images/", (None, None, [])))[2]

min_neighbors = 4
clasifier = "haarcascade_frontalface_alt_tree.xml"
face_images_path = "face_detection/face_images/"

for imagePath in filenames:
    imagePath = imagePath
    img = cv2.imread(face_images_path + imagePath)
    print(
        img.shape
    )  # width, height, dimensions(in other words: is rgb or not) (931, 1170, 3)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_image.shape)  # (931, 1170)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + clasifier
    )  # load clasifier for frontal faces detection

    # detect face
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(40, 40)
    )

    # edit img: draw rectangle around face
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # make image color again
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    directory = (
        "face_detection/min_neighbors_" + str(min_neighbors) + "/" + clasifier + "/"
    )
    os.makedirs(directory, exist_ok=True)
    plt.figure(figsize=(20, 10))
    plt.imsave(directory + imagePath, img_rgb)
    plt.axis("off")
