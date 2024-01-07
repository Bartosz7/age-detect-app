Face detection from opencv:

Usually finds faces when they are big and facing the screen.

Slightly rotated faces are usually not detected.

For min_neighbors < 20 there are a lot of false positives.

For min_neighbors > 50 some faces are not detected.

Dog is not recognized (thank god) as human's face.

Images without haarcascade subfolder were created using haarcasade_frontalface_alt.xml classifier

Ale_tree clasifier doesn't detect well.

Face detection from mediapipe:

Works pretty well.

Every face was recognised (except the dog's one).

There are included point indicating nose, ears and eyes. Not only the frame.

# Development Notes
To develop a new FaceDetector, implement a subclass of a FaceDetector, following the
template and override the detect_faces function to include relevant logic
for face detection:
```{python}
class ExampleFaceDetector(FaceDetector):

    def __init__(self):
        super().__init__()
    
    def detect_faces(self, image: np.ndarray) -> List[DetectedFace] | []:
        pass # implement relevant logic here
```
Next add your new `ExampleFaceDetector` to the `FaceDetectors` Enum.
Finally to use it within the GUI, add it to the `FACE_DETECTION_MODELS`
by supplying a string name by which it will be available in GUI and a corresponding Enum from FaceDetectors.

To use the relevant `FaceDetector`, provide the Enum name to the `FaceDetectorFactory` object.
The resulting face detector object supports the interface of `FaceDetector` class.
```{python}
factory = FaceDetectorFactory()
face_detector = factory.create_model(FaceDetectors.MEDIA_PIPE)
detected_faces = face_detector.detect_faces(img)
```


