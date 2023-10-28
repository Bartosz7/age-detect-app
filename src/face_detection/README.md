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

