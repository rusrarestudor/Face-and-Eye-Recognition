import cv2
import sys

# Get user supplied values
# Get the image name as command line argument
# First we pass the image and then the cascade for detecting faces and eyes provided by OpenCV

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
eyePath = 'haarcascade_eye.xml'


# Create the haar cascade
# Now we create the cascade and initialize it with our face and eye cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)

# Read the image and convert it to grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# The detectMultiScale() function is a general function that detects objects
faces = faceCascade.detectMultiScale(
    # The first attribute is the grayscale image
    gray,
    # The second attribute is used to scale faces bigger or smaller
    scaleFactor=1.1,
    # The third defines how many objects are detected near the current one before it declares the face found
    minNeighbors=5,
    # The forth gives the size of each moving window to detect objects
    minSize=(25, 25),
)

print("Found {0} faces!".format(len(faces)))

# Draw a purple rectangle around the faces and yellow rectangles around the eyes
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 105, 180), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eyeCascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 191, 255), 2)

# We display the image and wait for the user to press a key.
cv2.imshow("FaceDetect ™ ", image)
cv2.waitKey(0)
