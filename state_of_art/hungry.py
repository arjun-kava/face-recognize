import cv2
from libface_recogniser_cnn import *

feed_path = "../assets/videos/The Greatest Showman.mp4"

# get reference of feed
video_capture = cv2.VideoCapture(feed_path)
video_capture.set(cv2.CAP_PROP_FPS, 50)

# initialize face recognizer with
recognizer = FaceRecogniserCNN(known_face_dir="../assets/single_image_face_dataset")

# load images from known face directory
recognizer.load_know_faces()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # extract faces from frame
    face_locations, face_names = recognizer.recognize(frame, model="hog")

    # draw rectangle on found faces
    recognizer.draw_rect(frame, face_locations, face_names)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
