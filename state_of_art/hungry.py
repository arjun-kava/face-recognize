from libface_recogniser_cnn import *

# initialize properties
data_set_path = "../assets/face_dataset"
encoding_path = "face_encodings.pkl"
output_path = "../assets/videos/Inception - Official Trailer [HD] Output.avi"
feed_path = "../assets/videos/Inception - Official Trailer [HD].mp4"

# init recogniser
recogniser = FaceRecogniser(data_set_path, encoding_path, output_path, "cnn")

# encode face data set
recogniser.encode()

# detect and recognise faces
recogniser.feed(feed_path, is_display=True)
