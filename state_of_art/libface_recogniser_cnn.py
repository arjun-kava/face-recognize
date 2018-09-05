import face_recognition
import cv2
import os


class FaceRecogniserCNN:
    """
    Initialize recogniser
    :param known_face_dir: path of directory which contains known faces
    """

    def __init__(self, known_face_dir=None):
        self.known_face_dir = known_face_dir
        self.known_face_tuples = {}
        self.valid_images_ext = [".jpg", ".png"]
        self.resize_ratio = 1
        self.number_of_times_to_upsample = 1

    """
    Load all known as {name,encoding} tuple pairs
    """

    def load_know_faces(self):
        # list only valid images from specified known face directory
        for f in os.listdir(self.known_face_dir):
            filename, file_extension = os.path.splitext(f)
            if file_extension.lower() not in self.valid_images_ext:
                continue

            # extract full path of image
            known_image_face_path = os.path.join(self.known_face_dir, f)

            # load and encode faces from image using face_recognition
            self.known_face_tuples[filename] = \
                face_recognition.face_encodings(face_recognition.load_image_file(known_image_face_path))[0]

    """
    Recognise multiple know and unknown faces
    :param frame: image / frame of video
    :returns (array of face locations, array of face names)
    """

    def recognize(self, frame, model="hog"):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / self.resize_ratio, fy=1 / self.resize_ratio)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=self.number_of_times_to_upsample,
                                                         model=model)

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # process encodings and find matching faces
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(list(self.known_face_tuples.values()), face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = list(self.known_face_tuples.keys())[first_match_index]

            face_names.append(name)
        return face_locations, face_names

    """
    Draw rectangle on recognized faces
    :param frame: image / frame of video
    :param face_locations: array of face coordinates
    :param face_names: names of recognized faces
    """

    def draw_rect(self, frame, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= self.resize_ratio
            right *= self.resize_ratio
            bottom *= self.resize_ratio
            left *= self.resize_ratio

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
