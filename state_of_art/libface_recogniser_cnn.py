import os
import pickle
import time
import cv2
import face_recognition
import imutils


class FaceRecogniser:
    """
    Initialize recogniser
    :param data_set_path: path of data set contains face images
                          person_name __
                                        |__ person_name_000001.jpg
                                        |__ person_name_000002.jpeg
    :param encoding_path: path in which encoding will be stored
    :param detection_method: method of detection either cnn(slower) or hog(faster)
    """

    def __init__(self, data_set_path=None, encoding_path=None, output_path=None, detection_method="hog"):
        # Path of data set which contains multiple images of faces
        self.data_set_path = data_set_path

        # Path of serialized db of facial encoding
        self.encoding_path = encoding_path

        # Path of output
        self.output_path = output_path

        # method of detection hog/cnn
        self.detection_method = detection_method

        # initialize current frame
        self.current_frame = None

        # resize frame to speed up
        self._preprocess_width = 700

    """
    Encode images and save as pickle 
    """

    def encode(self, is_force=False):
        # validate required parameters
        if (self.data_set_path is None) or (os.path.exists(self.data_set_path) == False):
            raise Exception("Data set path not found!")
        if self.encoding_path is None:
            raise Exception("Encoding path not found!")

        # return if already encoded
        if os.path.exists(self.encoding_path) is True and is_force is False:
            return

        # grab the paths to the input images in our data set
        image_paths = list(self._list_images(self.data_set_path))

        # initialize the list of known encodings and known names
        known_encodings = []
        known_names = []

        # loop over the image paths
        for (i, image_path) in enumerate(image_paths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
            print("image_path.split", image_path.split("/")[-1])
            file_name, file_ext = os.path.splitext(image_path.split("/")[-1])  # name_000001
            person_name = file_name[:-7]

            if os.path.exists(image_path) is False:
                raise Exception("File does not exists!")

            # load the input image and convert it from RGB (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(image_path)

            if image is None or image.shape is None:
                raise Exception("Failed to read image")

            rgb = self._convert_to_rgb(image)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            # compute the facial embedding for the face

            boxes, encodings = self._recognise(rgb)

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                known_encodings.append(encoding)
                known_names.append(person_name)

        # dump encodings into file system
        self._dump_encodings(known_encodings, known_names)

    """
    process input feed and identify faces
    :param feed_path: path of video feed
    :param is_display: specified displaying video
    :param is_write: specify writing of video
    """

    def feed(self, feed_path, is_display=False, is_write=True):

        if os.path.exists(feed_path) is False:
            raise Exception("File does not exists!")

        # load encodings from file system
        data = self._load_encodings()

        # open stream
        vs = cv2.VideoCapture(feed_path)
        writer = None

        grabbed, frame = vs.read()

        if grabbed is False:
            raise Exception("Failed to read video!")

        # loop over frames from the video file stream
        while True:
            start_time = time.time()  # start time of the loop
            # grab the frame from the threaded video stream
            grabbed, frame = vs.read()
            if frame is None: break

            original_frame = frame.copy()

            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            frame = self._convert_to_rgb(frame)
            frame = self._preprocess_frame(frame)
            original_frame = self._preprocess_frame(original_frame)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes, encodings = self._recognise(frame)

            # map matching names and
            mapped_names = self._map_encodings(encodings, data["encodings"], data["names"])

            # draw rectangle on the faces
            self._draw_rectangle(original_frame, boxes, mapped_names)

            # write video if specified
            if is_write == True: writer = self._write_video(writer, original_frame)

            # check to see if we are supposed to display the output frame to
            # the screen
            if is_display == True and self._display(original_frame) == False: break
            print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop

        self._cleanup()

    """
    Convert BGR(Opencv) to RGB(dlib) format
    :param frame: single frame of video
    :returns frame: RGB coded frame
    """

    def _convert_to_rgb(self, frame):
        # convert the input frame from BGR to RGB then resize it to have
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    """
    Reshape frame for better speed 
    :param frame: single frame of video
    :returns reshaped_frame, transpose
    """

    def _preprocess_frame(self, frame):
        # resize width (to speedup)
        if self._preprocess_width is not None:
            frame_reshaped = imutils.resize(frame, width=self._preprocess_width)
            return frame_reshaped
        else:
            return frame

    """
    Detect bounding box of multiple face and 
    :param frame: single frame of video
    :returns boxes,encodings
    """

    def _recognise(self, frame):
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(frame, model=self.detection_method)
        encodings = face_recognition.face_encodings(frame, boxes)
        return boxes, encodings

    """
    load dumped encoding pickle
    :returns { encodings:<Array>, names:<Array> }
    """

    def _load_encodings(self):
        # validate required parameters
        if self.encoding_path is None or not os.path.exists(self.encoding_path):
            raise Exception("Encoding path not found!")

        file = open(self.encoding_path, 'rb')
        response = pickle.load(file)
        file.close()
        return response

    """
    Dump encodings and names dict. into specified path
    :param knows_encodings: ordered array of encodings
    :param known_names: ordered array of names
    """

    def _dump_encodings(self, knows_encodings, known_names):
        # validate required parameters
        print(" self.encoding_path", self.encoding_path)
        if self.encoding_path is None:
            raise Exception("Encoding path not found!")

        # dump the facial encodings + names to disk
        data = {"encodings": knows_encodings, "names": known_names}
        file = open(self.encoding_path, 'wb')
        pickle.dump(data, file)
        file.close()

    """
    Map names with specified encodings
    :param frame_encodings: array of current frame encodings
    :param encodings: array of ordered encodings
    :param name: array of ordered names
    :returns mapped_names: array of mapped named 
    """

    def _map_encodings(self, frame_encodings, encodings, names):
        mapped_names = []
        # loop over the facial embeddings
        for encoding in frame_encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(encodings, encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matched_id_xs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matched_id_xs:
                    name = names[i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            mapped_names.append(name)

        return mapped_names

    """
    Draw rectangle(s) on recognized faces ,also on unknown faces
    :param frame: single frame of video
    :param boxes: ordered array of face locations
    :param mapped_names: ordered array of names respected to face locations
    :param frame_transpose: transpose of reshaped image
    """

    def _draw_rectangle(self, frame, boxes, mapped_names):
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, mapped_names):
            # rescale the face coordinates
            top = int(top)
            right = int(right)
            bottom = int(bottom)
            left = int(left)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

    """
    Write frames with recognised frames
    :param writer: None or reference of video writing stream
    :param frame: single frame of video
    :returns writer: reference of video stream
    """

    def _write_video(self, writer, frame):
        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        if writer is None and self.output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(self.output_path, fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)

        # if the writer is not None, write the frame with recognized
        # faces to disk
        if writer is not None:
            writer.write(frame)

        return writer

    """
    Display video with recognised faces
    :param frame: single frame of video
    :returns boolean
    """

    def _display(self, frame):
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            return False

        return True

    """
    clean up video stream and writing stream
    :param vs: reference of video streamer
    :param writer: reference of writer
    """

    def _cleanup(self, vs=None, writer=None):
        # do a bit of cleanup
        cv2.destroyAllWindows()

        # stop video stream
        if vs is not None:
            vs.stop()

        # check to see if the video writer point needs to be released
        if writer is not None:
            writer.release()

    def _list_images(self, basePath, contains=None):
        # return the set of files that are valid
        return self._list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
                                contains=contains)

    def _list_files(self, basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
        # loop over the directory structure
        for (rootDir, dirNames, filenames) in os.walk(basePath):
            # loop over the filenames in the current directory
            for filename in filenames:
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if contains is not None and filename.find(contains) == -1:
                    continue

                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file is an image and should be processed
                if ext.endswith(validExts):
                    # construct the path to the image and yield it
                    imagePath = os.path.join(rootDir, filename)
                    yield imagePath
