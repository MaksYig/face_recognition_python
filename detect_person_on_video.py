import os.path
import pickle
import math
from cv2 import cv2
import face_recognition
import numpy as np
import pyttsx3
import datetime
import train_module_by_video_screenshot as train


#voice init
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate+0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def face_confidence(face_distance, face_match_threshold = 0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detect_person_on_video():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Initialize some variables
    data_path = os.listdir(f"Data/")
    face_locations = []
    face_encodings = []
    face_names = []
    count = 0
    process_this_frame = True
    speak('Welcome to Face Recognition Aplication, I hope you will enjoy!')
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 5
        frame_id = int(round(video_capture.get(1)))
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                name = "Unknown"
                # See if the face is a match for the known face(s)
                for data_item in data_path:
                    data = pickle.loads(open(f"Data/{data_item}", 'rb').read())
                    matches = face_recognition.compare_faces(data['encodings'], face_encoding)
                    # use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(data['encodings'], face_encoding)
                    best_match_index = np.argmin(face_distances)
                    confidence = face_confidence(face_distances[best_match_index])

                    if matches[best_match_index]:
                        name = f"{data['name']} ({confidence})"
                        if frame_id % multiplier < 5:
                            if count < 15:
                                cv2.imwrite(f"DataSet_from_Video/{count}_{timestamp}_screen.jpeg", frame)
                                count += 1
                                print(f"Take screenshot {count}")
                                train.train_module_by_video_screenshots()
                        # print(f"{name}, {confidence}")
                    if not matches[best_match_index]:
                        print(f"Took screenshot of Unknown")
                face_names.append(name)
                print(face_names)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, f"{name}", (left + 6, bottom - 6), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    detect_person_on_video()



if __name__ == '__main__':
    main()