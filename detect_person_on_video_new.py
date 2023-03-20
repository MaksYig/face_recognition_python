import os.path
import pickle
import math
# from cv2 import cv2
import cv2
import face_recognition
import numpy as np
from Voice_Speaker import speak
import datetime
import train_module_by_video_screenshot as train
from Analize_Unknown_person import analyze_unknown_person as analyze
from PIL import Image






def face_confidence(face_distance, face_match_threshold = 0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'



def detect_person_on_video():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
    if not os.path.exists('Data'):
        os.mkdir('Data')
    # Initialize some variables
    data_path = os.listdir(f"Data/")
    face_locations = []
    face_encodings = []
    face_names = []
    bodies_arr = []
    unknown_found = False
    known_count = 0
    unknown_count = 0
    process_this_frame = True
    known_encodings = []
    known_names = []
    speak('Welcome to Face Recognition Aplication, I hope you will enjoy!')
    # Prepare DATA for Work
    for file in data_path:
        if file.endswith(".pkl"):
            data = pickle.loads(open(f"Data/{file}", 'rb').read())
            known_encodings.append(data['encodings'][0])
            known_names.append(data['name'])
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 5
        frame_id = int(round(video_capture.get(cv2.CAP_PROP_POS_FRAMES)))
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        # Load the classifier
        classifier = cv2.CascadeClassifier('Assets/haarcascade_fullbody.xml')
        # Detect the person body
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(bodies) > 0:
            for (i, body) in enumerate(bodies):
                bodies_arr.append(f"{body}_{i}")
            if len(bodies_arr) > 0:
                for (i, body) in enumerate(bodies_arr):
                    speak(f"Person {i+1} was detected")
        else: bodies_arr = []

        # Draw a bounding box around the person
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            for face_location in face_locations:

                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                if len(face_encodings) > 0:
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            confidence = face_confidence(face_distances[best_match_index])
                            name = 'Unknown'
                            # See if the face is a match for the known face(s)
                            if matches[best_match_index]:
                                idx = matches.index(True)
                                k_name = known_names[idx]
                                name = f"{k_name} ({confidence})"
                                if frame_id % multiplier == 10:
                                    if not os.path.exists('DataSet_from_Video'):
                                        os.mkdir('DataSet_from_Video')
                                    if known_count < 15:
                                        cv2.imwrite(f"DataSet_from_Video/{known_count}_{timestamp}_screen.jpeg", frame)
                                        known_count += 1
                                        print(f"Take screenshot {known_count}")
                                        train.train_module_by_video_screenshots()
                                        # Update DATA for Work
                                        for file in data_path:
                                            if file.endswith(".pkl"):
                                                data = pickle.loads(open(f"Data/{file}", 'rb').read())
                                                known_encodings.append(data['encodings'][0])
                                                known_names.append(data['name'])

                            else:
                                if filter(lambda n:n == "Unknown", face_names):
                                    if not os.path.exists('Unknown_DataSet_from_video'):
                                        os.mkdir('Unknown_DataSet_from_video')
                                    speak("Unknown Person detected")

                                    print(face_locations)
                                    print(f"Took screenshot of Unknown Person {unknown_count}")
                                    top, right, bottom, left = face_location
                                    face_img = rgb_small_frame[top - 40:bottom + 20, left - 20:right + 20]
                                    pil_img = Image.fromarray(face_img)
                                    pil_img.save(f'Unknown_DataSet_from_video/{unknown_count}_unknown.jpg')
                                    img = f'Unknown_DataSet_from_video/{unknown_count}_unknown.jpg'
                                    print(img)
                                    analyze(img)
                                    unknown_count += 1

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
            cv2.rectangle(frame, (left, top), (right, bottom), (50, 249, 42), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (50, 249, 42), cv2.FILLED)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, f"{name}", (left + 6, bottom - 6), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if name == "Unknown":
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