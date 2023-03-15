import os

from cv2 import cv2
import face_recognition
from PIL import Image,ImageDraw
import pickle
import math
import numpy as np

def face_confidence(face_distance, face_match_threshold = 0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val -0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def face_rec():
    face_img = face_recognition.load_image_file("DataSet/Yigal_Maksimov/Yigal.JPG")
    face_location = face_recognition.face_locations(face_img)
    print(face_location)
    print(f'Faces found: {len(face_location)}')

    pil_img = Image.fromarray(face_img)
    pil_draw = ImageDraw.Draw(pil_img)

    for (top, right, buttom, left) in face_location:
        pil_draw.rectangle(((left, top), (right, buttom)), outline=(255, 255, 0), width=4)
    del pil_draw
    pil_img.save("images/new_yigal.jpg")

def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]
    print(img1_encodings)
    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]
    result = face_recognition.compare_faces([img1_encodings], img2_encodings)
    print(result)


def detect_person_on_video():
    data_path = os.listdir(f"Data/")
    # data = pickle.loads(open('Data/Yigal_Maksimov_encoding.pickle', 'rb').read())
    video = cv2.VideoCapture(0)
    while True:
        ret, image = video.read()
        locations = face_recognition.face_locations(image, model='hog')
        encoding = face_recognition.face_encodings(image, locations)
        for face_encoding, face_location in zip(encoding, locations):
            for data_item in data_path:
                data = pickle.loads(open(f"Data/{data_item}", 'rb').read())
                face_distances = face_recognition.face_distance(data['encodings'], face_encoding)
                best_match_index = np.argmin(face_distances)
                result = face_recognition.compare_faces(data['encodings'], face_encoding)
                # print(face_distances)
                # Draw a box around face
                left_top = (face_location[3], face_location[0])
                right_bottom = (face_location[1], face_location[2])
                color = [0, 255, 0]
                cv2.rectangle(image, left_top, right_bottom, color, 4)
                match = None
                if result[best_match_index]:
                    confidence = face_confidence(face_distances[best_match_index])
                    match = f"{data['name']} {confidence}"
                    # print(f"{data['name']}, {confidence}")
                    # Draw a lable with name
                    left_bottom = (face_location[3], face_location[2])
                    right_bottom2 = (face_location[1], face_location[2] + 24)
                    cv2.rectangle(image, left_bottom, right_bottom2, color, cv2.FILLED)
                    cv2.putText(
                        image,
                        match,
                        (face_location[3] + 10, face_location[2] + 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        1
                    )
                else:
                    # print('Unknown person')
                    match = 'Unknown'
                    # #Draw a lable with name
                    # left_bottom = (face_location[3], face_location[2])
                    # right_bottom2 = (face_location[1], face_location[2] + 20)
                    # cv2.rectangle(image, left_bottom, right_bottom2, color, cv2.FILLED)
                    # cv2.putText(
                    #     image,
                    #     match,
                    #     (face_location[3] + 15, face_location[2] + 15),
                    #     cv2.FONT_HERSHEY_PLAIN,
                    #     1,
                    #     (255, 255, 255),
                    #     1
                    # )

        cv2.imshow('detect_person_on_video is running', image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def main():
    # face_rec()
    # compare_faces("DataSet/Yigal.JPG", "images/YigalandOleg.jpg")
    detect_person_on_video()



if __name__ == '__main__':
    main()