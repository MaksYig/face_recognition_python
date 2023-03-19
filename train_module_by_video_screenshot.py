import os
import pickle
import sys
import face_recognition
import numpy as np


def train_module_by_video_screenshots():
    if not os.path.exists("DataSet_from_Video"):
        print("[ERROR] there is no directory 'DataSet_from_Video'")
        sys.exit()
    # Initialize some variables
    data_path = os.listdir(f"Data/")
    known_encodings = []
    known_names = []
    current_encoding = []
    # Prepare DATA for Work
    for file in data_path:
        if file.endswith(".pkl"):
            data = pickle.loads(open(f"Data/{file}", 'rb').read())
            known_encodings.append(data['encodings'][0])
            known_names.append(data['name'])
            print(len(known_encodings))


    images = os.listdir(f"DataSet_from_Video")
    if len(images) == 0:
        print("[ERROR] there is no files in 'DataSet_from_Video'")
        sys.exit()
    data_path = os.listdir(f"Data/")
    for (i, image) in enumerate(images):
        print(f'[+] processing {image}, img {i+1}/{len(images)}')
        # found faces on images
        face_img = face_recognition.load_image_file(f"DataSet_from_Video/{image}")
        face_encodings = face_recognition.face_encodings(face_img)
        if len(face_encodings) > 0:
            # See if the face is a match for the known face(s)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = ''
                print(len(known_encodings))
                print(matches[best_match_index])
                if matches[best_match_index]:
                    idx = matches.index(True)
                    k_name = known_names[idx]
                    name = f"{k_name}"
                    data = pickle.loads(open(f"Data/{name}_encoding.pkl", 'rb').read())
                    current_encoding = data['encodings']
                    current_encoding.append(face_encoding)
                    print(f'[INFO ⚠]This is {name}.Start to update....')
                    data_upload = {
                        "name": name,
                        "encodings": current_encoding
                    }
                    with open(f"Data/{name}_encoding.pkl", 'wb') as file:
                        file.write(pickle.dumps(data_upload))
                    print(f"[INFO] File {name}_encoding.pkl successfully Updated!!")
                    os.remove(f"DataSet_from_Video/{image}")
                else:
                    print("No mach with Faces in Data")
        else:
            print(f'No faces found!')
            # os.remove(f"DataSet_from_Video/{image}")




def main():
    train_module_by_video_screenshots()
if __name__ == '__main__':
    main()
