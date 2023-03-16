import os
import pickle
import sys
import face_recognition
import numpy as np



def train_module_by_video_screenshots():
    if not os.path.exists("DataSet_from_Video"):
        print("[ERROR] there is no directory 'DataSet_from_Video'")
        sys.exit()
    known_encodings = []
    images = os.listdir(f"DataSet_from_Video")
    if len(images) == 0:
        print("[ERROR] there is no files in 'DataSet_from_Video'")
        sys.exit()
    data_path = os.listdir(f"Data/")
    for (i, image) in enumerate(images):
        print(f'[+] processing {image}, img {i+1}/{len(images)}')
        #found faces on images
        face_img = face_recognition.load_image_file(f"DataSet_from_Video/{image}")
        face_encoding = face_recognition.face_encodings(face_img)[0]
        # See if the face is a match for the known face(s)
        for data_item in data_path:
            data = pickle.loads(open(f"Data/{data_item}", 'rb').read())
            matches = face_recognition.compare_faces(data['encodings'], face_encoding)
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data['encodings'], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                print(matches[best_match_index])
                name = data['name']
                known_encodings.append(face_encoding)
                print(f'This is {name}')
                data_upload = {
                   'name': name,
                   "encodings": known_encodings
                }
                with open(f"Data/{name}_encoding.pickle", 'wb') as file:
                    file.write(pickle.dumps(data_upload))
                os.remove(f"DataSet_from_Video/{image}")
                print(f"[INFO] File {name}_encoding.pickel successfully Updated!!")





def main():
    train_module_by_video_screenshots()
if __name__ == '__main__':
    main()
