import os.path
import pickle
import sys
from cv2 import cv2
import face_recognition
from take_screenshoot_from_video import main as video
from train_module_by_video_screenshot import main as screen_train

#For first time Training
def train_modul_by_img(name):
    if not os.path.exists("DataSet"):
        print("[ERROR] there is no directory 'DataSet'")
        sys.exit()
    known_encodings = []
    images = os.listdir(f"DataSet/{name}")
    print(images)
    for (i, image) in enumerate(images):
        print(f'[+] processing {image}, img {i+1}/{len(images)}')
        face_img = face_recognition.load_image_file(f"DataSet/{name}/{image}")
        face_encoding = face_recognition.face_encodings(face_img)[0]
        # print(face_encoding)
        if len(known_encodings) == 0:
            known_encodings.append(face_encoding)
        else:
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_encoding], known_encodings[item])
                # print(result)
                if result[0]:
                    known_encodings.append(face_encoding)
                    print('Same Person')
                    break
                else:
                    print('Another person')
                    break
    print(known_encodings)
    print(f"Length:{len(known_encodings)}")
    data = {
        'name': name,
        "encodings": known_encodings
    }
    if not os.path.exists('Data'):
        os.mkdir('Data')
    with open(f"Data/{name}_encoding.pkl", 'wb') as file:
        file.write(pickle.dumps(data))
    return f"[INFO] File {name}_encoding.pkl successfully created!!"


def main_first_train():
    if not os.path.exists("DataSet"):
        print("[ERROR] there is no directory 'DataSet'.Nothing to train.")
        sys.exit()
    print("Welcome to first training script.")
    train_dir = os.listdir('DataSet')
    for (i, person_folder) in enumerate(train_dir):
        print(f'To train {person_folder}, press ({i+1})')
    print("Take screenShots from live web streem, press (222)")
    print('Train from ScreenShots took from live streem, press (333)')
    person_to_train = int(input("Choose person to train:"))
    for (i, person_folder) in enumerate(train_dir):
        if person_to_train == i+1:
            print(person_folder)
            train_modul_by_img(person_folder)
            print(f"{train_dir[i]} was trained successfully!")
            v = input('For new train, press (t), For exit, press (x)' )
            if v == 't':
                main_first_train()
            else: sys.exit()
    if person_to_train == 222:
        video()
    if person_to_train == 333:
        screen_train()

def take_screenshot_from_video():
    video_capture = cv2.VideoCapture(0)
    count = 0
    if not os.path.exists('DataSet_from_Video'):
        os.mkdir('DataSet_from_Video')
    if not video_capture.isOpened():
        sys.exit('Video source is not found....')
    while True:
        ret, frame = video_capture.read()
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 5
        # print(fps)
        if ret:
            frame_id = int(round(video_capture.get(1)))
            # print(frame_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)
            if frame_id % multiplier == 0:
                if count < 15:
                    cv2.imwrite(f"DataSet_from_Video/{count}_screen.jpg", frame)
                    count += 1
                    print(f"Take screenshot {count}")
            if k == ord('s'):
                cv2.imwrite(f"DataSet_from_Video/{count}_screen_manual.jpg", frame)
                count += 1
                print(f"Take screenshot manual {count}")

        else:
            print(f'[Error] Cant get frame...')
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
      # print(train_modul_by_img("Orly_Novoselskaya"))
      # train_modul_by_img("Yigal_Maksimov")
      main_first_train()
if __name__ == '__main__':
    main()
