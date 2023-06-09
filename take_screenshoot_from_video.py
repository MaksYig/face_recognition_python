import os
import sys
# from cv2 import cv2
import cv2
import datetime

def take_screenshot_from_video(*args):
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
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
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            # print(frame_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)
            if frame_id % multiplier == 0:
                if count < 15:
                    if args:
                        for arg in args:
                            parent_path = 'Dataset'
                            new_folder_name = str(arg)
                            new_folder_path = os.path.join(parent_path, new_folder_name)
                            if not os.path.exists(new_folder_path):
                                os.makedirs(new_folder_path)
                            cv2.imwrite(f"{new_folder_path}/{arg}_{count}_{timestamp}_screen.jpeg", frame)
                            count += 1
                            print(f"Take screenshot {count}")
                    cv2.imwrite(f"DataSet_from_Video/{count}_{timestamp}_screen.jpeg", frame)
                    count += 1
                    print(f"Take screenshot {count}")
            if k == ord('s'):
                cv2.imwrite(f"DataSet_from_Video/{count}_screen_manual.jpg", frame)
                count += 1
                print(f"Take screenshot manual {count}")
            if k == ord('q'):
                print(f"EXIT from script")
                break

        else:
            print(f'[Error] Cant get frame...')
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main(*args):
    take_screenshot_from_video(*args)

if __name__ == '__main__':
    main()
