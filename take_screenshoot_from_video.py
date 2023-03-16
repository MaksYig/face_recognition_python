import os.path
import sys
from cv2 import cv2

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
            if k == ord('q'):
                print(f"EXIT from script")
                break

        else:
            print(f'[Error] Cant get frame...')
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    take_screenshot_from_video()

if __name__ == '__main__':
    main()
