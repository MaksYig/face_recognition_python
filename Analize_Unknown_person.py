from deepface import DeepFace
import os
import json
import datetime
import send_msg_telegram as msg_tg


def analyze_unknown_person(img_name):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    try:
        analyze_img = DeepFace.analyze(img_path=img_name, actions=['gender','age'])
        print(analyze_img)
        gender = analyze_img[0].get('dominant_gender')
        age = analyze_img[0].get('age')

        msg_tg.send_message_with_img(img_name, f'Unknown person detected:\nTime: {timestamp},\nGender: {gender},\nAge (around): {age}.')

        # old_data = open(f"Assets/Data/unknown_analyze.json", 'rb').read()
        # data=[old_data[0]]
        # new_data = analyze_img[0]
        # new_data.append({"timestamp": timestamp})
        # data.append(new_data)
        # with open('Assets/Data/unknown_analyze.json', 'w') as file:
        #     json.dump(new_data, file, indent=4, ensure_ascii=False)
    except Exception as _ex:
        return _ex




def main():
    analyze_unknown_person()
if __name__ == '__main__':
    main()