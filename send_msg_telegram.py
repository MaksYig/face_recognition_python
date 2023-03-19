import requests
import os
from dotenv import load_dotenv
load_dotenv('.env')

TOKEN = os.environ.get('TELEGRAM_BOOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_message_with_img(img,message):

    img = img
    message = message
    # Create url
    url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'

    # Create json link with message
    data = {'chat_id': CHAT_ID,
            'caption': message}
    requests.post(url, data, files={'photo': open(img, 'rb')})


def main():
    send_message_with_img()

if __name__ == '__main__':
    main()