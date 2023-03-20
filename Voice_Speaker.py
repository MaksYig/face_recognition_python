import pyttsx3

#voice init
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate+0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def main():
    speak()

if __name__ == '__main__':
    main()