import pyttsx3
import speech_recognition as sr
import time
import threading
from gtts import gTTS
import os
from playsound import playsound

r = sr.Recognizer()

end_event = threading.Event()


def listen_and_read():
    print("\nStart listening...")
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=3, phrase_time_limit=5)
        try:
            print("Processing...")
            text = r.recognize_google(audio, language="vi-VN")
            print("Bạn đã nói: " + text)

            print("Change to Vietnamese...")
            tts = gTTS(text=text, lang='vi')
            tts.save("output_vi.mp3")
            print(f"file output: output_vi.mp3")

            playsound("output_vi.mp3")

        except sr.UnknownValueError:
            print("I don't understand")
        except sr.RequestError as e:
            print(f"Error connection: {e}")
        except Exception as e:
            print(f"Error other: {e}")
    print("Done listening")


def counting(timeset):
    for i in range(timeset, 0, -1):
        if end_event.is_set():
            return
        print(f"Time left: {i} seconds", end='\r')
        time.sleep(1)
    print("\nOut of time")
    end_event.set()


def main():
    timeset = 5
    print(f"Your time: {timeset} seconds")

    thread = threading.Thread(target=counting, args=(timeset,))
    thread.start()

    while not end_event.is_set():
        listen_and_read()
        if end_event.is_set():
            break

    thread.join()
    print("End of program")

if __name__ == "__main__":
    main()
