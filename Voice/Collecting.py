import pyaudio
import wave

# constants
FORMAT = pyaudio.paInt16  # sound format: 16-bit
CHANNELS = 1           # channels: 1-mono, 2-stereo
RATE = 16000              # 16KHz sample rate
CHUNK = 1024              # record in chunks of 1024 samples
RECORD_SECONDS = 10        # time in seconds to record
OUTPUT_FILENAME = eval(input("Enter output file name: ")) + ".wav"  # output audio file name
DEVICE_INDEX = 1      # device index
# PyAudio
audio = pyaudio.PyAudio()

# open a stream using callback
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, input_device_index=DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

print("Start Recording....")

frames = []

# record for RECORD_SECONDS seconds
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished Recording.")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

# save as WAV
with wave.open('Data/' + OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Saved as: {'Data/' + OUTPUT_FILENAME}")