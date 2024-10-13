import pyaudio

audio = pyaudio.PyAudio()

# List devices
print("List devices:")
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    print(f"ID: {i}, Name: {device_info['name']}, Input Channels: {device_info['maxInputChannels']}")

audio.terminate()