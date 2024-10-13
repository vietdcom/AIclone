import pyttsx3
from pydub import AudioSegment
import sox


# Hàm chuyển văn bản thành giọng nói với pyttsx3
def text_to_speech(text, volume=1.0, speech_rate=200, pitch=1.0):
    engine = pyttsx3.init()

    # Điều chỉnh Volume
    engine.setProperty('volume', volume)

    # Điều chỉnh Speech Rate
    engine.setProperty('rate', speech_rate)

    # Điều chỉnh Pitch bằng cách thay đổi giọng nói (pyttsx3 không hỗ trợ trực tiếp)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Giọng nam (voices[1] là giọng nữ)

    # Lưu âm thanh vào file tạm thời
    engine.save_to_file(text, 'output_temp.wav')
    engine.runAndWait()

    # Điều chỉnh Pitch bằng SoX
    adjust_pitch('output_temp.wav', 'output_adjusted.wav', pitch)

    # Phát lại file sau khi điều chỉnh
    return 'output_adjusted.wav'


# Hàm điều chỉnh Pitch và Timbre bằng sox
def adjust_pitch(input_file, output_file, pitch_semitones):
    tfm = sox.Transformer()

    # Điều chỉnh Pitch
    tfm.pitch(pitch_semitones)  # Điều chỉnh cao độ

    # Lưu lại file sau khi điều chỉnh
    tfm.build(input_file, output_file)


# Hàm điều chỉnh âm lượng và tốc độ (speech rate) với Pydub
def adjust_volume_and_speed(input_file, volume_change, speed_factor):
    # Load âm thanh
    audio = AudioSegment.from_file(input_file)

    # Điều chỉnh âm lượng
    audio = audio + volume_change  # Tăng/giảm volume (đơn vị: dB)

    # Điều chỉnh tốc độ
    audio = audio.speedup(playback_speed=speed_factor)

    # Lưu lại file đã chỉnh sửa
    output_file = 'final_output.wav'
    audio.export(output_file, format="wav")
    return output_file


# Ví dụ sử dụng
text = "Xin chào, đây là ví dụ về điều chỉnh giọng nói."
volume = 0.9  # Âm lượng từ 0.0 đến 1.0
speech_rate = 180  # Tốc độ (từ 100 đến 200 là bình thường)
pitch = 2.0  # Tăng cao độ lên 2 nửa cung (semitones)
speed_factor = 1.2  # Tăng tốc độ phát âm lên 20%

# Chuyển văn bản thành giọng nói với pyttsx3 và điều chỉnh pitch
output_wav = text_to_speech(text, volume=volume, speech_rate=speech_rate, pitch=pitch)

# Điều chỉnh âm lượng và tốc độ bằng Pydub
final_output_wav = adjust_volume_and_speed(output_wav, volume_change=5, speed_factor=speed_factor)

print(f"File âm thanh đã được tạo: {final_output_wav}")

# Phát bộ file ám thanh