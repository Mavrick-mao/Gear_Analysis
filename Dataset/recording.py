import pyaudio
import wave
import datetime
import serial

#シリアル通信したいArduinoの指定
ser = serial.Serial('COM3',9600)

#録音関数を定義する

class_name = "new"
# class_name = "broken"
# class_name = "scoring"

def recording(output_path):
    #録音時間指定
    CHUNK = 2 ** 10
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    record_time = 0.4
    p = pyaudio.PyAudio()
    #録音開始
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Now Recording: " + output_path + " ...")

    frames = []
    for i in range(0, int(RATE / CHUNK * record_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Done.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    # 録音終了
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

i = 1
#   シリアル通信
while True:
    if ser.in_waiting > 0:  # 受信データがあるか確認
        data = ser.readline().decode().rstrip()  # データの受信

        recording_started = False
        samples_recorded = 0

        if data == "1" and not recording_started:
            print("Prepare for record")
            recording_started = True
            samples_recorded = 0
            #保存パス指定
            dt_now = datetime.datetime.now()
            file_name = str(dt_now.strftime('%m%d')) + "_" + str(i)
            output_path = f"./data/{class_name}/" + file_name + ".wav"
            i = i + 1


        if recording_started:
            # 録音データ
            recording(output_path)


ser.close()