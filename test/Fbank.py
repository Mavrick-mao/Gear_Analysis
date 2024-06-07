import librosa.display
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

frame_t = 5
hop_length_t = 5
win_length = 1024
hop_length = 160
n_fft = 1024
n_mels = 128
n_mfcc = 20

def plot(file_name,figure_number,title):
    y , fs = librosa.load(file_name , sr = 44100)

    plt.subplot(4,2,figure_number)

    plt.title(title, fontsize=16)
    fbank = librosa.feature.melspectrogram(y=y,
                                           sr=fs,
                                           win_length=win_length,
                                           n_fft=n_fft,
                                           n_mels=128,
                                           hop_length = 137
                                           )
    fbank_db = librosa.power_to_db(fbank, ref=np.max)

    img = librosa.display.specshow(fbank_db, x_axis='time', y_axis='mel', sr=fs, fmax=fs / 2, )
    fig.colorbar(img, format="%+2.f dB")
    plt.clim(-50, 5)

if __name__ == "__main__":

    # 音声ファイルの読み込み
    pitting = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\pitting\1207_2.wav"
    normal  = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\normal\0925_8.wav"
    ibutsu = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\ibutsu\1026_1.wav"
    ripping =r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\ripping\1121_2.wav"
    broken = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\broken\1126_1.wav"
    scoring = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\scoring\1125_2.wav"
    spalling = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\spalling\0929_2.wav"
    corrosion = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\corrosion\1128_3.wav"



    fig = plt.figure(figsize=(11, 18)) #全体のグラフを作成

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plot(normal, 1,"Normal")
    plot(pitting, 3, "Pitting")
    plot(ibutsu, 2, "Foreign Insert(Bearing)")
    plot(ripping, 4, "Rippling")
    plot(broken,5,"Fatigue Breakage")
    plot(scoring, 6, "Scoring")
    plot(corrosion, 7, "Electric Corrosion")
    plot(spalling, 8,"Spalling")
    plt.savefig("photo.png")
    plt.show()