import os
import random
import shutil
import pandas as pd
import librosa
import librosa.display
import wave
import pyaudio
import soundfile as sf

NOISE_PATH = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\noise\ambulance_pass.mp3"
SAMPLE_PATH = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data\normal\0925_2.wav"

def select(path,start_time = 2.0, duration = 0.4):
    y, sr = librosa.load(path)

    start_sample = int(start_time * sr)
    end_sample = start_sample + int(duration * sr)

    selected_part = y[start_sample:end_sample]

    return selected_part

def add(audio_path1, audio_path2):
    # それぞれの音声ファイルを読み込む
    y1, sr1 = librosa.load(audio_path1)
    y2, sr2 = librosa.load(audio_path2)

    # サンプリングレートが異なる場合はリサンプリング
    if sr1 != sr2:
        y2 = librosa.resample(y2, sr2, sr1)
        sr2 = sr1

    # サンプル数を揃える
    min_samples = min(len(y1), len(y2))
    y1 = y1[:min_samples]
    y2 = y2[:min_samples]

    # 二つの音源を足し合わせる
    combined_audio = y1 + y2

    return combined_audio, sr1

if __name__ == "__main__":

    combine, sr = add(NOISE_PATH , SAMPLE_PATH )
    # 保存する
    sf.write('combine.wav', combine, sr)


