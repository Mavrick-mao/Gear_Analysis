import random

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np



class AudioDataset(Dataset):
    def __init__(self, annotations_file,noise_class,win_length = 1024,n_fft = 1024,n_mels = 128,hop_length = 137):
        self.annotations = pd.read_csv(annotations_file)
        self.noise_class = noise_class
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        file_path = self.__get_audio_path__(idx)
        label = self.__get_audio_label__(idx)

        y, fs = librosa.load(file_path,sr = 44100)

        if self.noise_class == "whitenoise":
            y = self.__white_noise__(y)
        elif self.noise_class == "shiftsound":
            y = self.__shift_sound__(y, random.random() * 10)
        elif self.noise_class == "mix":
            y = self.__white_noise__(y)
            y = self.__shift_sound__(y, random.random() * 10)

        fbank = librosa.feature.melspectrogram(y=y,
                                               sr=fs,
                                               win_length=self.win_length,
                                               n_fft=self.n_fft,
                                               n_mels=self.n_mels,
                                               hop_length = self.hop_length
                                               )
        fbank_db = librosa.power_to_db(fbank)
        features = torch.from_numpy(fbank_db).float()
        features = features.unsqueeze(0)

        return features, label

    def __white_noise__(self,y,rate=0.01):
        return y + rate*np.random.rand(len(y))

    def __shift_sound__(self,y, rate=2):
        return np.roll(y, int(len(y)//rate))

    def __get_audio_path__(self,idx):
        return self.annotations.iloc[idx,0]

    def __get_audio_label__(self,idx):
        return self.annotations.iloc[idx,1]


if __name__ == "__main__":
    path = "Sampling Data/test_data.csv"
    result = AudioDataset(path, "mix")
    print(result)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    for i in range(4):
        for idx, (features, label) in enumerate(result):
            print(features)
            print(features.shape)

            # Plot on the corresponding subplot
            row = i // 2
            col = i % 2
            ax = axes[row, col]

            ax.imshow(features.squeeze(0).numpy(), cmap='viridis', origin='lower', aspect='auto')
            ax.set_xlabel('Time')
            ax.set_ylabel('Mel Frequency')
            ax.set_title(f'Label: {i}')
            ax.axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()