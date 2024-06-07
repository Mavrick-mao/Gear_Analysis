import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from NNdataset_mel import MFCC
# from cnn import CNNNetwork


class CNNNetwork(nn.Module):

    def __init__(self):
        super(CNNNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Flatten(),
            # nn.Linear(96, 24),
            # nn.Linear(24, 4),

        )

    def forward(self, input_data):
        predictions = self.model(input_data)
        return predictions

def create_data_loader(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size,pin_memory=True)
    return dataloader


if __name__ == "__main__":
    print(CNNNetwork())
    BATCH_SIZE = 1
    EPOCHS = 10
    LEARNING_RATE = 0.001

    ANNOTATIONS_FILE = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\train_data.csv"
    AUDIO_DIR = r"C:\Users\maver\PycharmProjects\pythonProject1\data"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = MFCC(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES)
    dataloader = DataLoader(usd, batch_size=64)

    cnn = CNNNetwork()

    step = 0

    for data in dataloader:
        item,target = data
        print(item.shape)
        output = cnn(item)
        print(output.shape)

        break