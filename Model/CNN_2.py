import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from Mel_dataset import AudioDataset

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            #   [64,1,128,128]
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,padding=3,stride=1,bias=True),
            nn.ReLU(),
            #  [64,64,128,128]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1), stride=1,bias=True),
            nn.ReLU(),
            #  [64,128,64,64]
            nn.MaxPool2d(2),
            #  [64,128,64,64]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            #   [64,128,64,64]
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # [64,128,32,32]
            nn.MaxPool2d(2),
            # [64,256,32,32]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # [64,256,32,32]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # [64,256,32,32]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # [64,256,16,16]
            nn.MaxPool2d(2),
            # [64,512,16,16]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # [64,512,16,16]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # [64,512,16,16]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            # [64,512,8,8]
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32768,10000),
            nn.Linear(10000,8),
            nn.Softmax(dim=1)
        )

    def forward(self, input_data):
        predictions = self.model(input_data)
        return predictions

def img_show(input,figure_number):
    plt.subplot(2, 1, figure_number)
    plt.imshow(input.squeeze().detach().numpy().T, cmap='viridis', origin='lower')

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    path = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\test_data.csv"
    AUDIO_DIR = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data\data"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    BATCH_SIZE = 1

    dataset = AudioDataset(path,"none")

    model = Model()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True)
    for i,(input,label) in tqdm(enumerate(dataloader)):
        output = model(input)
        print(output.shape)

        break



