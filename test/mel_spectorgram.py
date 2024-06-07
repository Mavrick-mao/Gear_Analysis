import datetime
import os
import numpy

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from NNdataset_mel import MFCC
from Model.NNmodel.CNN import CNNNetwork

import wandb
from tqdm import tqdm

SAMPLE_RATE = 16400
NUM_SAMPLES = 22050

TRAIN_ANNOTATIONS_FILE = r"Sampling Data/train_data.csv"
TEST_ANNOTATIONS_FILE = r"Sampling Data/test_data.csv"
AUDIO_DIR = r"Sampling Data/data"

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
dataset = MFCC(TRAIN_ANNOTATIONS_FILE,
                   AUDIO_DIR,
                   mel_spectrogram,
                   SAMPLE_RATE,
                   NUM_SAMPLES)
print(f"There are {len(dataset)} samples in the dataset.")

vector, label = dataset[0]

numpy.set_printoptions(threshold=1000)

print(vector)
print(vector.shape)

vector = vector.to('cpu').detach().numpy().copy()

print(vector)
