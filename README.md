# Shutter Gear Damage Prediction System
This repository contains the code and resources for a system designed to assess the current state of a shutter's gear by analyzing the sound it produces and predicting its lifespan. This project aims to develop a cost-effective and efficient method for monitoring the health of gears and bearings in shutters using voice analysis and neural networks.

## Table of Contents
- [Background](#background)
- [Research Motivation](#research-motivation)
- [System Configuration](#system-configuration)
- [Experimental Setup](#experimental-setup)
- [Neural Network Configuration](#neural-network-configuration)
- [Results & Conclusion](#results--conclusion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Background
The goal of this research is to create a predictive system that can detect subtle damage in shutters promptly, thus mitigating the risk of more severe issues and promoting an extended lifespan for the shutters. The system utilizes a neural network-based voice analysis program to distinguish between normal and damaged sounds, providing maintainers with remote monitoring capabilities.

## Research Motivation
Traditional shutter maintenance relies on periodic inspections and reactive repairs. This system aims to introduce a real-time, IoT-based approach to shutter maintenance, enabling early detection of potential issues and reducing overall maintenance costs.

## System Configuration
The system is designed to collect sound data from gears and bearings and upload it to a cloud server for analysis. The main components include:
- **Recorder**: Installed inside the shutter to capture sound.
- **Analysis Tool**: A neural network that processes the sound data and predicts damage.
- **Cloud Service**: For data storage and remote analysis.

## Experimental Setup
An experimental device was created to simulate various types of gear and bearing damage. This device includes:
- **Mechanical Part**: Simulates different types of damage using gears and bearings.
- ![Experimental Device1](https://github.com/Mavrick-mao/Gear_Analysis/blob/main/test/IMG_3534%20(1).jpg)
- **Electric Management**: Controls the recording process using an Arduino and Python script.
- ![Experimental Device2](https://github.com/Mavrick-mao/Gear_Analysis/blob/main/test/IMG_3535%20(1).jpg)

## Neural Network Configuration
Two neural network models were explored: Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN). The CNN model was selected due to its efficiency and accuracy. The process involved:
- **Noise Pre-processing**: Transforming sound data into images using Mel Spectrogram.
- **Data Augmentation**: Adding noise and shifting sound to create a diverse dataset.
- **Model Training**: Training the CNN model to classify different types of damage.

## Results & Conclusion
The trained neural network achieved a high accuracy rate of 99.54% in training and 99.57% in validation. The results demonstrate the model's ability to effectively classify different types of damage based on sound analysis. However, further optimization is needed to address the loss metrics and enhance the model's robustness.

## Future Work
Future research will focus on:
- Validating the model's performance in real-world scenarios with ambient noise.
- Improving the model's generalization capabilities to handle unseen data.
- Conducting real-world tests in various shutter environments to refine the system.

## Acknowledgements
Special thanks to Professor Yoshitaka Adachi for his guidance and support, and to all colleagues and industry partners who contributed to the success of this research.


Dataset:
https://huggingface.co/datasets/Maverick-mao/gear_vocie/tree/main
