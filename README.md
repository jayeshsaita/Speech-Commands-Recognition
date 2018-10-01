# Speech Commands Recognition in PyTorch

This project implements Speech Commands Recognition using Resnet34. The data used is a subset of [Tensorflow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge). It can detect 10 spoken commands (yes, no, on, off, up, down, left, right, go, stop) with an accuracy of 90.4%. It uses techniques like [Stochastic Gradient Descent with Restarts](https://arxiv.org/abs/1608.03983) and [Snapshot Ensembling](https://arxiv.org/abs/1704.00109). The model was trained on Google Colab.
**Read more about this project on my [blog](https://towardsdatascience.com/ok-google-how-to-do-speech-recognition-f77b5d7cbe0b)**.

### Requirements

* PyTorch
* Torchvision
* Librosa
* numpy
* matplotlib

Colab Notebook contains cell to install the above requirements.

## Usage
To try out this project, follow these steps:
1. Clone this repo in your Google Drive
2. Add the datasets and saved checkpoints (from links mentioned) in respective folders in your drive
3. Open training.ipynb (to see training code) or demo.ipynb (to see a demo of model) in Google Colaboratory and run the cells
