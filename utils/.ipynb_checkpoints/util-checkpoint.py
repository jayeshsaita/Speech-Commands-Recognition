import librosa
import matplotlib.pyplot as plt
import numpy as np
import os 
from tqdm import tqdm
from scipy import signal
import torch
from torchvision import datasets, transforms, models

def log_spectogram(audio, sampling_rate, window_size=20, step_size=10, eps=1e-10):
    #nperseg - n per segment, is the size of window for Short time fourier transform (STFT)
    #here we take 20% (window_size=20) of sampling_rate as window size
    #Note:Sampling rate is 44.1 KHz (kilo hertz), so to take 20%, we divide by 1000 instead of 100
    #Example:sampling_rate=44100
    #nperseg = 20 * 44100/1000 = 882 (samples)
    nperseg = int(round(window_size * sampling_rate / 1000))
    
    #noverlap is the number of samples to overlap between current window and next windows
    #Here we do 50% overlap (since step size=10, window_size=20, step_size is 50% of window_size)
    #Example:step_size=10
    #noverlap = 10 * 44100/1000 = 441 (samples)
    noverlap = int(round(step_size * sampling_rate / 1000))

    freqs, times, specgram = signal.spectrogram(audio,sampling_rate, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
    
    #We return the log of spectogram
    #We are using log of spectrogram since it is known to be closer to how human ears work
    #Note: We are adding eps to spectrogram before performing log to avoid errors since log(0) is undefined
    #eps - epsilon - very small value close to 0, here it is 1e-10 => 0.0000000001
    return freqs, times, np.log(specgram.T.astype(np.float32) + eps)


def audio_to_spectrogram(src_dir, dest_dir):

    # src_dir = '../datasets/data_audio/train/'
    # dest_dir = '../datasets/data_images/train/'
    dirs = ['yes','no','on','off','up','down','left','right','go','stop']

    for dir in dirs:
        audio_files = os.listdir(f'{src_dir}{dir}')
        print(f'Generating Spectograms for {dir} directory')
        for curr_file in audio_files:
            data, sr = librosa.load(f'{src_dir}{dir}/{curr_file}', sr=16000)
            freq, times, spec = log_spectogram(data, sr)
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            ax.imshow(spec.T, aspect='auto', origin='lower')
            ax.axis('off')
            fig.savefig(f'{dest_dir}{dir}/{curr_file[:-4]}.jpg')     
            plt.close(fig)

def get_mean_and_std(train_path, num_workers):
    
    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])
    dataset = datasets.ImageFolder(root=train_path, transform=tfms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    pop_mean = []
    pop_std = []

    print('Calculating mean and std of training dataset')
    for data in dataloader:
        np_image = data[0].numpy()

        batch_mean = np.mean(np_image, axis=(0,2,3))
        batch_std = np.std(np_image, axis=(0,2,3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std.append(batch_std)
        
        
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)

    return [pop_mean, pop_std]
    
def demo(path_model):
    state = torch.load(path_model, map_location='cpu')
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model.load_state_dict(state['model'])
    model.eval()

    src_dir = 'demo/audio/'
    dest_dir = 'demo/images/demo'
    audio_files = os.listdir(f'{src_dir}')
    print(f'Generating Spectograms')
    file_names = os.listdir(src_dir)

    for curr_file in audio_files:
        data, sr = librosa.load(f'{src_dir}/{curr_file}', sr=16000)
        freq, times, spec = log_spectogram(data, sr)
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(spec.T, aspect='auto', origin='lower')
        ax.axis('off')
        fig.savefig(f'{dest_dir}/{curr_file[:-4]}.jpg')     
        plt.close(fig)

    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.2582, 0.1298, 0.3936], [0.0526, 0.1985, 0.0859])
    ])

    demo_ds = datasets.ImageFolder('demo/images', transform=tfms)
    demo_dl = torch.utils.data.DataLoader(demo_ds, batch_size=1, shuffle=False)

    print('Performing prediction')

    classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
  
    for i, (demo_input,_) in enumerate(demo_dl):
        output = model(demo_input)
        output_labels = torch.max(output, dim=1)[1]
        print(f'Prediction for {file_names[i]}:{classes[output_labels]}')

