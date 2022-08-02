from genericpath import exists
from random import random
import numpy as np
import os
import torch
import torchaudio
import argparse
import random
import math

def mix_audio(signal, noise, snr):
    if len(noise) < len(signal):
        len_ratio = len(signal)//len(noise) + 1
        noise = noise.repeat(len_ratio)
    noise = noise[(len(noise)//2 - len(signal)//2):(len(noise)//2 + len(signal)//2 + len(signal)%2)]
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculates the gain to be applied to the noise 
    # to achieve the given SNR
    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))
    return a * signal + b * noise


my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--data_path',
                       metavar='data_path',
                       default='data',
                       type=str,
                       help='the path to dataset folder')
my_parser.add_argument('--noise_path',
                       metavar='noise_path',
                       default='musan',
                       type=str,
                       help='the path to noise dataset folder')
my_parser.add_argument('--noise_type',
                       metavar='noise_type',
                       default='music',
                       type=str,
                       help='the type of noise')

args = my_parser.parse_args()


original_data_dir = args.data_path
noisy_data_dir = args.noise_path 
noisy_wav_path = os.path.join(noisy_data_dir, args.noise_type)
final_data_path = os.path.join(original_data_dir, 'wav_data')

list_snr = [5, 10, 15, 20]

list_data_type = ['TRAIN', 'VAL', 'TEST']
list_noisy_wav = list(filter(lambda file_name: '.wav' in file_name, os.listdir(noisy_wav_path)))

for snr in list_snr:
    for data_type in list_data_type:
        os.makedirs(os.path.join(original_data_dir, 'wav_data_' + args.noise_type, str(snr), data_type), exist_ok=True)

for data_type in list_data_type:
    wav_dir = os.path.join(final_data_path, data_type)
    for wav_file in os.listdir(wav_dir):
        clean_wav, sr = torchaudio.load(os.path.join(wav_dir, wav_file))
        clean_wav = clean_wav.squeeze(0).detach().numpy()
        noisy_wav, sr = torchaudio.load(os.path.join(noisy_wav_path, random.choice(list_noisy_wav)))
        noisy_wav = noisy_wav.squeeze(0).detach().numpy()

        for snr in list_snr:
            mix_wav = mix_audio(clean_wav, noisy_wav, snr)
            path = os.path.join(original_data_dir, 'wav_data_' + args.noise_type, str(snr), data_type, wav_file)
            torchaudio.save(path, torch.from_numpy(mix_wav).unsqueeze(0), sr)        
