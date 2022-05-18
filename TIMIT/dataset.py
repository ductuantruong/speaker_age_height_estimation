from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
import random

class TIMITDataset(Dataset):
    def __init__(self,
    wav_folder,
    hparams,
    is_train=True,
    ):
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = hparams.speaker_csv_path
        self.df = pd.read_csv(self.csv_file)
        self.is_train = is_train

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        
        self.narrow_band = hparams.narrow_band
        
        if self.narrow_band:
            self.resampleDown = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
            self.resampleUp = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000) 
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        g_id = file.split('_')[0]
        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        height = self.df.loc[id, 'height']
        age =  self.df.loc[id, 'age']
        
        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file))
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            
        if self.narrow_band:
            wav = self.resampleUp(self.resampleDown(wav))
        
        return wav, torch.FloatTensor([height]), torch.IntTensor([round(height) - 140]), torch.FloatTensor([age]), torch.IntTensor([round(age)]), torch.FloatTensor([gender])
