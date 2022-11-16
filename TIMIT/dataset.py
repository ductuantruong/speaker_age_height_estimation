from audioop import avg
from random import weibullvariate
from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
from denseweight import DenseWeight

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
        self.model_task = hparams.model_task

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        
        self.narrow_band = hparams.narrow_band
        
        if self.narrow_band:
            self.resampleDown = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
            self.resampleUp = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000) 

        self.h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        self.h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        self.a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        self.a_std = self.df[self.df['Use'] == 'TRN']['age'].std()

        self.df['norm_height'] = (self.df['height'] - self.h_mean)/self.h_std
        self.df['norm_age'] = (self.df['age'] - self.a_mean)/self.a_std
        
        dw = DenseWeight(alpha=1.0)
        self.df['age_weight'] = dw.fit(self.df['norm_age'].to_numpy())
        self.df['height_weight'] = dw.fit(self.df['norm_height'].to_numpy())
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        g_id = file.split('_')[0]
        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        height = self.df.loc[id, 'norm_height']
        age =  self.df.loc[id, 'norm_age']
        age_weight = self.df.loc[id, 'age_weight']
        height_weight = self.df.loc[id, 'height_weight']

        if self.model_task == 'ahg':
            weight = (age_weight + height_weight) / 2
        elif self.model_task == 'a':
            weight = age_weight
        else:
            weight = height_weight

        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file))
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            
        if self.narrow_band:
            wav = self.resampleUp(self.resampleDown(wav))
        
        return wav, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender]), torch.FloatTensor([weight])
