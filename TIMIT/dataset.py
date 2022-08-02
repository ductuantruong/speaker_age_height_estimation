from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from torch.distributions.normal import Normal
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
        
        probability = 0.5
        if self.is_train and random.random() <= probability:
            mixup_idx = random.randint(0, len(self.files)-1)
            mixup_file = self.files[mixup_idx]
            mixup_id = mixup_file.split('_')[0][1:]
            mixup_gender = self.gender_dict[self.df.loc[mixup_id, 'Sex']]
            mixup_height = self.df.loc[mixup_id, 'height']
            mixup_age =  self.df.loc[mixup_id, 'age']

            mixup_wav, _ = torchaudio.load(os.path.join(self.wav_folder, mixup_file))

            if(mixup_wav.shape[0] != 1):
                mixup_wav = torch.mean(mixup_wav, dim=0) 
            
            if self.narrow_band:
                mixup_wav = self.resampleUp(self.resampleDown(mixup_wav))

            if(mixup_wav.shape[1] < wav.shape[1]):
                cnt = (wav.shape[1]+mixup_wav.shape[1]-1)//mixup_wav.shape[1]
                mixup_wav = mixup_wav.repeat(1,cnt)[:,:wav.shape[1]]
            
            if(wav.shape[1] < mixup_wav.shape[1]):
                cnt = (mixup_wav.shape[1]+wav.shape[1]-1)//wav.shape[1]
                wav = wav.repeat(1,cnt)[:,:mixup_wav.shape[1]]
            
            alpha = 1
            lam = np.random.beta(alpha, alpha)
            
            wav = lam*wav + (1-lam)*mixup_wav
            height = lam*height + (1-lam)*mixup_height
            age = lam*age + (1-lam)*mixup_age
            gender = lam*gender + (1-lam)*mixup_gender

        label_dist_transform = Normal(torch.tensor([height]), torch.tensor([4.0]))
        heigh_dist = label_dist_transform.cdf(torch.Tensor(list(range(int(height) - 5, int(height) + 5 + 1))))
        return wav, torch.FloatTensor([height]), torch.FloatTensor([age]), heigh_dist, torch.FloatTensor([gender])
