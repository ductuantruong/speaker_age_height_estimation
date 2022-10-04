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
        
        self.h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        self.h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        self.a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        self.a_std = self.df[self.df['Use'] == 'TRN']['age'].std()

        self.age_bins = [20, 30, 40, 60, 80]
        self.age_bin_labels = [0, 1, 2, 3]

        self.height_bins = [140, 160, 170, 180, 200]
        self.height_bin_labels = [0, 1, 2, 3]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        speaker_id = id
        g_id = file.split('_')[0]
        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        height = self.df.loc[id, 'height']
        age =  self.df.loc[id, 'age']

        height = (height - self.h_mean)/self.h_std
        age = (age - self.a_mean)/self.a_std
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
            mixup_age =  self.df.loc[mixup_id, 'age']

            mixup_wav, _ = torchaudio.load(os.path.join(self.wav_folder, mixup_file))

            if(mixup_wav.shape[0] != 1):
                mixup_wav = torch.mean(mixup_wav, dim=0) 
            
            if self.narrow_band:
                mixup_wav = self.resampleUp(self.resampleDown(mixup_wav))

            mixup_height = (mixup_height - self.h_mean)/self.h_std
            mixup_age = (mixup_age - self.a_mean)/self.a_std
            
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
        

        close_height_group = min(self.height_bins, key=lambda x:abs(x-height))
        height_group = (round(height//close_height_group) + self.height_bins.index(close_height_group))

        close_age_group = min(self.age_bins, key=lambda x:abs(x-age))
        age_group = (round(age//close_age_group) + self.age_bins.index(close_age_group))

        if 'test' not in self.wav_folder.lower():        
            return wav, torch.FloatTensor([height]), torch.IntTensor([height_group - 1]), torch.FloatTensor([age]), torch.IntTensor([age_group - 1]), torch.FloatTensor([gender])
        else:
            return wav, torch.FloatTensor([height]), torch.IntTensor([height_group - 1]), torch.FloatTensor([age]), torch.IntTensor([age_group - 1]), torch.FloatTensor([gender]), speaker_id
