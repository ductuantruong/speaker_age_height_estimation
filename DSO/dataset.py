from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
import random

class DSODataset(Dataset):
    def __init__(self,
    wav_folder,
    language,
    hparams,
    ):
        self.language = language
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = hparams.speaker_csv_path
        self.test_csv_file = hparams.test_speaker_csv_path
        self.df = pd.read_csv(self.test_csv_file, index_col='ID')
        self.train_df = pd.read_csv(self.csv_file)

        self.speaker_list = self.df.index.values.tolist()
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        speaker_id = int(file.split('.')[0])
        id = self.df[(self.df['Language'] == self.language) & (self.df['utt_begin_id'] <= speaker_id) & (self.df['utt_end_id'] >= speaker_id)].index
        gender = self.gender_dict[self.df.loc[id, 'Gender'].item()]
        height = self.df.loc[id, 'Height'].item()
        age =  self.df.loc[id, 'Age'].item()
        
        wav, sr = torchaudio.load(os.path.join(self.wav_folder, file))
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
        
        h_mean = self.train_df[self.train_df['Use'] == 'TRN']['height'].mean()
        h_std = self.train_df[self.train_df['Use'] == 'TRN']['height'].std()
        a_mean = self.train_df[self.train_df['Use'] == 'TRN']['age'].mean()
        a_std = self.train_df[self.train_df['Use'] == 'TRN']['age'].std()
        
        height = (height - h_mean)/h_std
        age = (age - a_mean)/a_std
        
        return wav, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender]), file
