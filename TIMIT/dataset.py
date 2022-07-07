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
        self.hparams = hparams
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = hparams.speaker_csv_path
        self.df = pd.read_csv(self.csv_file)
        self.list_feature = os.listdir(os.path.join(self.hparams.data_path, self.hparams.upstream_model))
        self.is_train = is_train

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        height = self.df.loc[id, 'height']
        age =  self.df.loc[id, 'age']

        list_wav = []
        for feature in self.list_feature:
            tensor_path = os.path.join(self.hparams.data_path, self.hparams.upstream_model, feature, self.wav_folder.split('/')[-1], file[:-4] + '.pt')
            wav = torch.load(tensor_path, map_location=torch.device('cpu'))
            list_wav.append(wav)
        multi_feature_wav = torch.cat(list_wav, dim=0).transpose(0, 1)

        h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        a_std = self.df[self.df['Use'] == 'TRN']['age'].std()
        
        height = (height - h_mean)/h_std
        age = (age - a_mean)/a_std
        
        return multi_feature_wav.detach(), torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender])
