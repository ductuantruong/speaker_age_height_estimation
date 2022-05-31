from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

from python_speech_features import logfbank
from scipy.io import wavfile
import torch.nn.functional as F
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

    def stacker(self, feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        g_id = file.split('_')[0]
        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        height = self.df.loc[id, 'height']
        age =  self.df.loc[id, 'age']
        
        wav_data, sample_rate = torchaudio.load(os.path.join(self.wav_folder, file))
        wav = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
        wav = self.stacker(wav, 4) # [T/stack_order_audio, F*stack_order_audio] || stack_order_audio = 4 theo nhu paper
        wav = torch.from_numpy(wav.astype(np.float32))
        wav = F.layer_norm(wav, wav.shape[1:])

        # if(wav.shape[0] != 1):
            # wav = torch.mean(wav, dim=0)
            
        if self.narrow_band:
            wav = self.resampleUp(self.resampleDown(wav))
        
        h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        a_std = self.df[self.df['Use'] == 'TRN']['age'].std()
        
        height = (height - h_mean)/h_std
        age = (age - a_mean)/a_std

        """
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

            mixup_height = (mixup_height - h_mean)/h_std
            mixup_age = (mixup_age - a_mean)/a_std
            
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
        """        
            
        return wav, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender])
