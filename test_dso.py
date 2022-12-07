from config import TIMITConfig

from argparse import ArgumentParser
from multiprocessing import Pool
import os

from DSO.dataset import DSODataset
from TIMIT.lightning_model_uncertainty_loss import LightningModel

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
import pytorch_lightning as pl

import torch
import torch.utils.data as data

from tqdm import tqdm 
import pandas as pd
import numpy as np

import torch.nn.utils.rnn as rnn_utils
def collate_fn(batch):
    (seq, height, age, gender, idx) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, height, age, gender, seq_length, idx

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default='/home/project/12001458/ductuan0/speaker_age_height_estimation/data/DSO_data')
    parser.add_argument('--speaker_csv_path', type=str, default=TIMITConfig.speaker_csv_path)
    parser.add_argument('--test_speaker_csv_path', type=str, default='/home/project/12001458/ductuan0/speaker_age_height_estimation/DSO/data_info_height_age.csv')
    parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
    parser.add_argument('--num_layers', type=int, default=TIMITConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=TIMITConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
    parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
    parser.add_argument('--upstream_model', type=str, default=TIMITConfig.upstream_model)
    parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        device = 'cpu'
        hparams.gpu = 0
    else:        
        device = 'cuda'
        print(f'Training Model on TIMIT Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')
    


    csv_path = hparams.speaker_csv_path
    df = pd.read_csv(csv_path)
    df.set_index('ID', inplace=True)
    h_mean = df[df['Use'] == 'TRN']['height'].mean()
    h_std = df[df['Use'] == 'TRN']['height'].std()
    a_mean = df[df['Use'] == 'TRN']['age'].mean()
    a_std = df[df['Use'] == 'TRN']['age'].std()

    #Testing the Model
    if hparams.model_checkpoint:
        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
        model.to(device)
        model.eval()
        list_test_set = ['ENGLISH', 'CHINESE']
        for test_set in list_test_set:
            height_pred = []
            height_true = []
            age_pred = []
            age_true = []
            gender_pred = []
            gender_true = []

            speaker_age_pred_dict = {}
            speaker_height_pred_dict = {}

            list_idx = []
            # Testing Dataset
            test_set = DSODataset(
                wav_folder = os.path.join(hparams.data_path, test_set),
                language = test_set.capitalize(),
                hparams = hparams
            )

            ## Testing Dataloader
            testloader = data.DataLoader(
                test_set, 
                batch_size=1, 
                shuffle=False, 
                num_workers=hparams.n_workers,
                collate_fn = collate_fn,
            )

            for batch in tqdm(testloader):
                x, y_h, y_a, y_g, x_len, idx = batch
                x = x.to(device)
                y_h = torch.stack(y_h).reshape(-1,)
                y_a = torch.stack(y_a).reshape(-1,)
                y_g = torch.stack(y_g).reshape(-1,)
                
                y_hat_h, y_hat_a, y_hat_g = model(x, x_len)
                y_hat_h = y_hat_h.to('cpu')
                y_hat_a = y_hat_a.to('cpu')
                y_hat_g = y_hat_g.to('cpu')
                height_pred.append((y_hat_h*h_std+h_mean).item())
                age_pred.append((y_hat_a*a_std+a_mean).item())
                gender_pred.append(y_hat_g>0.5)

                for i, speaker_id in enumerate(idx):
                    if speaker_id not in speaker_age_pred_dict:
                        speaker_age_pred_dict[speaker_id] = []
                        speaker_height_pred_dict[speaker_id] = []

                for i, speaker_id in enumerate(idx):
                    speaker_age_pred_dict[speaker_id].append(y_hat_a*a_std+a_mean)
                    speaker_height_pred_dict[speaker_id].append(y_hat_a*a_std+a_mean)

                height_true.append((y_h*h_std+h_mean).item())
                age_true.append(( y_a*a_std+a_mean).item())
                gender_true.append(y_g[0])
                list_idx.append(idx)

            for speaker_id in list_idx:
                speaker_age_pred_dict[speaker_id] = sum(speaker_age_pred_dict[speaker_id])/len(speaker_age_pred_dict[speaker_id])
                speaker_height_pred_dict[speaker_id] = sum(speaker_height_pred_dict[speaker_id])/len(speaker_height_pred_dict[speaker_id])
                df.at[speaker_id, 'age_prediction'] = round(speaker_age_pred_dict[speaker_id], 2)
                df.at[speaker_id, 'height_prediction'] = round(speaker_height_pred_dict[speaker_id], 2)

            df.to_csv('dso_{}_test_w2w2.csv'.format(test_set))

            female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
            male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

            height_true = np.array(height_true)
            height_pred = np.array(height_pred)
            age_true = np.array(age_true)
            age_pred = np.array(age_pred)

            hmae = mean_absolute_error(height_true[male_idx], height_pred[male_idx])
            hrmse = mean_squared_error(height_true[male_idx], height_pred[male_idx], squared=False)
            amae = mean_absolute_error(age_true[male_idx], age_pred[male_idx])
            armse = mean_squared_error(age_true[male_idx], age_pred[male_idx], squared=False)
            print('Test set {}'.format(test_set))
            print(hrmse, hmae, armse, amae)

            hmae = mean_absolute_error(height_true[female_idx], height_pred[female_idx])
            hrmse = mean_squared_error(height_true[female_idx], height_pred[female_idx], squared=False)
            amae = mean_absolute_error(age_true[female_idx], age_pred[female_idx])
            armse = mean_squared_error(age_true[female_idx], age_pred[female_idx], squared=False)
            print(hrmse, hmae, armse, amae)
            
            hmae = mean_absolute_error(height_true, height_pred)
            hrmse = mean_squared_error(height_true, height_pred, squared=False)
            amae = mean_absolute_error(age_true, age_pred)
            armse = mean_squared_error(age_true, age_pred, squared=False)
            print(hrmse, hmae, armse, amae)

            gender_pred_ = [int(pred[0][0] == True) for pred in gender_pred]
            #print(gender_pred)
            #print(gender_true)
            print(accuracy_score(gender_true, gender_pred_))
            print(confusion_matrix(gender_true, gender_pred_))
            #for i in range(len(gender_pred_)):
            #    if gender_pred_[i] != gender_true[i].item():
            #        print(list_idx[i], gender_pred_[i], gender_true[i].item())

    else:
        print('Model chekpoint not found for Testing !!!')
