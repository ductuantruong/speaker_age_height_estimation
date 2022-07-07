from email.utils import encode_rfc2231
import torch
import torchaudio

import os
import argparse
from tqdm import tqdm

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--device',
                       metavar='device',
                       type=str,
                       default='cuda',
                       help='the type of computing device')
my_parser.add_argument('--data_path',
                       metavar='data_path',
                       type=str,
                       default='data',
                       help='the path to dataset folder')
my_parser.add_argument('--wav_dir',
                       metavar='wav_dir',
                       type=str,
                       default='wav_data',
                       help='the path to wav folder')
my_parser.add_argument('--upstream_model',
                       metavar='upstream_model',
                       type=str,
                       default='wav2vec2',
                       help='upstream model name')
my_parser.add_argument('--data_type',
                       metavar='data_type',
                       type=str,
                       default='TRAIN',
                       help='the path to wav folder')
my_parser.add_argument('--encoder_layer',
                       metavar='encoder_layer',
                       type=str,
                       nargs="+",
                       default=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                       help='the path to wav folder')
args = my_parser.parse_args()

device = args.device
upstream_model_name = args.upstream_model
original_data_dir = args.data_path
wav_dir = args.wav_dir
list_encoder_layer = args.encoder_layer
list_encoder_layer = list(map(lambda x: 'hidden_state_' + x, list_encoder_layer))

feature_folder_path = os.path.join(original_data_dir, upstream_model_name)
data_type = args.data_type
upstream_model = torch.hub.load('s3prl/s3prl', upstream_model_name).to(device)

for encoder_layer in list_encoder_layer:
    os. makedirs(os.path.join(feature_folder_path, encoder_layer, data_type), exist_ok=True)
    
wav_folder_path = os.path.join(original_data_dir, wav_dir, data_type)
print("Processing {} ...".format(wav_folder_path))
list_file = os.listdir(wav_folder_path)
for wav_file in tqdm(list_file):
    wav_file_name = wav_file[:-4]
    for encoder_layer in list_encoder_layer:
        save_path = os.path.join(feature_folder_path, encoder_layer, data_type, '{}.pt'.format(wav_file_name))
        if os.path.exists(save_path):
            continue
        wav, sr = torchaudio.load(os.path.join(wav_folder_path, wav_file))
        wav = wav.to(device)
        while True:
            try:
                feature = upstream_model(wav)[encoder_layer]
                break
            except:
                pass
        del wav
        torch.save(feature, save_path)
        del feature
            
