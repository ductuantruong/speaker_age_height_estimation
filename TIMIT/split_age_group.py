import os
import argparse
import pandas as pd
import shutil

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--path',
                       metavar='path',
                       type=str,
                       default='data/wav_data/TRAIN_MIX',
                       help='the path to dataset folder')
my_parser.add_argument('--csv_path',
                       metavar='path',
                       type=str,
                       default='Dataset/mix_data_info_height_age.csv',
                       help='the path to dataset folder')
args = my_parser.parse_args()


original_data_dir = args.path

files = os.listdir(original_data_dir)
df = pd.read_csv(args.csv_path)
dst_dir = '/'.join(original_data_dir.split('/')[:-1]) + '/TRAIN_50'
os.mkdir(dst_dir)
for file in files:
    if file.startswith('common'):
        id = df.loc[df['path'] == file.replace('mp3', 'wav'), 'ID'].iloc[0]
    else:
        id = file.split('_')[0][1:]
    g_id = file.split('_')[0]
    age = df.loc[df['ID'] == id, 'age'].iloc[0]
    if age >= 50 and age < 60:
        shutil.copy(os.path.join(original_data_dir, file), dst_dir)