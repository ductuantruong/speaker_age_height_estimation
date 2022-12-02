# Speaker Profiling

This Repository contains the code for estimating the Age and Height of a speaker with their speech signal. This repository uses [s3prl](https://github.com/s3prl/s3prl) library to load various upstream models like wav2vec2, CPC, TERA etc. This repository uses TIMIT dataset. 

**_NOTE:_**  If you want to run the single encoder model, you should checkout the `singleEncoder` branch and follow the README in that branch.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

## Usage

### Download the TIMIT dataset
```bash
wget https://data.deepai.org/timit.zip
unzip timit.zip -d 'path to timit data folder'
```

### Prepare the dataset for training and testing
```bash
python TIMIT/prepare_timit_data.py --path='path to timit data folder'
```

### Update Config and Logger
Update the config.json file to update the upstream model, batch_size, gpus, lr, etc and change the preferred logger in train_.py files. Create a folder 'checkpoints' to save the best models. If you wish to perform narrow band experiment, just set narrow_band as true in config.json file. For DenseLoss experiment, there are some special arguments. For example:
* `--model_task='a'` for downstream task of age and gender estimation; `--model_task='h'` for downstream task of height and gender estimation; default `--model_task` for downstream task of height, age and gender estimation
* `--upstream_model=wav2vec2` for upstream model of wav2vec2; `--upstream_model=wavlm` for upstream model of WavLMBase+;

### Training
```bash
python train_timit.py --data_path='path to final data folder' --speaker_csv_path='path to this repo/SpeakerProfiling/Dataset/data_info_height_age.csv'
```

Example:
```bash
python train_timit.py --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --speaker_csv_path=/notebooks/SpeakerProfiling/Dataset/data_info_height_age.csv
```

### Testing
```bash
python test_timit.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test_timit.py --data_path=/notebooks/SpeakerProfiling/TIMIT_Dataset/wav_data/ --model_checkpoint=checkpoints/epoch=1-step=245-v3.ckpt
```

### Pretrained Model
We have uploaded a pretrained model of our experiments. You can download the from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/ductuan001_e_ntu_edu_sg/EtOAqMxQgP9Mpguu_YbZUrIBWTdN04RAdn8jkV3dk7o9Hg?e=U0q94O).

Download it and put it into the model_checkpoint folder.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen

