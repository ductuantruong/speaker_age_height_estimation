import torch
import torch.nn as nn
import torch.nn.functional as F
import s3prl.hub as hub

class Wav2vec2BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        self.mfcc_extractor = getattr(hub, 'mfcc')()
        
        self.path1_cnn = nn.Conv2d(in_channels=39, out_channels=32, kernel_size=(11,1), padding="same", stride=(1,1))
        self.path2_cnn = nn.Conv2d(in_channels=39, out_channels=32, kernel_size=(1, 9), padding="same", stride=(1,1))
        self.path3_cnn = nn.Conv2d(in_channels=39, out_channels=32, kernel_size=(3, 3), padding="same", stride=(1,1))

        self.path1_bn = nn.BatchNorm1d(num_features=32)
        self.path2_bn = nn.BatchNorm1d(num_features=32)
        self.path3_bn = nn.BatchNorm1d(num_features=32)

        self.avg_pool = nn.AvgPool1d(kernel_size=2, padding="same")

        self.feature_extractor_blk1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding='same', bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(2, 2), padding="same")
        )

        self.feature_extractor_blk2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3,3), stride=1, padding='same', bias=False),
            nn.BatchNorm1d(num_features=96),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(2,2), padding="same")
        )

        self.feature_extractor_blk3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3,3), stride=1, padding='same', bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(2,1) , padding="same")
        )

        self.feature_extractor_blk4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=(3,3), stride=1, padding='same', bias=False),
            nn.BatchNorm1d(num_features=160),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(2,1) , padding="same")
        )

        self.feature_extractor_blk5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(1,1), stride=1, padding='same', bias=False),
            nn.BatchNorm1d(num_features=320),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=320)
        )

        self.dropout = nn.Dropout(0.3)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x_input = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.mfcc_extractor(x_input)["last_hidden_state"]#.transpose(1, 2)
        print(x.shape)
        import time
        time.sleep(1999)
        path1 = self.path1_bn(self.path1_cnn(x))
        path2 = self.path2_bn(self.path2_cnn(x))
        path3 = self.path3_bn(self.path3_cnn(x))

        path1 = self.avg_pool(F.relu(path1))
        path2 = self.avg_pool(F.relu(path2))
        path3 = self.avg_pool(F.relu(path3))
        
        x = nn.cat((path1, path2, path3), dim=1)
        
        x = self.feature_extractor_blk1(x)
        x = self.feature_extractor_blk2(x)
        x = self.feature_extractor_blk3(x)
        x = self.feature_extractor_blk4(x)
        x = self.feature_extractor_blk5(x)
        x = self.dropout(x)
        
        
        gender = self.gender_classifier(x)
        height = self.height_regressor(x)
        age = self.age_regressor(x)
        return height, age, gender
