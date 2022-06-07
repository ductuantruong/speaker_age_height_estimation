import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2vec2BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(0.1)
        
        for param in self.upstream.parameters():
            param.requires_grad = True
            #param.requires_grad = False
       
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False

        self.shared_cnn = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(512, momentum=0.1, affine=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(512, momentum=0.1, affine=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(512, momentum=0.1, affine=True),
        )

        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)

        self.fc_xv = nn.Linear(1024, feature_dim)
        self.fcM = nn.Linear(2*feature_dim, 1024)
        self.fcF = nn.Linear(2*feature_dim, 1024)
        
        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        batch_size = x.size(0)
        T_len = x.size(1)
        x = self.dropout(x)
        x = x.reshape(batch_size * T_len, -1, self.feature_dim).transpose(-1, -2)
        x = self.shared_cnn(x)
        stats = torch.cat((torch.mean(x, dim=2), torch.std(x, dim=2, unbiased=False)), dim=1)
        x = self.fc_xv(stats)
        x = x.reshape(batch_size, T_len, self.feature_dim)
        xM = self.transformer_encoder_M(x)
        xF = self.transformer_encoder_F(x)
        xM = torch.cat((torch.mean(xM, dim=1), torch.std(xM, dim=1)), dim=1)
        xF = torch.cat((torch.mean(xF, dim=1), torch.std(xF, dim=1)), dim=1)
        xM = F.relu(self.dropout(self.fcM(xM)))
        xF = F.relu(self.dropout(self.fcF(xF)))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender
