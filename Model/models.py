import torch
import torch.nn as nn
import s3prl.hub as hub

class Wav2vec2BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        self.mfcc_extractor = getattr(hub, 'mfcc')()
        
        for param in self.upstream.parameters():
            param.requires_grad = True
       
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
        
        self.mfcc_cnn = nn.Sequential(
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

        mfcc_encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=39, nhead=3, batch_first=True)
        self.transformer_mfcc_encoder_M = torch.nn.TransformerEncoder(mfcc_encoder_layer_M, num_layers=num_layers//2)

        mfcc_encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=39, nhead=3, batch_first=True)
        self.transformer_mfcc_encoder_F = torch.nn.TransformerEncoder(mfcc_encoder_layer_F, num_layers=num_layers//2)

        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)
        
        self.fcM = nn.Linear(2*(feature_dim+39), 1024+39)
        self.fcF = nn.Linear(2*(feature_dim+39), 1024+39)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024+39, 1)
        self.age_regressor = nn.Linear(1024+39, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*(1024+39), 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x_input = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x_input)['last_hidden_state']
        x_mfcc = self.mfcc_extractor(x_input)["last_hidden_state"]

        xM_mfcc = self.transformer_mfcc_encoder_M(x_mfcc)
        xF_mfcc = self.transformer_mfcc_encoder_F(x_mfcc)
        xM_mfcc = self.dropout(torch.cat((torch.mean(xM_mfcc, dim=1), torch.std(xM_mfcc, dim=1)), dim=1))
        xF_mfcc = self.dropout(torch.cat((torch.mean(xF_mfcc, dim=1), torch.std(xF_mfcc, dim=1)), dim=1))

        xM = self.transformer_encoder_M(x)
        xF = self.transformer_encoder_F(x)
        xM = self.dropout(torch.cat((torch.mean(xM, dim=1), torch.std(xM, dim=1)), dim=1))
        xF = self.dropout(torch.cat((torch.mean(xF, dim=1), torch.std(xF, dim=1)), dim=1))

        xM = torch.cat((xM, xM_mfcc), dim=1)
        xF = torch.cat((xF, xF_mfcc), dim=1)

        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender