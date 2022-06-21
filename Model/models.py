import torch
import torch.nn as nn
import s3prl.hub as hub

class Wav2vec2BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        self.mfcc_extractor = getattr(hub, 'mfcc')()
        self.fbank_extractor = getattr(hub, 'fbank')()
        
        for param in self.upstream.parameters():
            param.requires_grad = True
       
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
        
        encoder_layer_wav2vec2 = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_wav2vec2 = torch.nn.TransformerEncoder(encoder_layer_wav2vec2, num_layers=num_layers)
        
        encoder_layer_mfcc = torch.nn.TransformerEncoderLayer(d_model=39, nhead=3, batch_first=True)
        self.transformer_encoder_mfcc = torch.nn.TransformerEncoder(encoder_layer_mfcc, num_layers=num_layers//3)

        encoder_layer_fbank = torch.nn.TransformerEncoderLayer(d_model=240, nhead=4, batch_first=True)
        self.transformer_encoder_fbank = torch.nn.TransformerEncoder(encoder_layer_fbank, num_layers=num_layers//2)

        self.fc = nn.Linear(2*1047, 1024)
        
        self.dropout = nn.Dropout(0.3)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x_input = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]

        x = self.upstream(x_input)['last_hidden_state']
        x_mfcc = self.mfcc_extractor(x_input)["last_hidden_state"]
        x_fbank = self.fbank_extractor(x_input)["last_hidden_state"]

        x_wav2vec2 = self.transformer_encoder_wav2vec2(x)
        x_mfcc = self.transformer_encoder_mfcc(x_mfcc)
        x_fbank = self.transformer_encoder_fbank(x_fbank)
        
        x_wav2vec2 = self.dropout(torch.cat((torch.mean(x_wav2vec2, dim=1), torch.std(x_wav2vec2, dim=1)), dim=1))
        x_mfcc = self.dropout(torch.cat((torch.mean(x_mfcc, dim=1), torch.std(x_mfcc, dim=1)), dim=1))
        x_fbank = self.dropout(torch.cat((torch.mean(x_fbank, dim=1), torch.std(x_fbank, dim=1)), dim=1))

        x = torch.cat((x_wav2vec2, x_mfcc, x_fbank), dim=1)
        x = self.dropout(self.fc(x))
        
        gender = self.gender_classifier(x)
        height = self.height_regressor(x)
        age = self.age_regressor(x)
        return height, age, gender
