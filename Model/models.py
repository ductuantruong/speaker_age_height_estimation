import torch
import torch.nn as nn

class Wav2vec2BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        n_cnn_layer = len(self.upstream.model.feature_extractor.conv_layers)
        for i in range(n_cnn_layer):
            self.upstream.model.feature_extractor.conv_layers.insert(2*i + 1, AdapterBlock())
        
        for param in self.upstream.parameters():
            param.requires_grad = False
       
        for i, conv_layer in enumerate(self.upstream.model.feature_extractor.conv_layers):
            if i % 2 != 0:
                for param in self.upstream.model.feature_extractor.conv_layers[i].parameters():
                    param.requires_grad = True
        
        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)
        
        self.fcM = nn.Linear(2*feature_dim, 1024)
        self.fcF = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        xM = self.transformer_encoder_M(x)
        xF = self.transformer_encoder_F(x)
        xM = self.dropout(torch.cat((torch.mean(xM, dim=1), torch.std(xM, dim=1)), dim=1))
        xF = self.dropout(torch.cat((torch.mean(xF, dim=1), torch.std(xF, dim=1)), dim=1))
        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender
    
class AdapterBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(512, 512, kernel_size=1)
        self.gn = nn.GroupNorm(num_groups=512, num_channels=512)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.gn(x)
        return x + residual