from turtle import forward
from matplotlib.style import context
import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2vec2BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.parameters():
            param.requires_grad = True
       
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
        
        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)
        
        self.fcM = nn.Linear(feature_dim, 512)
        self.fcF = nn.Linear(feature_dim, 512)
        
        self.sap = SAPoolingLayer(feature_dim=feature_dim)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(512, 1)
        self.age_regressor = nn.Linear(512, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        xM = self.transformer_encoder_M(x)
        xF = self.transformer_encoder_F(x)
        xM = self.sap(xM)
        xF = self.sap(xF)
        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender
    
    
class SAPoolingLayer(nn.Module):
    def __init__(self, feature_dim=756):
        super(SAPoolingLayer, self).__init__()
        self.feature_dim = feature_dim
        self.context_weight = nn.Parameter(torch.Tensor(feature_dim, 1))
        self.context_weight.data.normal_(0.0, 0.05)
        self.fc_layer = nn.Sequential(
                            nn.Linear(feature_dim, feature_dim), 
                            nn.ReLU()
                        ) 
    def forward(self, x):
        x_input = x
        x = self.fc_layer(x)
        x = torch.matmul(x, self.context_weight)
        attention_score = F.softmax(x)
        attention_score = attention_score.expand(-1, -1, self.feature_dim)
        output = torch.mul(x_input, attention_score)
        output = torch.sum(output, dim=1)
        return output