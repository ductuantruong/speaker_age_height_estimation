import torch
import torch.nn as nn
import fairseq

class Wav2vec2BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)

        n_encoder_layer = len(self.upstream.model.encoder.layers)
        for i in range(n_encoder_layer):
            self.upstream.model.encoder.layers.insert(2*i + 1, ResidualAdapterBlock(feature_dim, 384))

        for param in self.upstream.parameters():
            param.requires_grad = False
       
        for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True
        
        for i, encoder_layer in enumerate(self.upstream.model.encoder.layers):
            if i % 2 != 0:
                for param in self.upstream.model.encoder.layers[i].parameters():
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
        x = self.upstream(x)['hidden_state_2']
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




class Wav2vec2BiEncoderAgeEstimation(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        """
        state_dict = torch.load('/home/project/12001458/ductuan0/SpeakerProfiling/libri960_basemodel_sre0810_finetune_48epoch.pt')['model']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict['model.' + key] = state_dict[key]
        del state_dict
        self.upstream.load_state_dict(new_state_dict)
        """

        for param in self.upstream.parameters():
            param.requires_grad = True
       
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
        
        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)
        
        self.fcM = nn.Linear(2*feature_dim, 1024)
        self.fcF = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

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
        age = self.age_regressor(output)
        return age, gender


class ResidualAdapterBlock(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.ln = nn.LayerNorm(normalized_shape=(self.input_dim, ))
        self.down_proj = nn.Linear(self.input_dim, self.proj_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(self.proj_dim, self.input_dim)

    def forward(self, x, self_attn_padding_mask=None, need_weights=None):
        residual = x
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x + residual, None
