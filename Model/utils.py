import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch.nn.functional import cross_entropy, mse_loss, kl_div

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class UncertaintyLoss(Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.loss = None
        self.loss_height = None
        self.loss_gender = None
        self.log_var_height = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32, device=device))
        self.log_var_kl_height = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32, device=device))
        self.log_var_gender = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32, device=device))

    def forward(self, input, target, input_h_dist, target_h_dist):
        pred_arr = torch.split(input, input.shape[0]//2)
        height_pred, gender_pred = pred_arr

        target_arr = torch.split(target, target.shape[0]//2)
        height_target, gender_target = target_arr
        
        self.loss_gender = mse_loss(input=gender_pred, target=gender_target)
        self.loss_gender_var = torch.exp(-self.log_var_gender) * self.loss_gender + self.log_var_gender
        
        self.loss_height = mse_loss(input=height_pred, target=height_target)
        self.loss_height_var = torch.exp(-self.log_var_height) * self.loss_height + self.log_var_height

        self.kl_loss_height = kl_div(input=input_h_dist, target=target_h_dist)
        self.kl_loss_height_var = torch.exp(-self.log_var_kl_height) * self.kl_loss_height + self.log_var_kl_height

        self.loss = self.loss_gender + self.loss_height + self.kl_loss_height
        
        self.loss_var = self.loss_gender_var + self.loss_height_var + self.kl_loss_height_var

        return self.loss_var

class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_height, end_height):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_height = start_height
        self.end_height = end_height

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start_height, self.end_height + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()
        
        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss
