import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch.nn.functional import cross_entropy, mse_loss

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
        self.alpha = 2
        self.loss_height = None
        self.loss_age = None
        self.loss_gender = None
        self.log_var_height = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32, device=device))
        self.log_var_age = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32, device=device))
        self.log_var_gender = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32, device=device))
    
    def forward(self, input, target):
        pred_arr = torch.split(input, input.shape[0]//3)
        height_pred, age_pred, gender_pred = pred_arr

        target_arr = torch.split(target, target.shape[0]//3)
        height_target, age_target, gender_target = target_arr
        
        self.loss_gender = mse_loss(input=gender_pred, target=gender_target)
        self.loss_gender_var = torch.exp(-self.log_var_gender) * self.loss_gender + self.log_var_gender
        
        self.loss_height = mse_loss(input=height_pred, target=height_target) \
            + self.alpha * mse_loss(input=torch.add(torch.log(height_pred), 1e-9), target=torch.add(torch.log(height_target), 1e-9))
        self.loss_height_var = torch.exp(-self.log_var_height) * self.loss_height + self.log_var_height

        self.loss_age = mse_loss(input=age_pred, target=age_target) + self.alpha \
            * self.alpha * mse_loss(input=torch.add(torch.log(age_pred), 1e-9) , target=torch.add(torch.log(age_target), 1e-9))
        self.loss_age_var = torch.exp(-self.log_var_age) * self.loss_age + self.log_var_age

        self.loss = self.loss_gender + self.loss_height + self.loss_age
        
        self.loss_var = self.loss_gender_var + self.loss_height_var + self.loss_age_var

        return self.loss_var

