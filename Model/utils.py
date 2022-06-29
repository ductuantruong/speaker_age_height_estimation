import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch.nn.functional import cross_entropy, mse_loss
import torchvision as tv

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
        
        self.loss_height = mse_loss(input=height_pred, target=height_target)
        self.loss_height_var = torch.exp(-self.log_var_height) * self.loss_height + self.log_var_height

        self.loss_age = mse_loss(input=age_pred, target=age_target)
        self.loss_age_var = torch.exp(-self.log_var_age) * self.loss_age + self.log_var_age

        self.loss = self.loss_gender + self.loss_height + self.loss_age
        
        self.loss_var = self.loss_gender_var + self.loss_height_var + self.loss_age_var

        return self.loss_var


def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def frame_signal(signal: torch.Tensor,
                 frame_length: int,
                 hop_length: int,
                 window: torch.Tensor = None) -> torch.Tensor:

    if window is None:
        window = torch.ones(frame_length, dtype=signal.dtype, device=signal.device)

    if window.shape[0] != frame_length:
        raise ValueError('Wrong `window` length: expected {}, got {}'.format(window.shape[0], frame_length))

    signal_length = signal.shape[-1]

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((1.0 * signal_length - frame_length) / hop_length))

    pad_len = int((num_frames - 1) * hop_length + frame_length)
    if pad_len > signal_length:
        zeros = torch.zeros(pad_len - signal_length, device=signal.device, dtype=signal.dtype)

        while zeros.dim() < signal.dim():
            zeros.unsqueeze_(0)

        pad_signal = torch.cat((zeros.expand(*signal.shape[:-1], -1)[..., :zeros.shape[-1] // 2], signal), dim=-1)
        pad_signal = torch.cat((pad_signal, zeros.expand(*signal.shape[:-1], -1)[..., zeros.shape[-1] // 2:]), dim=-1)
    else:
        pad_signal = signal

    indices = torch.arange(0, frame_length, device=signal.device).repeat(num_frames, 1)
    indices += torch.arange(
        0,
        num_frames * hop_length,
        hop_length,
        device=signal.device
    ).repeat(frame_length, 1).t_()
    indices = indices.long()

    frames = pad_signal[..., indices]
    frames = frames * window

    return frames


class ToTensor1D(tv.transforms.ToTensor):

    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])

        return tensor_2d.squeeze_(0)