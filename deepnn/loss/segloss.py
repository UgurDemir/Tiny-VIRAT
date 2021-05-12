import torch
import torch.nn as nn
from torch.autograd import Variable

class SegLoss(nn.Module):
    def __init__(self, loss_type='bce', target_real_label=1.0, target_fake_label=0.0):
        super(SegLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None

        if loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif loss_type == 'bce':
            self.loss = nn.BCELoss()
        elif loss_type == 'bce_logit':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == 'ce':
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == 'nll_2d':
            self.loss = nn.NLLLoss2d()
        else:
            raise Exception('Unknown loss type ' + str(loss_type))

    def get_target(self, input, isreal):
        target_tensor = None

        if isreal:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()) or
                            (self.real_label_var.device != input.device))
            if create_label:
                self.real_label_var = input.new_full(input.size(), self.real_label, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()) or
                            (self.real_label_var.device != input.device))

            if create_label:
                self.fake_label_var = input.new_full(input.size(), self.fake_label, requires_grad=False)
            target_tensor = self.fake_label_var

        return target_tensor

    def __call__(self, input, isreal=None, mask=None):
        if mask is None:
            target = self.get_target(input, isreal)
        else:
            target = mask
        return self.loss(input, target)
