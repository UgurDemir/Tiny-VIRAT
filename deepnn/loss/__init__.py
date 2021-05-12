import torch
import torch.nn as nn

def build(name, **kwargs):

    if name == 'l1':
        l = nn.L1Loss
    elif name == 'l2':
        l = nn.MSELoss
    elif name == 'bce':
        l = nn.BCELoss
    elif name == 'bce_logit':
        l = nn.BCEWithLogitsLoss
    elif name == 'nll':
        l = nn.NLLLoss
    elif name == 'nll2d':
        l = nn.NLLLoss2d
    else:
        raise Exception('Unknown loss name ' + str(name))

    return l(**kwargs)


# Taken from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
def total_variation_loss(image):
    #loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    loss = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])) / image.size(0)
    return loss