import torch.nn as nn
from math import log10

class PSNR():
    def __init__(self, peak=1.0):
        #self.l2loss = nn.MSELoss()
        self.peak = peak
        pass

    # Formula:
    #   psnr = 10*log(peak^2/mse)
    def __call__(self, x, y):
        #mse = self.l2loss(x, y).data[0]
        mse = ((x-y)**2).mean()
        pval = 10 * log10(self.peak**2 / mse)
        return pval
