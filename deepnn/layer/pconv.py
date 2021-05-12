"Taken from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py"
import torch
import torch.nn as nn

class PartialConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
            
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, False)

        self.input_conv.apply(self.weights_init('gaussian'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find(
                    'Linear') == 0) and hasattr(m, 'weight'):
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        return init_fun
