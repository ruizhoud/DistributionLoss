import torch
import torch.nn as nn
import math
from torch.autograd import Function


class BinarizeF(Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class HardTanh_bin(nn.Module):
    def __init__(self):
        super(HardTanh_bin, self).__init__()
        self.hardtanh = nn.Hardtanh(inplace=False)
        self.binarize = BinarizeF.apply

    def forward(self, input):
        output = self.hardtanh(input)
        output = self.binarize(output)
        return output


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=self.weight.org.sign()
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out = out + self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=self.weight.org.sign()
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out = out + self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
