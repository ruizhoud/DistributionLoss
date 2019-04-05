import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from .binarized_modules import  BinarizeLinear,BinarizeConv2d,HardTanh_bin
from .distrloss_layer import Distrloss_layer

__all__ = ['alexnet_binary_vs_xnor']

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, ratioInfl=1):
        super(AlexNet, self).__init__()
        self.ratioInfl=ratioInfl
        self.activation_func = HardTanh_bin
        self.channels = [3, int(96*self.ratioInfl), int(256*self.ratioInfl),
                        int(384*self.ratioInfl), int(384*self.ratioInfl),
                        256]
        self.features0 = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(self.channels[1], momentum=None),
        )
        self.features1 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            BinarizeConv2d(self.channels[1], self.channels[2], kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(self.channels[2], momentum=None),
        )
        self.features2 = nn.Sequential(
            self.activation_func(),
            BinarizeConv2d(self.channels[2], self.channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[3], momentum=None),
        )
        self.features3 = nn.Sequential(
            self.activation_func(),
            BinarizeConv2d(self.channels[3], self.channels[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[4], momentum=None),
        )
        self.features4 = nn.Sequential(
            self.activation_func(),
            BinarizeConv2d(self.channels[4], self.channels[5], kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(self.channels[5], momentum=None),
        )
        self.features5 = nn.Sequential(
            self.activation_func(),
        )

        self.neurons = [self.channels[5] * 6 * 6, 4096, 4096, num_classes]
        self.classifier0 = nn.Sequential(
            BinarizeLinear(self.neurons[0], self.neurons[1]),
            nn.BatchNorm1d(self.neurons[1], momentum=None),
        )
        self.classifier1 = nn.Sequential(
            self.activation_func(),
            BinarizeLinear(self.neurons[1], self.neurons[2]),
            nn.BatchNorm1d(self.neurons[2], momentum=None),
        )
        self.classifier2 = nn.Sequential(
            self.activation_func(),
            nn.Linear(self.neurons[2], self.neurons[3]),
            nn.BatchNorm1d(self.neurons[3], momentum=None),
            nn.LogSoftmax()
        )

        self.distrloss_layers = []
        for i in range(2,6):
            self.distrloss_layers.append(Distrloss_layer(self.channels[i]))
        for i in range(1,3):
            self.distrloss_layers.append(Distrloss_layer(self.neurons[i]))

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-2},
            20: {'lr': 1e-3},
            40: {'lr': 1e-4},
            50: {'lr': 1e-5},
            60: {'lr': 1e-6},
            64: {'lr': 0},
        }
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
            ])
        }

    def forward(self, x):
        loss = []
        x = self.features0(x)
        x = self.features1(x)
        loss.append(self.distrloss_layers[0](x))
        x = self.features2(x)
        loss.append(self.distrloss_layers[1](x))
        x = self.features3(x)
        loss.append(self.distrloss_layers[2](x))
        x = self.features4(x)
        loss.append(self.distrloss_layers[3](x))
        x = self.features5(x)

        x = x.view(-1, 256 * 6 * 6)

        x = self.classifier0(x)
        loss.append(self.distrloss_layers[4](x))
        x = self.classifier1(x)
        loss.append(self.distrloss_layers[5](x))
        x = self.classifier2(x)

        distrloss1 = sum([ele[0] for ele in loss]) / len(loss)
        distrloss2 = sum([ele[1] for ele in loss]) / len(loss)
        return x, distrloss1.view(1,1), distrloss2.view(1,1)

def alexnet_binary_vs_xnor(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    infl_ratio = 1.
    if 'infl_ratio' in kwargs:
        infl_ratio = kwargs['infl_ratio']
    return AlexNet(num_classes, infl_ratio)
