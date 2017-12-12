import torch
from torch import nn


# TODO Normalize inputs
class ReducedAlexNetClassifier(nn.Module):
    def __init__(self, output_size=256):
        super().__init__()
        self.ac = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096 , 4096 // 2),  # XXX Size here depends on feature outputs
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096 // 2, output_size),
            #nn.ReLU(inplace=True),
            #nn.Linear(4096 // 4, output_size),
        )

    def forward(self, x):
        x = self.ac(x)
        return x


class ReducedAlexNetFeatures(nn.Module):
    """Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """
    def __init__(self, input_dim=2, channels=3):
        super().__init__()

        if input_dim == 1:
            conv = nn.Conv1d
            maxpool = nn.MaxPool1d
        elif input_dim == 2:
            conv = nn.Conv2d
            maxpool = nn.MaxPool2d
        elif input_dim == 3:
            conv = nn.Conv3d
            maxpool = nn.MaxPool3d
        else:
            assert False

        self.features = nn.Sequential(
            conv(channels, 64, kernel_size=11, stride=4, padding=2),
            nn.Hardtanh(inplace=True),
            maxpool(kernel_size=3, stride=2),
            conv(64, 128, kernel_size=5, padding=2),
            nn.Hardtanh(inplace=True),
            maxpool(kernel_size=3, stride=2),
            conv(128, 128, kernel_size=3, padding=1),
            nn.Hardtanh(inplace=True),
            #conv(384, 256, kernel_size=3, padding=1),
            #nn.Hardtanh(inplace=True),
            #conv(256, 512, kernel_size=3, padding=1),
            #nn.Hardtanh(inplace=True),
            maxpool(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = torch.cat((x[0], x[1]), 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class AlexNetFeatures(nn.Module):
    """Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """
    def __init__(self, input_dim=2, channels=3):
        super().__init__()

        if input_dim == 1:
            conv = nn.Conv1d
            maxpool = nn.MaxPool1d
        elif input_dim == 2:
            conv = nn.Conv2d
            maxpool = nn.MaxPool2d
        elif input_dim == 3:
            conv = nn.Conv3d
            maxpool = nn.MaxPool3d
        else:
            assert False

        self.features = nn.Sequential(
            conv(channels, 64, kernel_size=11, stride=8, padding=2),
            nn.ReLU(inplace=True),
            maxpool(kernel_size=3, stride=2),
            conv(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            maxpool(kernel_size=3, stride=2),
            conv(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            conv(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            conv(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            maxpool(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = torch.cat((x[0], x[1]), 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class BasicFeatures(nn.Module):
    def __init__(self, input_dim=2, channels=3):
        super().__init__()

        if input_dim == 1:
            conv = nn.Conv1d
            maxpool = nn.MaxPool1d
        elif input_dim == 2:
            conv = nn.Conv2d
            maxpool = nn.MaxPool2d
        elif input_dim == 3:
            conv = nn.Conv3d
            maxpool = nn.MaxPool3d
        else:
            assert False

        self.f1 = nn.Sequential(
            conv(channels, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            maxpool(kernel_size=3, stride=1))
        self.f2 = nn.Sequential(
            conv(channels, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            maxpool(kernel_size=3, stride=1))


    def forward(self, x):
        x = torch.cat((self.f1(x[0]), self.f2(x[1])), 2)
        x = x.view(x.size(0), -1)
        return x


class AlexNetClassifier(nn.Module):
    def __init__(self, output_size=1024):
        super().__init__()
        self.ac = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6144 // 8, 4096 // 4),  # XXX Size here depends on feature outputs
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096 // 4, 4096 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(4096 // 4, output_size),
        )

    def forward(self, x):
        x = self.ac(x)
        return x


class BasicClassifier(nn.Module):
    def __init__(self, output_size=1024):
        super().__init__()
        self.ac = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 100),  # XXX Size here depends on feature outputs
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, output_size),
        )

    def forward(self, x):
        x = self.ac(x)
        return x


class FutureFrame(nn.Module):
    def __init__(self, output_dim, channels=3):
        super().__init__()

        if output_dim == 1:
            conv = nn.Conv1d
        elif output_dim == 2:
            conv = nn.Conv2d
        elif output_dim == 3:
            conv = nn.Conv3d
        else:
            assert False

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            conv(256 * 3, 384, kernel_size=3, padding=1),  # XXX Size here depends on feature outputs
            conv(384, 384, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            conv(384, 192, kernel_size=3, padding=1),
            conv(192, 192, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            conv(192, 64, kernel_size=5, padding=2),
            conv(64, 64, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2),
            conv(64, 32, kernel_size=5, padding=2),
            conv(32, 32, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2),
            conv(32, 3, kernel_size=5, padding=2),
            conv(3, channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Model(nn.Module):
    def __init__(self, dim, channels, output_size):
        super().__init__()
        self.af = ReducedAlexNetFeatures(input_dim=dim, channels=channels)
        #self.af = BasicFeatures(input_dim=dim, channels=channels)
        self.ac = ReducedAlexNetClassifier(output_size=output_size)
        #self.ac = BasicClassifier(output_size=output_size)
        self.ff = FutureFrame(output_dim=dim, channels=channels)
        self.dim = dim

    def forward(self, x):
        features = self.af(x)

        position = self.ac(features)

        prediction = None
        # TODO: Is this right? - Alex
        #for i in range(self.dim):
        #    features = features.unsqueeze(i + 2)
        #prediction = self.ff(features)
        return position, prediction

    def save(self, path='saved_model'):
        torch.save(self.state_dict(), path)

    def load(self, path='saved_model'):
        print("loading saved model")
        self.load_state_dict(torch.load(path))
