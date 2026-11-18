"""Li 2024 Figure 3 Model A, implemented in PyTorch."""
from torch import nn

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[]
        for cin,cout in [(1,64),(64,128),(128,256)]:
            blocks.extend([nn.Conv2d(cin,cout,11),nn.BatchNorm2d(cout),nn.ReLU(),nn.MaxPool2d(2)])
        self.features=nn.Sequential(*blocks)
        self.regressor=nn.Sequential(nn.Flatten(),nn.Linear(256*23*23,256),nn.ReLU(),nn.Linear(256,1))
        self.apply(self.initialize)
    @staticmethod
    def initialize(layer):
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    def forward(self,images):
        if tuple(images.shape[1:])!=(1,256,256):
            raise ValueError('Expected N x 1 x 256 x 256')
        return self.regressor(self.features(images)).squeeze(1)

def cnn_model(pixels=256):
    if pixels!=256: raise ValueError('Model A requires 256 x 256')
    return ModelA()
