import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class EncoderTemplate(nn.Module):
    def __init__(self, input_dim, hidden_layers, z_dim):
        super(EncoderTemplate, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_layers * 2),
                                    nn.BatchNorm1d(hidden_layers * 2),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_layers * 2, hidden_layers),
                                    nn.BatchNorm1d(hidden_layers),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(hidden_layers, z_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.out(out)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 35),
                                    #nn.LayerNorm(35),
                                    nn.ReLU(),
                                    nn.Linear(35, z_dim))

    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        #out = self.out(out)
        return out


class GeneratorTemplate(nn.Module):
    def __init__(self, z_dim, hidden_layers, input_dim):
        super(GeneratorTemplate, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(z_dim, hidden_layers),
                                    nn.BatchNorm1d(hidden_layers),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_layers, hidden_layers * 2),
                                    nn.BatchNorm1d(hidden_layers * 2),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(hidden_layers * 2, input_dim),
                                 nn.BatchNorm1d(input_dim),
                                 nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.out(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim, input_dim):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(z_dim, 35),#+ n_classes
                                    nn.ReLU(),
                                    nn.Linear(35, input_dim))

    def forward(self, x, y=None):
        #if y is not none:
        #    x = torch.cat((x, y), dim=1)
        return self.layer1(x) 


# spectral normalization
class DiscriminatorTemplate(nn.Module):
    def __init__(self, input_dim, hidden_layers, out_dim):
        super(DiscriminatorTemplate, self).__init__()
        self.layer1 = nn.Sequential(spectral_norm(nn.Linear(input_dim, hidden_layers * 2)),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(spectral_norm(nn.Linear(hidden_layers * 2, hidden_layers)),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(hidden_layers, out_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.out(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, out_dim, out_dim2=0):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(spectral_norm(nn.Linear(input_dim, 25 * 2)),
                                    nn.ReLU(),
                                    spectral_norm(nn.Linear(25 * 2, 25)),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(25, out_dim), nn.Sigmoid())

#        if n_classes > 0:
        self.classify = nn.Sequential(nn.Linear(25, out_features=out_dim2),
                                            nn.Softmax(dim=1))

    def forward(self, x):
        h = self.layer1(x)
        classify = self.classify(h)
        out = self.out(h)
        return out, classify
