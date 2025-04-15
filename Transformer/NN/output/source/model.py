import os
import argparse

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from torch.distributions import Beta

def load_data(data_dir, norm_params=None, use_numpy=False):
    
    snapshot_number = 38
    file_path = os.path.join(data_dir, f"subgroup.{snapshot_number}.txt")
    data = np.loadtxt(file_path)
    logm = data[:, 0]
    log_sfr = data[:, 7]
    r = data[:, 8]
    vr = data[:, 9]

    mask = r > 0 # train with satellites only
    logm = logm[mask]
    log_sfr = log_sfr[mask]
    r = r[mask]
    vr = vr[mask]

    r = np.log10(r)
    vr = np.sign(vr) * np.log10(np.abs(vr) + 1)

    count = 0
    source = []
    for s in [logm, log_sfr]:
        xmin, xmax = norm_params[count]
        s = ( s - xmin ) / ( xmax - xmin )
        s = np.clip(s, 0, 1)
        source.append(s)
        count += 1

    target = []
    for t in [r, vr]:
        ymin, ymax = norm_params[count]
        t = ( t - ymin ) / ( ymax - ymin )
        t = np.clip(t, 0, 1)
        target.append(t)
        count += 1
    
    source = np.array(source).T
    target = np.array(target).T

    if not use_numpy:
        source = torch.tensor(source, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

    return source, target

class MyDataset(Dataset):
    def __init__(self, data_dir, norm_params=None, ndata=None):
        self.x, self.y = load_data(data_dir, norm_params=norm_params)
        if ndata is not None:
            self.x = self.x[:ndata]
            self.y = self.y[:ndata]


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def my_NN_model(args):
    
    model = JointDensityEstimator2D(num_features_in=args.num_features_in, hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout)
    #model = NeuralNet(num_features_in=args.num_features_in, num_features_out=args.num_features_out, hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout)

    return model


class JointDensityEstimator2D(nn.Module):  # num_features_out should be 2
    def __init__(self, num_features_in=2, hidden_dim=64, output_dim=32, dropout=0):
        super().__init__()

        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(num_features_in, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * output_dim),
        )

        self.last_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch, 2)
        return: (batch, 2, output_dim)
        """
        out = self.net(x)  # (batch, output_dim * output_dim)
        out = self.last_activation(out)  # (batch, output_dim * output_dim)
        out = out.view(-1, self.output_dim, self.output_dim)  # (batch, output_dim, output_dim)
        
        return out
    
    def generate(self, x):
        """
        x: (batch, 2)
        return: (batch, 2, output_dim)
        """
        out = self.net(x)
        out = self.last_activation(out) # (batch, output_dim * output_dim)

        # Generate a sample from the distribution
        batch_size = out.size(0)
        bin_indices = torch.multinomial(out, num_samples=1).squeeze(-1) # (batch,)
        i, j = bin_indices // self.output_dim, bin_indices % self.output_dim # (batch, )
        di = torch.rand(batch_size, device=out.device) # (batch, )
        dj = torch.rand(batch_size, device=out.device)
        y1 = ( i.float() + di ) / self.output_dim
        y2 = ( j.float() + dj ) / self.output_dim

        return torch.stack([y1, y2], dim=1), out.view(-1, self.output_dim, self.output_dim)



class NeuralNet(nn.Module): 
    def __init__(self, num_features_in=2, num_features_out=2, hidden_dim=64, output_dim=32, dropout=0):
        super().__init__()
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(num_features_in, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features_out * output_dim),
        )

        self.last_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch, 2)
        return: (batch, 2, output_dim)
        """
        out = self.net(x)  # (batch, 2 * output_dim)
        out = out.view(-1, 2, self.output_dim)  # (batch, 2, output_dim)
        out = self.last_activation(out)  # (batch, 2, output_dim)

        return out