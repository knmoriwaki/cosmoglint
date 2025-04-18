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
    v = data[:, 10]

    mask = r > 0 # train with satellites only
    logm = logm[mask]
    log_sfr = log_sfr[mask]
    r = r[mask]
    vr = vr[mask]
    v = v[mask]

    r = np.log10(r)
    vr = np.sign(vr) * np.log10(np.abs(vr) + 1)
    vt = np.log10(v)

    count = 0
    source = []
    for s in [logm, log_sfr]:
        if norm_params is not None:
            xmin, xmax = norm_params[count]
            s = ( s - xmin ) / ( xmax - xmin )
            s = np.clip(s, 0, 1)
        source.append(s)
        count += 1

    target = []
    for t in [r, vr, vt]:
        if norm_params is not None:
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
    
from nflows.flows import Flow
from nflows.distributions import StandardNormal, ConditionalDiagonalNormal, MADEMoG
from nflows.transforms import CompositeTransform, RandomPermutation, AffineCouplingTransform, ActNorm, PiecewiseRationalQuadraticCDF
from nflows.nn.nets import ResidualNet
from nflows.utils import torchutils

from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent


class BimodalNormal(StandardNormal):
    """A simple mixture of two Gaussians (bimodal) with fixed means."""

    def __init__(self, shape, offset=2.0):
        super().__init__(shape)
        self.offset = offset

    def _sample(self, num_samples, context=None):
        if context is None:
            device = self._log_z.device
            half = num_samples // 2

            # サンプルを2つの山に分ける
            samples1 = torch.randn(half, *self._shape, device=device) - self.offset
            samples2 = torch.randn(num_samples - half, *self._shape, device=device) + self.offset
            samples = torch.cat([samples1, samples2], dim=0)
            return samples[torch.randperm(num_samples)]  # ランダムに混ぜる

        else:
            # context に応じて num_samples × context_size 分のサンプルを出す
            context_size = context.shape[0]
            total = context_size * num_samples
            device = context.device
            half = total // 2

            samples1 = torch.randn(half, *self._shape, device=device) - self.offset
            samples2 = torch.randn(total - half, *self._shape, device=device) + self.offset
            samples = torch.cat([samples1, samples2], dim=0)
            samples = samples[torch.randperm(total)]

            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _log_prob(self, inputs, context=None):
        # 2つの正規分布の平均を取った log_prob（混合分布）
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        x = inputs.view(inputs.size(0), -1)

        # 計算を安定させるため log-sum-exp を使う
        logp1 = -0.5 * torch.sum((x + self.offset)**2, dim=1) - self._log_z
        logp2 = -0.5 * torch.sum((x - self.offset)**2, dim=1) - self._log_z

        log_prob = torch.logaddexp(logp1, logp2) - np.log(2.0)
        return log_prob

    def _mean(self, context):
        # 平均は0（左右対称のため）
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            return context.new_zeros(context.shape[0], *self._shape)

class ContextualBimodalBase(nn.Module):
    def __init__(self, context_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # context → offset (1,), logit(weight) (1,), scale (1,)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim + 2)  # mean1, mean2, scale, weight_logits
        )

    def get_distribution(self, context):
        batch_size = context.shape[0]
        params = self.net(context)  # (batch, 2*latent_dim + 2)

        # 分解
        mean1 = params[:, :self.latent_dim]
        mean2 = params[:, self.latent_dim:2*self.latent_dim]
        scale_raw = params[:, 2*self.latent_dim:2*self.latent_dim+1]  # shape (batch, 1)
        weight_logits = params[:, 2*self.latent_dim+1:]  # shape (batch, 1)

        # 保護処理
        scale = F.softplus(scale_raw) + 1e-3

        # 作成
        mix_logits = torch.cat([weight_logits, -weight_logits], dim=1)  # (batch, 2)
        mix_dist = Categorical(logits=mix_logits)

        means = torch.stack([mean1, mean2], dim=1)  # (batch, 2, latent_dim)
        scales = scale.unsqueeze(1).expand_as(means)  # (batch, 2, latent_dim)

        comp_dist = Independent(Normal(means, scales), 1)  # treat latent_dim as event dim
        mixture = MixtureSameFamily(mix_dist, comp_dist)
        return mixture

    def sample(self, num_samples, context):
        """
        context: (batch, context_dim)
        return: (batch, num_samples, latent_dim)
        """
        dist = self.get_distribution(context)  # batch個の mixture dist
        samples = dist.sample((num_samples,))  # (num_samples, batch, latent_dim)
        return samples.permute(1, 0, 2)  # → (batch, num_samples, latent_dim)

    def log_prob(self, x, context):
        """
        x: (batch, latent_dim)
        context: (batch, context_dim)
        return: (batch,)
        """
        dist = self.get_distribution(context)
        return dist.log_prob(x)


def my_flow_model(args):
    transforms = []
    for ilayer in range(args.num_layers):
        #transforms.append(ActNorm(args.num_features_out)) # for stabilyzing training and quick convergence
        transforms.append(
            AffineCouplingTransform(
                #mask=torch.arange(args.num_features_out) % 2,
                mask = torch.arange(args.num_features_out) % 2 if ilayer % 2 == 0 else (torch.arange(args.num_features_out) + 1) % 2,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.hidden_dim,
                    context_features=args.num_features_in,
                    num_blocks=2,
                    activation=nn.ReLU()
                )
            )
        )
        transforms.append(PiecewiseRationalQuadraticCDF(shape=[args.num_features_out], num_bins=8, tails='linear', tail_bound=3.0))
        transforms.append(RandomPermutation(args.num_features_out))
        
    transform = CompositeTransform(transforms)
    #base_dist = StandardNormal(shape=[args.num_features_out])
    #base_dist = ConditionalDiagonalNormal(shape=[args.num_features_out], context_encoder=nn.Linear(args.num_features_in, out_features=4))
    #base_dist = MADEMoG(features=args.num_features_out, hidden_features=16, context_features=args.num_features_in, num_mixture_components=2)
    #base_dist = BimodalNormal(shape=[args.num_features_out], offset=2.0)
    base_dist = ContextualBimodalBase(context_dim=args.num_features_in, latent_dim=args.num_features_out, hidden_dim=args.hidden_dim)
    flow = Flow(transform, base_dist)

    return flow
    