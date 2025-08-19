
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Transformer1, Transformer2, Transformer3, Transformer1WithAttn, Transformer2WithAttn, Transformer3WithAttn

def transformer_nf_model(args, **kwargs):
    
    ### Transformer model ###
    if args.model_name == "transformer1":
        model_class = Transformer1
    elif args.model_name == "transformer2":
        model_class = Transformer2
    elif args.model_name == "transformer3":
        model_class = Transformer3
    elif args.model_name == "transformer1_with_attn":
        model_class = Transformer1WithAttn
    elif args.model_name == "transformer2_with_attn":
        model_class = Transformer2WithAttn
    elif args.model_name == "transformer3_with_attn":
        model_class = Transformer3WithAttn
    else:
        raise ValueError(f"Invalid model: {args.model_name}")
    
    model = model_class(num_condition=args.num_features_cond, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features, num_features_out=args.num_context, last_activation=nn.Tanh(), pred_prob=False, **kwargs)

    ### Flow model ###
    transforms = []
    for ilayer in range(args.num_flows):
        #transforms.append(ActNorm(args.num_features_out)) # for stabilyzing training and quick convergence
        transforms.append(
            AffineCouplingTransform(
                #mask=torch.arange(args.num_features_out) % 2,
                mask = torch.arange(args.num_features) % 2 if ilayer % 2 == 0 else (torch.arange(args.num_features) + 1) % 2,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.hidden_dim,
                    context_features=args.num_context,
                    num_blocks=2,
                    activation=nn.ReLU()
                )
            )
        )
        transforms.append(PiecewiseRationalQuadraticCDF(shape=[args.num_features], num_bins=8, tails='linear', tail_bound=3.0))
        transforms.append(RandomPermutation(args.num_features))
        
    transform = CompositeTransform(transforms)
    
    if args.base_dist == "normal":
        base_dist = StandardNormal(shape=[args.num_features])
    elif args.base_dist == "conditional_normal":
        base_dist = ConditionalDiagonalNormal(shape=[args.num_features], 
            context_encoder=nn.Sequential(nn.Linear(args.num_context, args.hidden_dim), 
                                          nn.LeakyReLU(), 
                                          nn.Linear(args.hidden_dim, 2 * args.num_features)
            )
        )

    elif args.base_dist == "bimodal":
        base_dist = BimodalNormal(shape=[args.num_features], offset=2.0)
    elif args.base_dist == "conditional_bimodal":
        base_dist = ConditionalBimodal(context_dim=args.num_context, latent_dim=args.num_features, hidden_dim=args.hidden_dim)
    else:
        raise ValueError(f"Invalid base distribution: {args.base_dist}")
    flow = Flow(transform, base_dist)

    return model, flow
    


def calculate_transformer_nf_loss(transformer, flow, condition, seq, mask, stop=None, stop_predictor=None):
    device = next(transformer.parameters()).device
    condition = condition.to(device)
    seq = seq.to(device)
    mask = mask.to(device) # (batch, max_length, num_context)

    input_seq = seq[:, :-1] # (batch, max_length-1, num_features)
    output = transformer(condition, input_seq) # (batch, max_length, num_context)

    mask = mask[:,:,0].reshape(-1) # (batch * max_length)

    _, _, num_features = seq.shape
    _, _, num_context = output.shape

    context_for_flow = output.reshape(-1, num_context)[mask]  # flow context (num_galaxies, num_context)
    target_for_flow = seq.reshape(-1, num_features)[mask] # flow target (num_galaxies, num_features)
    log_prob = flow.log_prob(target_for_flow, context=context_for_flow)
    loss = - log_prob.mean() / num_features

    if stop is not None and stop_predictor is not None:
        stop = stop.to(device) # (batch, max_length)
        stop = stop.reshape(-1)[mask] # (num_galaxies,)
        stop_pred = stop_predictor(context_for_flow)
        loss_stop = torch.nn.BCELoss()(stop_pred, stop)  # (num_galaxies,)
        return loss, loss_stop
    else:
        return loss

def generate_with_transformer_nf(transformer, flow, x_cond, stop_predictor=None, stop_threshold=None):
    transformer.eval()
    flow.eval()
    if stop_predictor is not None:
        stop_predictor.eval()

    batch_size = len(x_cond)
    max_length = transformer.max_length
    num_features = transformer.num_features_in

    x_seq = torch.zeros(batch_size, max_length, num_features).to(x_cond.device)
    stop_flags = torch.zeros(batch_size, dtype=torch.bool).to(x_cond.device)

    for t in range(max_length):
        with torch.no_grad():
            context = transformer(x_cond, x_seq[:,:t,:]) # (batch, t+1, num_context)
            context = context[:, -1, :] # (batch, num_context)
            sample = flow.sample(1, context) # (batch, 1, num_features)
            x_seq[:, t, :] = sample.squeeze(1)
               
            if stop_predictor is not None:  
                stop_prob = stop_predictor(context) # (batch, )
                stop_now = stop_prob > stop_threshold
                stop_flags = stop_now
            
            elif stop_threshold is not None:
                if t > 0:
                    stop_now = x_seq[:,t,0] < stop_threshold # (batch, )
                    stop_flags |= stop_now

            if t == 0:
                if num_features > 1:
                    x_seq[:,0,1] = torch.randn(batch_size).to(x_cond.device) * 1e-3 # Random distance for central
                if num_features > 2:
                    x_seq[:,0,2] = torch.randn(batch_size).to(x_cond.device) * 1e-3
            
            if stop_flags.all():
                break

    return x_seq

def my_stop_predictor(args):
    return StopPredictor(context_dim=args.num_context, hidden_dim=args.hidden_dim_stop)


class StopPredictor(nn.Module):
    def __init__(self, context_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context):
        x = self.layers(context)
        return x.squeeze(-1)  # (batch_size,) in [0, 1]



from nflows.flows import Flow
from nflows.distributions import StandardNormal, ConditionalDiagonalNormal
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

            samples1 = torch.randn(half, *self._shape, device=device) - self.offset
            samples2 = torch.randn(num_samples - half, *self._shape, device=device) + self.offset
            samples = torch.cat([samples1, samples2], dim=0)
            return samples[torch.randperm(num_samples)]  

        else:
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
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        x = inputs.view(inputs.size(0), -1)

        logp1 = -0.5 * torch.sum((x + self.offset)**2, dim=1) - self._log_z
        logp2 = -0.5 * torch.sum((x - self.offset)**2, dim=1) - self._log_z

        log_prob = torch.logaddexp(logp1, logp2) - np.log(2.0)
        return log_prob

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            return context.new_zeros(context.shape[0], *self._shape)

class ConditionalBimodal(nn.Module):
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

        scale = F.softplus(scale_raw) + 1e-3

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
        dist = self.get_distribution(context)  
        samples = dist.sample((num_samples,))  # (num_samples, batch, latent_dim)
        return samples.permute(1, 0, 2)  # (batch, num_samples, latent_dim)

    def log_prob(self, x, context):
        """
        x: (batch, latent_dim)
        context: (batch, context_dim)
        return: (batch,)
        """
        dist = self.get_distribution(context)
        return dist.log_prob(x)
    
