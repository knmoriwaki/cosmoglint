import os
import argparse

import random
import numpy as np

import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from torch.distributions import Beta

def load_halo_data(data_dir, max_length=10, norm_params=None, use_numpy=False, use_dist=False, ndata=None):
    if norm_params is None:
        xmin, xmax = 0., 1.
        ymin = np.zeros(10)
        ymax = np.ones(10)
    else:
        xmin = norm_params[0,0]
        xmax = norm_params[0,1]
        ymin = norm_params[1:,0]
        ymax = norm_params[1:,1]

    y_list = []

    def convert_to_log(val, val_min):
        log_val = np.full_like(val, val_min)
        mask = val > 10**val_min
        log_val[mask] = np.log10(val[mask])
        return log_val

    def convert_to_log_with_sign(val, val_min):
        return np.sign(val) * np.log10(np.abs(val + 1))
    
    def normalize(val, val_min, val_max):
        return (val - val_min) / (val_max - val_min)
    
    def load_values(f, key, min_val, max_val):
        try:
            data = f[key][:]
            if key == "HaloMass":
                data *= 1e10 / f.attrs["Hubble"]
            
            mask = data > 10 ** min_val
            data = convert_to_log(data, min_val)
            data = normalize(data, min_val, max_val)

            return data, mask
        except KeyError:
            print(f"Key '{key}' not found in the file.")
            return None, None

    snapshot_number = 38
    file_path = os.path.join(data_dir, f"TNG300-1_{snapshot_number}.h5")
    with h5py.File(file_path, "r") as f:
        mass, mask = load_values(f, "HaloMass", xmin, xmax)
        sfr, _ = load_values(f, "SubgroupSFR", ymin[0], ymax[0])
        if use_dist:
            dist, _ = load_values(f, "SubgroupDist", ymin[1], ymax[1])
        #vrad, _ = load_values(f, "Subgrouprad", ymin[2], ymax[2])

        num_subgroups = f["NumSubgroups"][:]
        offset = f["Offset"][:]
        
        for j in range(len(mass)):
            start = offset[j]
            end = start + num_subgroups[j]
            if use_dist:
                y_j = np.stack([sfr[start:end], dist[start:end]], axis=1) # (num_subgroups, 2)
            else:
                y_j = sfr[start:end, None] # (num_subgroups, 1)

            if not use_numpy:
                y_j = torch.tensor(y_j, dtype=torch.float32)

            y_j = y_j[:max_length] # truncate
            y_list.append(y_j)
            
        y_list = [ y_list[i] for i in range(len(y_list)) if mask[i] ]

    mass = mass[mask]

    if ndata is not None:
        mass = mass[:ndata]
        y_list = y_list[:ndata]

    return mass, y_list


def load_halo_data_old(data_dir, max_length=10, norm_params=None, use_numpy=False, use_dist=False, ndata=None):
    
    if norm_params is None:
        xmin, xmax = 0., 1.
        ymin, ymax = 0., 1.
        ymin2, ymax2 = 0., 1.
    else:
        xmin, xmax = norm_params[0]
        ymin, ymax = norm_params[1]
        ymin2, ymax2 = norm_params[2]

    ymins = [ymin, ymin2]
    ymaxs = [ymax, ymax2]

    x_list = []
    y_list = []

    count = 0
    for snapshot_number in [38]:
        file_path = os.path.join(data_dir, f"group.central-satellite.{snapshot_number}.txt")
        file_path2 = os.path.join(data_dir, f"group.central-satellite.rad.{snapshot_number}.txt")
        with open(file_path, 'r') as f, open(file_path2, "r") as f2:
            for line, line2 in zip(f, f2):
                line = line.strip()
                line2 = line2.strip()

                if not line or line.startswith("#"):
                    continue
                
                tokens = line.split()
                tokens2 = line2.split()

                ### halo mass
                try:
                    m_val = float(tokens[0])
                except ValueError:
                    continue  

                if m_val < xmin:
                    continue

                if np.abs( float(tokens2[0]) - m_val ) > 1e-4:
                    raise ValueError("Mass values are not consistent!")
                
                ### central luminosity
                try:
                    y_central = float(tokens[1])
                    y2_central = float(tokens2[1])
                    if y_central < ymin:
                        y_central = ymin
                except ValueError:
                    y_central = ymin
                    y2_central = ymin2

                ### satellite luminosities
                try:
                    y_values = [float(tok) for tok in tokens[2:]]
                    y2_values = [float(tok) for tok in tokens2[2:]]
                except ValueError:
                    y_values = []
                    y2_values = []

                ### sort by luminosity
                sorted_indices = sorted(range(len(y_values)), key=lambda k: y_values[k], reverse=True)
                if use_dist:
                    y_lists = [y_values, y2_values]
                    y_centrals = [y_central, y2_central]
                else:
                    y_lists = [y_values]
                    y_centrals = [y_central]

                processed_lists = []
                for y, y_c, ymin_, ymax_ in zip(y_lists, y_centrals, ymins, ymaxs):
                    y = [y_c] + [y[i] for i in sorted_indices] # sort
                    y = y[:max_length] # truncate
                    y = [np.log10(val) if val > 10**ymin_ else ymin_ for val in y] # log
                    y = (np.array(y) - ymin_) / (ymax_ - ymin_) # normalize

                    processed_lists.append(y)

                m_val = (m_val - xmin) / (xmax - xmin)
                x_list.append(m_val)

                y = np.array(processed_lists, dtype=np.float32) # (2, max_length)
                y = np.transpose(y, (1,0)) # (num_params, max_length) -> (max_length, num_params)
                if not use_numpy:
                    y = torch.tensor(y)
                y_list.append(y)

                ### stop if count >= ndata
                count += 1
                if ndata is not None and count >= ndata:
                    break

    x_list = np.array(x_list, dtype=np.float32)

    return x_list, y_list

class MyDataset(Dataset):
    def __init__(self, data_dir, max_length=10, norm_params=None, use_dist=False, ndata=None):
        
        self.x, self.y = load_halo_data(data_dir, max_length=max_length, norm_params=norm_params, use_dist=use_dist, ndata=ndata)

        _, num_params = (self.y[0]).shape

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.x = self.x.unsqueeze(1) # (N, 1)

        self.y_padded = torch.zeros(len(self.x), max_length, num_params) 
        self.mask = torch.zeros(len(self.x), max_length, num_params)

        for i, y_i in enumerate(self.y):
            length = len(y_i)
            self.y_padded[i, :length, :] = y_i
            self.mask[i, :length+1, :] = 1            

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_padded[idx], self.mask[idx]
    
def my_model(args):
    
    if "transformer" in args.model_name:
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
        
        model = model_class(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out)

    elif args.model_name == "neuralnet":
        model_class = NeuralNet(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, max_length=args.max_length, num_features_out=args.num_features_out)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    return model

class TransformerBase(nn.Module):
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out

    def forward(self, context, x):
        raise NotImplementedError("forward method not implemented")

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
    def generate(self, context, seq=None, teacher_forcing_ratio=0.0, temperature=1.0, stop_criterion=None):
        # context: (batch, num_condition)
        batch_size = len(context)

        generated = torch.zeros(batch_size, self.max_length, self.num_features_in).to(context.device) # (batch, max_length, num_features_in)

        for t in range(self.max_length):

            if seq is not None and t < seq.size(1) and random.random() < teacher_forcing_ratio:
                next_token = seq[:, t]
            else:
                x = self(context, generated[:, :t]) # generated[:, :t]: (batch, t, num_features_in)
                # (batch, t+1, num_faetures_in, num_features_out)

                x_last = x[:, -1, :, :] 
                # last taken (batch, num_features_in, num_features_out)

                if self.num_features_out == 1:
                    next_token = x
                elif self.num_features_out == 2:
                    # Beta distribution
                    alpha = x[:,0] + 1e-4
                    beta = x[:,1] + 1e-4
                    beta_dist = Beta(alpha / temperature, beta / temperature)
                    next_token = beta_dist.sample().unsqueeze(1)  # (batch, 1)
                else:
                    x_last = x_last / temperature
                    if seq is None: # Do not do this during training otherwise in-place error is raised.
                        #prob_threshold = x_last[:,0,1].view(-1, 1, 1)
                        prob_threshold = 1e-4
                        x_last[x_last < prob_threshold] = 0
                    next_token = torch.zeros(batch_size, self.num_features_in).to(context.device)

                    x_last = x_last.reshape(-1, self.num_features_out)  # (batch * num_features_in, num_features_out)
                    bin_indices = torch.multinomial(x_last, 1).squeeze(-1).float()  # (batch * num_features_in,)
                    bin_indices = bin_indices.view(-1, self.num_features_in)  # (batch, num_features_in)
                    uniform_noise = torch.rand_like(bin_indices, device=context.device)  # (batch, num_features_in)
                    next_token = (bin_indices + uniform_noise) / self.num_features_out  # (batch, num_features_in)
                
            generated[:, t, :] = next_token # (batch, num_features_in)           
            #generated = torch.cat([generated, next_token], dim=1)

            if stop_criterion is not None:
                if torch.all(next_token[:,0] < stop_criterion):
                    return generated, x

        if seq is not None and teacher_forcing_ratio > 0:
            x = self(context, generated[:,:-1])
        
        return generated, x

class Transformer1(TransformerBase): # add logM at first in the sequence
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout)

        self.embedding_layers = nn.Sequential(
            nn.Linear(num_features_in, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        ) 
        self.context_embedding_layers = nn.Sequential(
            nn.Linear(num_condition, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(max_length, d_model))

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, num_features_in*num_features_out) 

        if num_features_out == 1:
            self.out_activation = nn.Sigmoid()
        elif num_features_out == 2:
            self.out_activation = nn.Softplus()
        else:
            self.out_activation = nn.Softmax(dim=-1)

    def forward(self, context, x):
        # context: (batch, num_condition)
        # x: (batch, seq_length, num_features_in)
        
        batch_size, seq_length, num_features_in = x.shape  
        total_seq_length = seq_length + 1 # add start token (=context)

        # concatenate embeddings of context and x 
        context = context.view(batch_size, 1, -1) # (batch, 1, num_condition)
        for layer in self.context_embedding_layers:
            context = layer(context)  
        # context: (batch, 1, d_model)

        for layer in self.embedding_layers:
            x = layer(x)
        x = torch.cat([context, x], dim=1)  # (batch, seq_length + 1, d_model)
        
        # add position embedding
        x = x + self.pos_embedding[:total_seq_length, :].unsqueeze(0) # (batch, seq_length + 1, d_model)

        # decode
        causal_mask = self.generate_square_subsequent_mask(total_seq_length).to(x.device)
        dummy_memory = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        x = self.decoder(x, memory=dummy_memory, tgt_mask=causal_mask)  # (batch, seq_length + 1, d_model)
    
        # output layer
        x = self.output_layer(x)  # (batch, seq_length + 1, num_features_in * num_features_out)
        x = self.out_activation(x)

        x = x.view(batch_size, total_seq_length, self.num_features_in, -1) # (batch, seq_length + 1, num_features_in, num_features_out)

        return x


class Transformer2(TransformerBase): # embed context, position, and x together
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout)
        
        self.start_token = torch.ones(1) 
        
        self.position_ids = torch.linspace(0, 1, max_length) # (max_length+1, )
        
        self.embedding_layers = nn.Sequential(
            nn.Linear(num_condition+num_features_in+1, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, num_features_in*num_features_out)

        if num_features_out == 1:
            self.out_activation = nn.Sigmoid()
        elif num_features_out == 2:
            self.out_activation = nn.Softplus()
        else:
            self.out_activation = nn.Softmax(dim=-1)
    
    def forward(self, context, x):
        # context: (batch, num_condition)
        # x: (batch, seq_length, num_features_in)

        batch_size, seq_length, num_features_in = x.shape  

        context = context.view(batch_size, 1, -1) # (batch, 1, num_condition)
        context = context.expand(batch_size, seq_length+1, -1) # (batch, seq_length+1, num_condition)

        position = self.position_ids[:seq_length+1].to(x.device) # (seq_length+1, )
        position = position.expand(batch_size, -1).unsqueeze(-1) # (batch, seq_length+1, 1)

        ## add start token
        start_token = self.start_token.expand(batch_size, 1, self.num_features_in).to(x.device) # (batch, 1, num_features_in)
        x = torch.cat([start_token, x], dim=1) # (batch, seq_length+1, num_features_in) 

        ## concatenate context, position ids, and x
        x = torch.cat([context, position, x], dim=2)  # (batch, seq_length+1, num_condition + 1 + num_features_in)

        ## embedding
        for layer in self.embedding_layers:
            x = layer(x)  
        # x: (batch, seq_length+1, d_model)
        
        # decode
        causal_mask = self.generate_square_subsequent_mask(seq_length+1).to(x.device)
        dummy_memory = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        x = self.decoder(x, memory=dummy_memory, tgt_mask=causal_mask)  # (batch, seq_length+1, d_model)
    
        # output layer
        x = self.output_layer(x)  # (batch, seq_length+1, num_features_in * num_features_out)
        x = self.out_activation(x)

        x = x.view(batch_size, seq_length+1, self.num_features_in, -1) # (batch, seq_length+1, num_features_in, num_features_out)

        return x


class Transformer3(TransformerBase): # embed x and context together and then add positional embedding
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout)
        
        self.start_token = torch.ones(1) 
        
        self.pos_embedding = nn.Parameter(torch.randn(max_length, d_model))

        self.embedding_layers = nn.Sequential(
            nn.Linear(num_condition+num_features_in, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, num_features_in*num_features_out)

        if num_features_out == 1:
            self.out_activation = nn.Sigmoid()
        elif num_features_out == 2:
            self.out_activation = nn.Softplus()
        else:
            self.out_activation = nn.Softmax(dim=-1)
    
    def forward(self, context, x):
        # context: (batch, num_condition)
        # x: (batch, seq_length, num_features_in)

        batch_size, seq_length, num_features_in = x.shape  

        context = context.view(batch_size, 1, -1) # (batch, 1, num_condition)
        context = context.expand(batch_size, seq_length+1, -1) # (batch, seq_length+1, num_condition)

        ## add start token
        start_token = self.start_token.expand(batch_size, 1, self.num_features_in).to(x.device) 
        x = torch.cat([start_token, x], dim=1) # (batch, seq_length+1, num_features_in) 

        ## concatenate context, position ids, and x
        x = torch.cat([context, x], dim=-1)  # (batch, seq_length+1, num_condition+1)

        ## embedding
        for layer in self.embedding_layers:
            x = layer(x)
        # x: (batch, seq_length+1, d_model)

        x = x + self.pos_embedding[:seq_length+1, :].unsqueeze(0) # (batch, seq_length+1, d_model)
            
        # mask for padding
        causal_mask = self.generate_square_subsequent_mask(seq_length+1).to(x.device)

        # decoder        
        dummy_memory = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        x = self.decoder(x, memory=dummy_memory, tgt_mask=causal_mask)  # (batch, seq_length+1, d_model)
    
        # output layer
        x = self.output_layer(x)  # (batch, seq_length+1, num_features_out)
        x = self.out_activation(x)

        x = x.view(batch_size, seq_length+1, self.num_features_in, -1)

        return x

class NeuralNet(nn.Module): 
    def __init__(self, num_condition=1, d_model=128, num_layers=4, max_length=10, num_features_out=1):
        super().__init__()

        self.d_model = d_model

        self.max_length = max_length
        self.num_features_out = num_features_out

        self.start_token = torch.ones(1) 

        self.layers = nn.Sequential(
            nn.Linear(num_condition, d_model),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_features_out*max_length),
        )
        
        if num_features_out == 1:
            self.out_activation = nn.Sigmoid()
        elif num_features_out == 2:
            self.out_activation = nn.Softplus()
        else:
            self.out_activation = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x: (batch, num_condition)

        x = x.view(x.size(0), -1) # (batch, num_condition)

        batch_size, num_condition = x.shape
        for layer in self.layers:
            x = layer(x)

        if self.num_features_out > 1:
            x = x.view(batch_size, self.max_length, self.num_features_out)
        
        x = self.out_activation(x)
        return x

    def generate(self, x, temperature=1.0):

        x = self.forward(x)  

        if self.num_features_out > 2:
            batch_size, seq_length, num_features_out = x.shape
            output = torch.zeros(batch_size, seq_length)

            for i in range(batch_size):
                bin_indices = torch.multinomial(x[i], num_samples=1).squeeze(-1)  # (seq_length,)
                uniform_noise = torch.rand(seq_length, device=x.device)
                output[i] = ( bin_indices.float() + uniform_noise ) / self.num_features_out 
        else:
            output = x
            
        return output, x
    
from typing import Optional

class TransformerDecoderLayerWithAttn(nn.TransformerDecoderLayer):
    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
    
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )

        self.attn_weights = attn_weights.detach().cpu()
        return self.dropout1(attn_output)
    

class Transformer1WithAttn(Transformer1):
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout)
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

class Transformer2WithAttn(Transformer2):
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout)
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

class Transformer3WithAttn(Transformer3):
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout)
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

