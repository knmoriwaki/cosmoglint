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

def load_halo_data(data_dir, max_length=10, norm_params=None, use_numpy=False, use_dist=False, ndata=None):
    
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
    
    if args.model_name == "transformer1":
        model = Transformer1(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out)
    elif args.model_name == "transformer2":
        model = Transformer2(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out)
    elif args.model_name == "transformer3":
        model = Transformer3(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out)
    elif args.model_name == "transformer1_with_attn":
        model = Transformer1WithAttn(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out)
    elif args.model_name == "transformer2_with_attn":
        model = Transformer2WithAttn(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out)
    elif args.model_name == "transformer3_with_attn":
        model = Transformer3WithAttn(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, max_length=args.max_length, num_features_in=args.num_features_in, num_features_out=args.num_features_out)
    elif args.model_name == "neuralnet":
        model = NeuralNet(num_condition=1, d_model=args.d_model, num_layers=args.num_layers, max_length=args.max_length, num_features_out=args.num_features_out)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    return model

class Transformer1(nn.Module): # add logM at first in the sequence
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        
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

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
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

    def generate(self, context, seq=None, teacher_forcing_ratio=0.0, temperature=1.0):
        # context: (batch, num_condition)
        batch_size = len(context)

        generated = torch.zeros(batch_size, 0, self.num_features_in).to(context.device) # (batch, 0, num_features_in)

        for t in range(self.max_length):
            # generated: (batch, t, num_features_in)

            if seq is not None and t < seq.size(1) and random.random() < teacher_forcing_ratio:
                next_token = seq[:, t].unsqueeze(1)
            else:
                x = self(context, generated)  
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
                    prob_threshold = x_last[:,0,1].view(-1, 1, 1)
                    #x_last[x_last < prob_threshold] = 0
                    next_token = torch.zeros(batch_size, 1, self.num_features_in).to(context.device)
                    for iparam in range(self.num_features_in):
                        bin_indices = torch.multinomial(x_last[:,iparam,:], num_samples=1).squeeze(-1)  # (batch,)
                        uniform_noise = torch.rand(batch_size, device=context.device) # (batch, )
                        next_token[:,0,iparam] = ( bin_indices.float() + uniform_noise ) / self.num_features_out
                        
            generated = torch.cat([generated, next_token], dim=1)

        if seq is not None and teacher_forcing_ratio > 0:
            x = self(context, generated[:,:-1])
        
        return generated, x

class Transformer2(nn.Module): # embed context, position, and x together
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__()

        self.d_model = d_model
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out

        self.start_token = torch.ones(1) 
        
        self.max_length = max_length
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

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
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

    def generate(self, context, seq=None, teacher_forcing_ratio=0.0, temperature=1.0, prob_min=1e-3):
        # context: (batch, num_condition)

        batch_size = len(context)
        generated = torch.zeros(batch_size, 0, self.num_features_in).to(context.device) # (batch, 0, num_features_in)

        for t in range(self.max_length):
            # generated: (batch, t, num_features_in)
        
            if seq is not None and t < seq.size(1) and random.random() < teacher_forcing_ratio:
                next_token = seq[:, t].unsqueeze(1)
            else:
                x = self(context, generated) 
                # (batch, t+1, num_features_in, num_features_out)
                
                x_last = x[:, -1, :, :] 
                # last token (batch, num_features_in, num_features_out)

                if self.num_features_out == 1:
                    next_token = x_last
                elif self.num_features_out == 2:
                    # Beta distribution
                    alpha = x_last[:,0] + 1e-4
                    beta = x_last[:,1] + 1e-4
                    beta_dist = Beta(alpha / temperature, beta / temperature)
                    next_token = beta_dist.sample().unsqueeze(1)  # (batch, 1)
                else:
                    x_last = x_last / temperature
                    prob_threshold = x_last[:,0,1].view(-1, 1, 1)
                    #x_last[x_last < prob_threshold] = 0
                    next_token = torch.zeros(batch_size, 1, self.num_features_in).to(context.device) 
                    for iparam in range(self.num_features_in):
                        bin_indices = torch.multinomial(x_last[:,iparam,:], num_samples=1).squeeze(1)  # (batch,) 
                        uniform_noise = torch.rand(batch_size, device=context.device) # (batch,)
                        next_token[:,0,iparam] = ( bin_indices.float() + uniform_noise ) / self.num_features_out 
                    
            generated = torch.cat([generated, next_token], dim=1)


        if seq is not None and teacher_forcing_ratio > 0:
            x = self(context, generated[:,:-1])

        #x[x < x[:,:,0,1].view(batch_size, -1, 1,1)] = 0

        return generated, x

class Transformer3(nn.Module): # embed x and context together and then add positional embedding
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0):
        super().__init__()

        self.d_model = d_model
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out

        self.start_token = torch.ones(1) 
        
        self.max_length = max_length
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

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
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

    def generate(self, context, seq=None, teacher_forcing_ratio=0.0, temperature=1.0, prob_min=1e-4):
        # use seq as input if teacher_forcing_ratio > 0

        batch_size = len(context)

        generated = torch.zeros(batch_size, 0, self.num_features_in).to(context.device) # (batch, 0)

        for t in range(self.max_length):
            # generated: (batch, t, num_features_in)

            if seq is not None and t < seq.size(1) and random.random() < teacher_forcing_ratio:
                next_token = seq[:, t].unsqueeze(1)
            else:
                x = self(context, generated)
                # (batch, t+1, num_features_in, num_features_out)

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
                    #prob_threshold = x_last[:,0,1].view(-1, 1, 1) 
                    #x_last[x_last < prob_threshold] = 0
                    next_token = torch.zeros(batch_size, 1, self.num_features_in).to(context.device)
                    for iparam in range(self.num_features_in):
                        bin_indices = torch.multinomial(x_last[:,iparam,:], num_samples=1).squeeze(1)  # (batch,) 
                        uniform_noise = torch.rand(batch_size, device=context.device) # (batch,)
                        next_token[:,0,iparam] = ( bin_indices.float() + uniform_noise ) / self.num_features_out 
                    
            generated = torch.cat([generated, next_token], dim=1)

        if seq is not None and teacher_forcing_ratio > 0:
            x = self(context, generated[:,:-1])

        #x[x < prob_min] = 0 # This causes in_place operation error

        return generated, x

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

