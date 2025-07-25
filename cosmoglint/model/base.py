import random

import torch
import torch.nn as nn

from torch.distributions import Categorical


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
    
    def _set_to_zero(self, x, mask):
        zero_tensor = torch.tensor(0.0).to(x.device)
        return torch.where(mask, zero_tensor, x)

    
    def generate(self, context, seq=None, teacher_forcing_ratio=0.0, temperature=1.0, stop_criterion=None, prob_threshold=0, cutoff=True, max_ids=None, buffer_percent=0.05):
        # context: (batch, num_condition)
        batch_size = len(context)

        buffer = max(int(buffer_percent * self.num_features_out), 1)
        # used for cutoff and max_ids
        # buffer = 1 indicates the bins just above the max_ids are avoided.
        if max_ids is not None:
            max_ids = max_ids.to(context.device) + buffer # (nbins, ) 

        if len(context.shape) == 1:
            context = context.unsqueeze(-1)

        generated = torch.zeros(batch_size, self.max_length, self.num_features_in).to(context.device) # (batch, max_length, num_features_in)
        mask_all_batch = torch.ones(batch_size, dtype=torch.bool).to(context.device)
                    
        for t in range(self.max_length):

            if seq is not None and t < seq.size(1) and random.random() < teacher_forcing_ratio:
                next_token = seq[:, t]
            else:
                x = self(context, generated[:, :t]) # generated[:, :t]: (batch, t, num_features_in)
                # (batch, t+1, num_faetures_in, num_features_out)
                
                x_last = x[:, -1, :, :] 
                # last taken (batch, num_features_in, num_features_out)
            
                x_last = x_last / temperature

                if t > 0:
                    # Set the probability of minimum bin to zero when sampling -- otherwise zero distance is sampled even for satellite galaxies with non-zero sfr 
                    for iparam in range(1, self.num_features_in):
                        mask = (x_last[:, iparam, 1:] >= prob_threshold).any(axis=1) # (batch,)
                        x_last[:, iparam, 0] = self._set_to_zero(x_last[:,iparam,0], mask) 

                if cutoff:
                    if t > 1: 
                        # Set the probability for SFR bins above the previous SFR bin to zero
                        # This is applied only from the second satellite galaxy
                        previous_token_bin = (generated[:, t-1, 0] * self.num_features_out).long() + buffer
                        previous_token_bin = previous_token_bin.contiguous().view(-1, 1) # (batch, 1)
                        bin_indices = torch.arange(self.num_features_out, device=context.device).view(1, -1) # (1, num_features_out)
                        mask = (bin_indices > previous_token_bin) # (batch, num_features_out)
                        mask = mask & (x_last[:, 0, :] >= prob_threshold) # (batch, num_features_out)
                        x_last[:, 0, :] = self._set_to_zero(x_last[:, 0, :], mask) # set the probability to zero for bins above the previous bin

                    elif max_ids is not None:
                        # Set the probability for SFR bins above the maximum SFR bin to zero
                        # This is applied only for the first satellite galaxy
                        # For the other galaxies, the previous token bin is used
                        nbins = len(max_ids)
                        context_bins = torch.linspace(0, 1, nbins, device=context.device) # (nbins, )
                        context_bin_indices = torch.bucketize(context[:, 0], context_bins) - 1 # (batch, )
                        bin_indices = torch.arange(self.num_features_out, device=context.device) # (num_features_out, )
                        mask = (bin_indices.unsqueeze(0) > max_ids[context_bin_indices].unsqueeze(1)) # (batch, num_features_out)
                        x_last[:, 0, :] = self._set_to_zero(x_last[:, 0, :], mask)

                x_last = self._set_to_zero(x_last, x_last < prob_threshold) # set the probability to zero if less than prob_threshold

                x_last = x_last.reshape(-1, self.num_features_out) # (batch * num_features_in, num_features_out)
                bin_indices = Categorical(probs=x_last).sample().float().view(-1, self.num_features_in) # (batch, num_features_in)
                uniform_noise = torch.rand_like(bin_indices, device=context.device)  # (batch, num_features_in)
                next_token = (bin_indices + uniform_noise) / self.num_features_out  # (batch, num_features_in)

                next_token[:, 0] = self._set_to_zero(next_token[:, 0], next_token[:,0] < 1./ self.num_features_out) # strictly set the sfr to zero if sfr is less than 1/num_features_out 

                mask_all_batch = torch.ones(batch_size, dtype=torch.bool).to(context.device)
                if t == 0:
                    if self.num_features_in > 1:
                        next_token[:, 1] = self._set_to_zero(next_token[:, 1], mask_all_batch) # Set the distance to zero for central
                    if self.num_features_in > 2:
                        next_token[:, 2] = 0.5 + self._set_to_zero(next_token[:, 2], mask_all_batch) # Set the radial velocity to zero (i.e., 0.5 after normalization) for central

            generated[:, t, :] = next_token # (batch, num_features_in)           
            
            if stop_criterion is not None:
                if torch.all(next_token[:,0] < stop_criterion):
                    return generated, x

        if seq is not None and teacher_forcing_ratio > 0:
            x = self(context, generated[:,:-1])

        return generated, x

class Transformer1(TransformerBase): # add logM at first in the sequence
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0, last_activation=nn.Softmax(dim=-1), pred_prob=True):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout)

        _d_model = d_model - 1

        self.embedding_layers = nn.Sequential(
            nn.Linear(num_features_in, _d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(_d_model, _d_model),
        ) 
        self.context_embedding_layers = nn.Sequential(
            nn.Linear(num_condition, _d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(_d_model, _d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, 1))

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.pred_prob = pred_prob
        if pred_prob:
            self.output_layer = nn.Linear(d_model, num_features_in*num_features_out) 
        else:
            self.output_layer = nn.Linear(d_model, num_features_out)

        self.out_activation = last_activation
        

    def forward(self, context, x):
        # context: (batch, num_condition)
        # x: (batch, seq_length, num_features_in)
        
        batch_size, seq_length, num_features_in = x.shape  
        total_seq_length = seq_length + 1 # add start token (=context)

        # concatenate embeddings of context and x 
        context = context.view(batch_size, 1, -1) # (batch, 1, num_condition)
        for layer in self.context_embedding_layers:
            context = layer(context)  
        # context: (batch, 1, d_model - 1)

        for layer in self.embedding_layers:
            x = layer(x)
        x = torch.cat([context, x], dim=1)  # (batch, seq_length + 1, d_model - 1)
        
        # add position embedding
        position = self.pos_embedding[:, :total_seq_length, :].expand(batch_size, -1, -1) # (batch, seq_length + 1, 1)
        x = torch.cat([x, position], dim=2) # (batch, seq_length + 1, d_model)

        # decode
        causal_mask = self.generate_square_subsequent_mask(total_seq_length).to(x.device)
        dummy_memory = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        x = self.decoder(x, memory=dummy_memory, tgt_mask=causal_mask)  # (batch, seq_length + 1, d_model)
    
        # output layer
        x = self.output_layer(x)  # (batch, seq_length + 1, num_features_in * num_features_out) or (batch, seq_length + 1, num_features_out)
        
        if self.pred_prob:
            x = x.view(batch_size, total_seq_length, self.num_features_in, -1) # (batch, seq_length + 1, num_features_in, num_features_out)

        x = self.out_activation(x)
        # x = entmax(x, dim=-1)

        return x

class Transformer2(TransformerBase): # embed context, position, and x together
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0, last_activation=nn.Softmax(dim=-1), pred_prob=True):
        
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

        self.pred_prob = pred_prob
        if pred_prob:
            self.output_layer = nn.Linear(d_model, num_features_in*num_features_out)
        else:
            self.output_layer = nn.Linear(d_model, num_features_out)

        self.out_activation = last_activation
    
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
        x = self.output_layer(x)  # (batch, seq_length+1, num_features_in * num_features_out) or (batch, seq_length+1, num_features_out)

        if self.pred_prob:
            x = x.view(batch_size, seq_length+1, self.num_features_in, -1) # (batch, seq_length+1, num_features_in, num_features_out)
        
        x = self.out_activation(x)

        return x


class Transformer3(TransformerBase): # embed x and context together and then add positional embedding
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0, last_activation=nn.Softmax(dim=-1), pred_prob=True):
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

        self.pred_prob = pred_prob
        if pred_prob:
            self.output_layer = nn.Linear(d_model, num_features_in*num_features_out)
        else:
            self.output_layer = nn.Linear(d_model, num_features_out)

        self.out_activation = last_activation
    
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
        x = self.output_layer(x)  # (batch, seq_length+1, num_features_in * num_features_out) or (batch, seq_length+1, num_features_out)

        if self.pred_prob:
            x = x.view(batch_size, seq_length+1, self.num_features_in, -1) # (batch, seq_length+1, num_features_in, num_features_out)

        x = self.out_activation(x)

        return x

    
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
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0, last_activation=nn.Softmax(dim=-1), pred_prob=True):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout, last_activation=last_activation, pred_prob=pred_prob)
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

class Transformer2WithAttn(Transformer2):
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0, last_activation=nn.Softmax(dim=-1), pred_prob=True):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout, last_activation=last_activation, pred_prob=pred_prob)
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

class Transformer3WithAttn(Transformer3):
    def __init__(self, num_condition=1, d_model=128, num_layers=4, num_heads=8, max_length=10, num_features_in=1, num_features_out=1, dropout=0, last_activation=nn.Softmax(dim=-1), pred_prob=True):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, dropout=dropout, last_activation=last_activation, pred_prob=pred_prob)
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

