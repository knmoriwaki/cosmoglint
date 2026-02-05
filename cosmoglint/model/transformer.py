import random

import torch
import torch.nn as nn

from torch.distributions import Categorical

def transformer_model(args, **kwargs):
    
    if "transformer" in args.model_name:
        if args.model_name == "transformer1":
            model_class = Transformer1
        elif args.model_name == "transformer2":
            model_class = Transformer2
        elif args.model_name == "transformer1_with_global_cond":
            model_class = TransformerWithGlobalCond
            transformer_cls = Transformer1
        elif args.model_name == "transformer2_with_global_cond":
            model_class = TransformerWithGlobalCond
            transformer_cls = Transformer2
        elif args.model_name == "transformer1_with_attn":
            model_class = Transformer1WithAttn
        elif args.model_name == "transformer2_with_attn":
            model_class = Transformer2WithAttn
        else:
            raise ValueError(f"Invalid model: {args.model_name}")

        if len(args.output_features) != args.num_features_in:
            raise ValueError(f"num_features ({args.num_features_in}) is not consistent with the list of output features ({args.output_features})")      
          
        common_args = dict(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_condition=args.num_features_cond,
            num_features_out=args.num_features_out,
            output_features=args.output_features,
            **kwargs,
        )

        if args.use_flat_representation:
            common_args["max_length"] = args.max_length * args.num_features_in
            common_args["num_features_in"] = 1
            common_args["num_token_types"] = args.num_features_in
        
        else:
            common_args["max_length"] = args.max_length 
            common_args["num_features_in"] = args.num_features_in
            common_args["num_token_types"] = 1
        
        
        if "with_global_cond" in args.model_name:
            common_args["num_features_global"] = args.num_features_global
            common_args["transformer_cls"] = transformer_cls
        
        model = model_class(**common_args)

    else:
        raise ValueError(f"Invalid model: {args.model}")

    return model

class TransformerBase(nn.Module):
    def __init__(
            self, 
            num_condition = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = ["SubhaloSFR"], 
            central_values = {"SubhaloDist": 0.0, "SubhaloVrad": 0.5}, 
            dropout = 0
        ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.num_token_types = num_token_types

        self.output_idx_map = {name: i for i, name in enumerate(output_features)}
        self.output_features = output_features
        self.central_values = central_values

        # Pisition and feature type embedding
        actual_max_length = max_length // num_token_types
        self.pos_embedding = nn.Embedding(actual_max_length, d_model)
        self.token_type_embedding = nn.Embedding(num_token_types, d_model)

        token_pos_id = torch.arange(actual_max_length).repeat_interleave(num_token_types)
        token_type_id = torch.arange(num_token_types).repeat(actual_max_length)
        self.register_buffer("token_pos_id", token_pos_id.long())
        self.register_buffer("token_type_id", token_type_id.long())

    def forward(self, context, x, global_cond=None):
        raise NotImplementedError("forward method not implemented")

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
    def _set_to_zero(self, x, mask):
        zero_tensor = torch.tensor(0.0).to(x.device)
        return torch.where(mask, zero_tensor, x)
        
    def generate(
            self, 
            context, 
            global_cond = None, 
            seq = None, 
            teacher_forcing_ratio = 0.0, 
            temperature = 1.0, 
            stop_criterion = None, 
            prob_threshold = 1e-5, 
            monotonicity_start_index = 1, 
            max_ids = None, 
            buffer_percent = 0.05
        ):
        """
        context: (B, num_features_in)
        global_cond: (B, num_features_global) -- simply ignored
        seq: (B, L, num_features_out) 
        teacher_forcing_ratio: 
        monotonicity_start_index: from which galaxy to enforce monotonicity. No enforcement if < 0.
        max_ids: torch tensor listing the indices of maximum value for the primary parameter for differen
        """
        # context: (batch, num_condition)
        batch_size = len(context)

        if len(context.shape) == 1:
            context = context.unsqueeze(-1)

        # used for enforce_monotonicity and max_ids
        # buffer = 1 indicates the bins just above the max_ids are avoided.
        buffer = max(int(buffer_percent * self.num_features_out), 1)
        if max_ids is not None:
            max_ids = max_ids.to(context.device) + buffer # (nbins, ) 
            nbins = len(max_ids)
            context_bins = torch.linspace(0, 1, nbins, device=context.device) # (nbins, )
            context_bin_indices = torch.bucketize(context[:, 0], context_bins) - 1 # (batch, )
            bin_indices = torch.arange(self.num_features_out, device=context.device) # (num_features_out, )
            mask_max_ids = (bin_indices.unsqueeze(0) > max_ids[context_bin_indices].unsqueeze(1)) # (batch, num_features_out)

        generated = torch.zeros(batch_size, self.max_length, self.num_features_in).to(context.device) # (batch, max_length, num_features_in)
        mask_all_batch = torch.ones(batch_size, dtype=torch.bool).to(context.device)
                    
        for t in range(self.max_length):

            if seq is not None and t < seq.size(1) and random.random() < teacher_forcing_ratio:
                next_token = seq[:, t]
            else:
                x = self(context, generated[:, :t], global_cond=global_cond) 
                # generated[:, :t]: (batch, t, num_features_in)
                # x: (batch, t+1, num_faetures_in, num_features_out)
                
                x_last = x[:, -1, :, :] 
                # last taken (batch, num_features_in, num_features_out)
            
                x_last = x_last / temperature

                token_type = t % self.num_token_types

                if token_type == 0:
                    if monotonicity_start_index is not None:
                        # Set the probability at x(t) >= x(t-1) to zero for the primary parameter
                        if t > monotonicity_start_index: 
                            previous_token_bin = (generated[:, t - self.num_token_types, 0] * self.num_features_out).long() + buffer
                            previous_token_bin = previous_token_bin.contiguous().view(-1, 1) # (batch, 1)
                            bin_indices = torch.arange(self.num_features_out, device=context.device).view(1, -1) # (1, num_features_out)
                            mask = (bin_indices > previous_token_bin) # (batch, num_features_out)
                            mask = mask & (x_last[:, 0, :] >= prob_threshold) # (batch, num_features_out)
                            x_last[:, 0, :] = self._set_to_zero(x_last[:, 0, :], mask) # set the probability to zero for bins above the previous bin

                    if max_ids is not None:
                        # Even when monotonicity is not enforced, set the probability at x > x_max to zero for the primary parameter if max_ids is defined.
                        x_last[:, 0, :] = self._set_to_zero(x_last[:, 0, :], mask_max_ids)

                x_last = self._set_to_zero(x_last, x_last < prob_threshold) # set the probability to zero if less than prob_threshold

                x_last = x_last.reshape(-1, self.num_features_out) # (batch * num_features_in, num_features_out)
                bin_indices = Categorical(probs=x_last).sample().float().view(-1, self.num_features_in) # (batch, num_features_in)
                uniform_noise = torch.rand_like(bin_indices, device=context.device)  # (batch, num_features_in)
                next_token = (bin_indices + uniform_noise) / self.num_features_out  # (batch, num_features_in)

                if token_type == 0:
                    next_token[:, 0] = self._set_to_zero(next_token[:, 0], next_token[:,0] < 1./ self.num_features_out) # strictly set the sampled primary parameter to zero if it is less than 1/num_features_out 

                mask_all_batch = torch.ones(batch_size, dtype=torch.bool).to(context.device)
                
                # Set the central galaxy's parameters to fixed values
                is_first_gal = ( t // self.num_token_types == 0 )
                if is_first_gal:
                    if self.num_token_types == 1:
                        for feat, cval in self.central_values.items():
                            idx = self.output_idx_map.get(feat)
                            if idx is not None:
                                next_token[:, idx] = cval + self._set_to_zero(next_token[:, idx], mask_all_batch)
                    else:
                        feat = self.output_features[token_type]
                        cval = self.central_values.get(feat)
                        if cval is not None:
                            next_token[:, 0] = cval + self._set_to_zero(next_token[:, 0], mask_all_batch)

            # Stop generation if the primary parameter is below criterion
            if token_type == 0:
                if stop_criterion is not None:            
                    if torch.all(next_token[:,0] < stop_criterion):
                        return generated, x

            generated[:, t, :] = next_token # (batch, num_features_in)           
            
        if seq is not None and teacher_forcing_ratio > 0:
            x = self(context, generated[:,:-1])

        return generated, x


class Transformer1(TransformerBase): # add logM at first in the sequence
    def __init__(
            self, 
            num_condition = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = ["SubhaloSFR"], 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True    
        ):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features, dropout=dropout)

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

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.pred_prob = pred_prob
        if pred_prob:
            self.output_layer = nn.Linear(d_model, num_features_in*num_features_out) 
        else:
            self.output_layer = nn.Linear(d_model, num_features_out)

        self.out_activation = last_activation
        

    def forward(self, context, x, global_cond=None):
        # context: (batch, num_condition)
        # x: (batch, seq_length, num_features_in)
        
        batch_size, seq_length, num_features_in = x.shape  
        total_seq_length = 1 + seq_length # total length of (context and x)

        # concatenate embeddings of context and x 
        context = context.view(batch_size, 1, -1) # (batch, 1, num_condition)
        for layer in self.context_embedding_layers:
            context = layer(context)  
        # context: (batch, 1, d_model)

        for layer in self.embedding_layers:
            x = layer(x)
        x = torch.cat([context, x], dim=1)  # (batch, seq_length + 1, d_model)
        
        # add position and type embedding
        pos_emb = self.pos_embedding(self.token_pos_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        type_emb = self.token_type_embedding(self.token_type_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        x = x + pos_emb + type_emb # (batch, seq_length + 1, d_model)

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

class Transformer2(TransformerBase): # embed context and x together, and then add positional embedding
    def __init__(
            self, 
            num_condition = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = ["SubhaloSFR"], 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):

        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features, dropout=dropout)
        
        self.start_token = torch.ones(1) 
                
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
    
    def forward(self, context, x, global_cond=None):
        # context: (batch, num_condition)
        # x: (batch, seq_length, num_features_in)

        batch_size, seq_length, num_features_in = x.shape 
        total_seq_length = 1 + seq_length # total length of (start token and x)

        context = context.view(batch_size, 1, -1) # (batch, 1, num_condition)
        context = context.expand(batch_size, seq_length+1, -1) # (batch, seq_length+1, num_condition)

        ## add start token
        start_token = self.start_token.expand(batch_size, 1, self.num_features_in).to(x.device) # (batch, 1, num_features_in)
        x = torch.cat([start_token, x], dim=1) # (batch, seq_length+1, num_features_in) 

        ## concatenate context and x
        x = torch.cat([context, x], dim=2)  # (batch, seq_length+1, num_condition + num_features_in)

        ## embedding
        for layer in self.embedding_layers:
            x = layer(x)  
        # x: (batch, seq_length+1, d_model)

        # add position and type embedding
        pos_emb = self.pos_embedding(self.token_pos_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        type_emb = self.token_type_embedding(self.token_type_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        x = x + pos_emb + type_emb # (batch, seq_length + 1, d_model)
        
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


class TransformerWithGlobalCond(nn.Module): 
    def __init__(
            self, 
            num_features_global, 
            transformer_cls = Transformer1,
            num_condition = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = ["SubhaloSFR"], 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__()

        self.transformer = transformer_cls(
            num_condition = num_condition + num_features_global,
            d_model = d_model, 
            num_layers = num_layers, 
            num_heads = num_heads, 
            max_length = max_length, 
            num_features_in = num_features_in, 
            num_features_out = num_features_out, 
            num_token_types = num_token_types, 
            output_features = output_features,
            dropout = dropout, 
            last_activation = last_activation, 
            pred_prob = pred_prob
        )
        
    def forward(self, context, x, global_cond):
        if len(context.shape) == 1:
            context = context.unsqueeze(-1)
        ctx = torch.cat([context, global_cond], dim=1)
        return self.transformer(ctx, x)
    
    def generate(self, context, global_cond, **kwargs):
        if len(context.shape) == 1:
            context = context.unsqueeze(-1)
        ctx = torch.cat([context, global_cond], dim=1)
        return self.transformer.generate(ctx, **kwargs)

    
from typing import Optional

class TransformerDecoderLayerWithAttn(nn.TransformerDecoderLayer):
    def _sa_block(
            self, 
            x: torch.Tensor, 
            attn_mask: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor], 
            is_causal: bool = False
        ) -> torch.Tensor:
    
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
    def __init__(
            self, 
            num_condition = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = ["SubhaloSFR"], 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features, dropout=dropout, last_activation=last_activation, pred_prob=pred_prob)
        
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

class Transformer2WithAttn(Transformer2):
    def __init__(
            self, 
            num_condition = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = ["SubhaloSFR"], 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__(num_condition=num_condition, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features, dropout=dropout, last_activation=last_activation, pred_prob=pred_prob)

        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


