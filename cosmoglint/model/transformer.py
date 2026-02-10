import random

import torch
import torch.nn as nn

from torch.distributions import Categorical
#from .xattn_transformer import MeshConditionedXAttnTransformer, MeshSequenceConditionedXAttnTransformer

import torch.nn.functional as F

def transformer_model(cfg, **kwargs):
    
    if "transformer" in cfg.model_name:
        if cfg.model_name == "transformer1":
            model_class = Transformer1
        elif cfg.model_name == "transformer2":
            model_class = Transformer2
        elif cfg.model_name == "mesh_conditioned_transformer":
            model_class = MeshConditionedXAttnTransformer
        elif cfg.model_name == "mesh_sequence_conditioned_transformer":
            model_class = MeshSequenceConditionedXAttnTransformer
        elif cfg.model_name == "transformer1_with_global_cond":
            model_class = TransformerWithGlobalCond
            transformer_cls = Transformer1
        elif cfg.model_name == "transformer2_with_global_cond":
            model_class = TransformerWithGlobalCond
            transformer_cls = Transformer2
        elif cfg.model_name == "transformer1_with_attn":
            model_class = Transformer1WithAttn
        elif cfg.model_name == "transformer2_with_attn":
            model_class = Transformer2WithAttn
        else:
            raise ValueError(f"Invalid model: {cfg.model_name}")

        if len(cfg.output_features) != cfg.num_features_in:
            raise ValueError(f"num_features ({cfg.num_features_in}) is not consistent with the list of output features ({cfg.output_features})")      
          
        common_args = dict(
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            num_features_cond=cfg.num_features_cond,
            num_features_out=cfg.num_features_out,
            output_features=cfg.output_features,
            **kwargs,
        )

        if cfg.use_flat_representation:
            common_args["max_length"] = cfg.max_length * cfg.num_features_in
            common_args["num_features_in"] = 1
            common_args["num_token_types"] = cfg.num_features_in
        
        else:
            common_args["max_length"] = cfg.max_length 
            common_args["num_features_in"] = cfg.num_features_in
            common_args["num_token_types"] = 1
        
        
        if "with_global_cond" in cfg.model_name:
            common_args["num_features_global"] = cfg.num_features_global
            common_args["transformer_cls"] = transformer_cls

        if "mesh" in cfg.model_name:
            common_args["cond_npix"] = cfg.npix_patch

        model = model_class(**common_args)

    else:
        raise ValueError(f"Invalid model: {cfg.model}")

    return model

class TransformerBase(nn.Module):
    def __init__(
            self, 
            d_model = 128, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
        ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.num_token_types = num_token_types

        if output_features is None:
            output_features = ["" for _ in range(num_features_in)]
        self.output_idx_map = {name: i for i, name in enumerate(output_features)}
        self.output_features = output_features
        
        # Pisitional embedding
        actual_max_length = max_length // num_token_types
        self.pos_embedding = nn.Embedding(actual_max_length, d_model)
        self.token_type_embedding = nn.Embedding(num_token_types, d_model)

        token_pos_id = torch.arange(actual_max_length).repeat_interleave(num_token_types)
        token_type_id = torch.arange(num_token_types).repeat(actual_max_length)
        self.register_buffer("token_pos_id", token_pos_id.long())
        self.register_buffer("token_type_id", token_type_id.long())

    def forward(self, condition, x, global_cond=None, cond_mask=None):
        raise NotImplementedError("forward method not implemented")
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
    def _set_to_zero(self, x, mask):
        zero_tensor = torch.tensor(0.0).to(x.device)
        return torch.where(mask, zero_tensor, x)
    
    def calc_loss(self, batch, weight=None):
        
        device = next(self.parameters()).device
        seq = batch["target"].to(device)     # (batch, max_length, num_features_in)
        mask = batch["mask"].to(device)   # (batch, max_length)
        condition = batch["condition"]

        if isinstance(condition, dict):
            condition = {k: v.to(device) for k, v in condition.items()}
        else:
            condition = condition.to(device)
        global_cond = batch["global_cond"].to(device) # (batch, num_features_global)
        
        input_seq = seq[:, :-1]
        target = seq

        output = self(condition, input_seq, global_cond=global_cond) # (batch, max_length, num_features_in, num_features_out)
        #_, output = model.generate(condition, seq=seq, teacher_forcing_ratio=teacher_forcing_ratio) 
        # output: (batch, max_length, num_features_in, num_features_out)

        if weight is None:
            weight = torch.ones_like(target, dtype=torch.float32, device=target.device) # (batch, seq_length)

        weight = mask * weight

        log_prob = torch.log( output + 1e-8 )
        target_bins = (target * self.num_features_out).long() # (batch, seq_length, num_features_in) [0, 1] -> [0, num_features_out-1]
        target_bins = torch.clamp(target_bins, min=0, max=self.num_features_out - 1)

        log_prob_flatten = log_prob.contiguous().view(-1, self.num_features_out) # (batch * seq_length * num_features_in, num_features_out)
        target_bins_flatten = target_bins.contiguous().view(-1) # (batch * seq_length * num_features_in, )
        weight_flatten = weight.contiguous().view(-1) # (batch * seq_length * num_features_in, )

        loss_nll = F.nll_loss(log_prob_flatten, target_bins_flatten, reduction='none') 
        loss = (loss_nll * weight_flatten).sum() / ( (weight_flatten).sum() + 1e-8 )

        return loss
        
    def generate(
            self, 
            condition, 
            global_cond = None, 
            seq = None, 
            cond_mask = None,
            teacher_forcing_ratio = 0.0, 
            temperature = 1.0, 
            stop_criterion = None, 
            prob_threshold = 1e-5, 
            monotonicity_start_index = 1, 
            max_ids = None, 
            buffer_percent = 0.05,
            first_values = {"SubhaloDist": 0.0, "SubhaloVrad": 0.5},
        ):
        """
        condition: a dict of conditions
        global_cond: (B, num_features_global)
        seq: (B, L, num_features_out) 
        teacher_forcing_ratio: 
        monotonicity_start_index: from which galaxy to enforce monotonicity. No enforcement if < 0.
        max_ids: torch tensor listing the indices of maximum value for the primary parameter for differen
        """

        if isinstance(condition, dict):
            cond_ref = next(iter(condition.values())) # One of the conditions 
        else:
            cond_ref = condition

        batch_size = len(cond_ref)
        device = cond_ref.device

        # used for enforce_monotonicity and max_ids
        # buffer = 1 indicates the bins just above the max_ids are avoided.
        buffer = max(int(buffer_percent * self.num_features_out), 1)
        if max_ids is not None:
            max_ids = max_ids.to(device) + buffer # (nbins, ) 
            nbins = len(max_ids)
            condition_bins = torch.linspace(0, 1, nbins, device=device) # (nbins, )
            condition_bin_indices = torch.bucketize(cond_ref[:, 0], condition_bins) - 1 # (batch, )
            bin_indices = torch.arange(self.num_features_out, device=device) # (num_features_out, )
            mask_max_ids = (bin_indices.unsqueeze(0) > max_ids[condition_bin_indices].unsqueeze(1)) # (batch, num_features_out)

        generated = torch.zeros(batch_size, self.max_length, self.num_features_in).to(device) # (batch, max_length, num_features_in)
        mask_all_batch = torch.ones(batch_size, dtype=torch.bool).to(device)
                    
        for t in range(self.max_length):

            if seq is not None and t < seq.size(1) and random.random() < teacher_forcing_ratio:
                next_token = seq[:, t]
            else:
                x = self(condition, generated[:, :t], global_cond=global_cond, cond_mask=cond_mask) 
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
                            bin_indices = torch.arange(self.num_features_out, device=device).view(1, -1) # (1, num_features_out)
                            mask = (bin_indices > previous_token_bin) # (batch, num_features_out)
                            mask = mask & (x_last[:, 0, :] >= prob_threshold) # (batch, num_features_out)
                            x_last[:, 0, :] = self._set_to_zero(x_last[:, 0, :], mask) # set the probability to zero for bins above the previous bin

                    if max_ids is not None:
                        # Even when monotonicity is not enforced, set the probability at x > x_max to zero for the primary parameter if max_ids is defined.
                        x_last[:, 0, :] = self._set_to_zero(x_last[:, 0, :], mask_max_ids)

                x_last = self._set_to_zero(x_last, x_last < prob_threshold) # set the probability to zero if less than prob_threshold

                x_last = x_last.reshape(-1, self.num_features_out) # (batch * num_features_in, num_features_out)
                
                bin_indices = Categorical(probs=x_last).sample().float().view(-1, self.num_features_in) # (batch, num_features_in)
                uniform_noise = torch.rand_like(bin_indices, device=device)  # (batch, num_features_in)
                next_token = (bin_indices + uniform_noise) / self.num_features_out  # (batch, num_features_in)

                if token_type == 0:
                    next_token[:, 0] = self._set_to_zero(next_token[:, 0], next_token[:,0] < 1./ self.num_features_out) # strictly set the sampled primary parameter to zero if it is less than 1/num_features_out 

                mask_all_batch = torch.ones(batch_size, dtype=torch.bool).to(device)
                
                # Set the first galaxy's parameters to fixed values
                is_first_gal = ( t // self.num_token_types == 0 )
                if is_first_gal:
                    if self.num_token_types == 1:
                        for feat, cval in first_values.items():
                            idx = self.output_idx_map.get(feat)
                            if idx is not None:
                                next_token[:, idx] = cval + self._set_to_zero(next_token[:, idx], mask_all_batch)
                    else:
                        feat = self.output_features[token_type]
                        cval = first_values.get(feat)
                        if cval is not None:
                            next_token[:, 0] = cval + self._set_to_zero(next_token[:, 0], mask_all_batch)

            # Stop generation if the primary parameter is below criterion
            if token_type == 0:
                if stop_criterion is not None:            
                    if torch.all(next_token[:,0] < stop_criterion):
                        return generated, x

            generated[:, t, :] = next_token # (batch, num_features_in)           
            
        if seq is not None and teacher_forcing_ratio > 0:
            x = self(condition, generated[:,:-1], cond_mask=cond_mask)

        return generated, x


class Transformer1(TransformerBase): # add logM at first in the sequence
    def __init__(
            self, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_cond = 1, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True    
        ):
        super().__init__(d_model=d_model, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features)

        self.embedding_layers = nn.Sequential(
            nn.Linear(num_features_in, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        ) 
        self.condition_embedding_layers = nn.Sequential(
            nn.Linear(num_features_cond, d_model),
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
        
    def forward(self, condition, x, global_cond=None, cond_mask=None):
        """
        condition: (batch, num_features_cond)
        x: (batch, seq_length, num_features_in)
        """
        
        if isinstance(condition, dict):
            condition = condition["halo"]
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(-1)
        
        batch_size, seq_length, num_features_in = x.shape  
        total_seq_length = 1 + seq_length # total length of (condition and x)

        # concatenate embeddings of condition and x 
        condition = condition.view(batch_size, 1, -1) # (batch, 1, num_features_cond)
        for layer in self.condition_embedding_layers:
            condition = layer(condition)  
        # condition: (batch, 1, d_model)

        for layer in self.embedding_layers:
            x = layer(x)
        x = torch.cat([condition, x], dim=1)  # (batch, seq_length + 1, d_model)
        
        ## add positional embedding
        pos_emb = self.pos_embedding(self.token_pos_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        type_emb = self.token_type_embedding(self.token_type_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        x = x + pos_emb + type_emb # (batch, seq_length + 1, d_model)

        ## decode
        causal_mask = self.generate_square_subsequent_mask(total_seq_length).to(x.device)
        dummy_memory = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        x = self.decoder(x, memory=dummy_memory, tgt_mask=causal_mask)  # (batch, seq_length + 1, d_model)
    
        ## output layer
        x = self.output_layer(x)  # (batch, seq_length + 1, num_features_in * num_features_out) or (batch, seq_length + 1, num_features_out)
        
        if self.pred_prob:
            x = x.view(batch_size, total_seq_length, self.num_features_in, -1) # (batch, seq_length + 1, num_features_in, num_features_out)

        x = self.out_activation(x)
        
        return x

class Transformer2(TransformerBase): # embed condition and x together, and then add positional embedding
    def __init__(
            self, 
            num_features_cond = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):

        super().__init__(d_model=d_model, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features)
        
        self.start_token = nn.Parameter(torch.zeros(1, 1, num_features_in))

        self.embedding_layers = nn.Sequential(
            nn.Linear(num_features_cond+num_features_in, d_model),
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
    
    def forward(self, condition, x, global_cond=None, cond_mask=None):
        """
        condition: (batch, num_features_cond)
        x: (batch, seq_length, num_features_in)
        """

        if isinstance(condition, dict):
            condition = condition["halo"]
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(-1)

        batch_size, seq_length, num_features_in = x.shape 
        total_seq_length = 1 + seq_length # total length of (start token and x)

        condition = condition.view(batch_size, 1, -1) # (batch, 1, num_features_cond)
        condition = condition.expand(batch_size, seq_length+1, -1) # (batch, seq_length+1, num_features_cond)

        ## Add start token
        start = self.start_token.expand(batch_size, 1, -1).to(x.device)      # (batch, 1, num_features_in)
        x = torch.cat([start, x], dim=1) # (batch, seq_length+1, num_features_in) 

        ## Concatenate condition and x
        x = torch.cat([condition, x], dim=2)  # (batch, seq_length+1, num_features_cond + num_features_in)

        ## embedding
        for layer in self.embedding_layers:
            x = layer(x)  
        # x: (batch, seq_length+1, d_model)

        ## Add positional embedding
        pos_emb = self.pos_embedding(self.token_pos_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        type_emb = self.token_type_embedding(self.token_type_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        x = x + pos_emb + type_emb # (batch, seq_length + 1, d_model)
        
        ## Decode
        causal_mask = self.generate_square_subsequent_mask(total_seq_length).to(x.device)
        dummy_memory = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        x = self.decoder(x, memory=dummy_memory, tgt_mask=causal_mask)  # (batch, seq_length+1, d_model)
    
        ## Output layer
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
            num_features_cond = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__()

        self.transformer = transformer_cls(
            num_features_cond = num_features_cond + num_features_global,
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
        
    def forward(self, condition, x, global_cond, cond_mask=None):
        if isinstance(condition, dict):
            condition = condition["halo"]
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(-1)
        ctx = torch.cat([condition, global_cond], dim=1)
        return self.transformer(ctx, x)
    
    def generate(self, condition, global_cond, **kwargs):
        if isinstance(condition, dict):
            condition = condition["halo"]

        if len(condition.shape) == 1:
            condition = condition.unsqueeze(-1)
        ctx = torch.cat([condition, global_cond], dim=1)
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
            num_features_cond = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__(num_features_cond=num_features_cond, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features, dropout=dropout, last_activation=last_activation, pred_prob=pred_prob)
        
        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

class Transformer2WithAttn(Transformer2):
    def __init__(
            self, 
            num_features_cond = 1, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__(num_features_cond=num_features_cond, d_model=d_model, num_layers=num_layers, num_heads=num_heads, max_length=max_length, num_features_in=num_features_in, num_features_out=num_features_out, num_token_types=num_token_types, output_features=output_features, dropout=dropout, last_activation=last_activation, pred_prob=pred_prob)

        decoder_layer = TransformerDecoderLayerWithAttn(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)









###### cross-attention #######
class XAttnTransformer(TransformerBase): 
    def __init__(
            self, 
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):

        super().__init__(
            d_model=d_model, 
            max_length=max_length, 
            num_features_in=num_features_in, 
            num_features_out=num_features_out, 
            num_token_types=num_token_types, 
            output_features=output_features
        )
        
        self.start_token = nn.Parameter(torch.zeros(1, 1, num_features_in))
                
        self.x_embed = nn.Sequential(
            nn.Linear(num_features_in, d_model),
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
    
    def forward(self, condition, x, global_cond=None, cond_mask=None):
        # condition: (batch, L, d_model)
        # x: (batch, seq_length, num_features_in)

        batch_size, seq_length, num_features_in = x.shape 
        total_seq_length = 1 + seq_length # total length of (start token and x)

        ## Add start token
        start = self.start_token.expand(batch_size, 1, self.num_features_in).to(x.device) # (batch, 1, num_features_in)
        x = torch.cat([start, x], dim=1) # (batch, seq_length+1, num_features_in) 

        ## Embedding
        x = self.x_embed(x) # (batch, seq_length+1, d_model)

        ## Add positional embedding
        pos_emb = self.pos_embedding(self.token_pos_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        type_emb = self.token_type_embedding(self.token_type_id[:total_seq_length]).unsqueeze(0) # (1, seq_length + 1, d_model)
        x = x + pos_emb + type_emb # (batch, seq_length + 1, d_model)

        ## Decode
        causal_mask = self.generate_square_subsequent_mask(total_seq_length).to(x.device)
        x = self.decoder(x, memory=condition, tgt_mask=causal_mask, memory_key_padding_mask=cond_mask)  # (batch, seq_length+1, d_model)
    
        ## Output layer
        x = self.output_layer(x)  # (batch, seq_length+1, num_features_in * num_features_out) or (batch, seq_length+1, num_features_out)

        if self.pred_prob:
            x = x.view(batch_size, seq_length+1, self.num_features_in, -1) # (batch, seq_length+1, num_features_in, num_features_out)
        
        x = self.out_activation(x)

        return x


class MeshEncoder(nn.Module):
    """
    3D ViT encoder.
    Input:  mesh (B, npix, npix, npix) or (B, C, npix, npix, npix)
    Output: tokens (B, L, d_model), where L = (npix/patch)^3
    """
    def __init__(
        self,
        npix: int = 16,
        patch_size: int = 2,
        in_channels: int = 1,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        if npix % patch_size != 0:
            raise ValueError(f"N={npix} must be divisible by patch_size={patch_size}.")

        self.npix = npix
        self.patch_size = patch_size
        self.d_model = d_model

        n = npix // patch_size
        self.L = n * n * n

        # 3D patch embedding: (B,C,npix,npix,npix) -> (B,d,n,n,n)
        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # learned positional embedding 
        self.pos_emb = nn.Parameter(torch.zeros(1, self.L, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.drop = nn.Dropout(dropout)

        # Transformer Encoder (ViT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,  
            norm_first=pre_norm 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,npix,npix,npix) or (B,C,npix,npix,npix)
        returns: (B,L,d_model)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,1,npix,npix,npix)
        if x.dim() != 5:
            raise ValueError(f"Expected x to have 4 or 5 dims, got {x.shape}")

        B, C, Nx, Ny, Nz = x.shape
        if (Nx, Ny, Nz) != (self.npix, self.npix, self.npix):
            raise ValueError(f"Expected (npix,npix,npix)=({self.npix},{self.npix},{self.npix}), got {(Nx,Ny,Nz)}")

        # patchify -> tokens
        feat = self.patch_embed(x)                 # (B, D, n, n, n)
        tokens = feat.flatten(2).transpose(1, 2)   # (B, L, D)

        # add positional embedding
        tokens = tokens + self.pos_emb
        tokens = self.drop(tokens)

        # 3-layer ViT encoder
        tokens = self.encoder(tokens)              # (B, L, D)
        tokens = self.norm(tokens)
        return tokens
    
class SequenceEncoder(nn.Module):
    def __init__(
        self,
        num_features_in: int = 1,
        max_length: int = 100,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.num_features_in = num_features_in

        self.x_embed = nn.Sequential(
            nn.Linear(num_features_in, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.start_token = nn.Parameter(torch.zeros(1, 1, num_features_in))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.ctx_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, mask):

        batch_size = len(x)

        start = self.start_token.expand(batch_size, 1, self.num_features_in).to(x.device) # (batch, 1, num_features_in)
        x = torch.cat([start, x], dim=1) # (batch, seq_length+1, num_features_in) 
        mask = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device), mask], dim=1)   

        x = self.x_embed(x)
        tokens = self.ctx_encoder(x, src_key_padding_mask=mask)
        return tokens, mask
    
class MeshConditionedXAttnTransformer(nn.Module):
    def __init__(
            self, 
            cond_npix = 16,
            num_features_cond = 1,
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__()

        self.xattn_transformer = XAttnTransformer(
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

        self.mesh_encoder = MeshEncoder(npix=cond_npix, d_model=d_model, in_channels=num_features_cond)

    def _encode(self, condition):
        if isinstance(mesh, dict):
            mesh = condition["mesh3d"]
        else:
            mesh = condition
        cond_emb = self.mesh_encoder(mesh) # (batch, cond_length, d_model)

        return cond_emb

    def forward(self, condition, x, global_cond=None):
        cond_emb = self._encode(condition)
        return self.xattn_transformer(cond_emb, x, global_cond=global_cond)

    def generate(self, condition, **kwargs):
        cond_emb = self._encode(condition)
        return self.xattn_transformer(cond_emb, **kwargs)

class MeshSequenceConditionedXAttnTransformer(nn.Module):
    def __init__(
            self, 
            cond_npix = 16,
            num_features_cond = 1,
            d_model = 128, 
            num_layers = 4, 
            num_heads = 8, 
            max_length = 10, 
            num_features_in = 1, 
            num_features_out = 1, 
            num_token_types = 1, 
            output_features = None, 
            dropout = 0, 
            last_activation = nn.Softmax(dim=-1), 
            pred_prob = True
        ):
        super().__init__()

        self.xattn_transformer = XAttnTransformer(
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

        self.mesh_encoder = MeshEncoder(npix=cond_npix, in_channels=num_features_cond, d_model=d_model)
        
        self.actual_max_length = max_length // num_token_types
        actual_num_features_in = max_length // self.actual_max_length
        self.seq_encoder = SequenceEncoder(num_features_in=actual_num_features_in, max_length=self.actual_max_length, d_model=d_model)

        self.b_encoder = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
        )

    def _encode(self, condition):
        mesh = condition["mesh3d"]
        x_ctx = condition["context"]
        mask_ctx = condition["mask_ctx"]
        boundary = condition["boundary"]

        B, L, M = x_ctx.shape 
        x_ctx = x_ctx.reshape(B, self.actual_max_length, -1) # (batch, ctx_length, num_params)
        mask_ctx = ~mask_ctx.reshape(B, self.actual_max_length, -1)[...,0] # (batch, ctx_length)
        # Note: tokens with mask=True will be masked (ignored) 
        
        ## Embed condition
        mesh_emb = self.mesh_encoder(mesh) # (batch, cond_length, d_model)
        ctx_emb, mask_ctx = self.seq_encoder(x_ctx, mask_ctx) # (batch, ctx_length, d_model)
        b_emb = self.b_encoder(boundary).unsqueeze(1) # (batch, 1, d_model)

        cond_emb = torch.cat([mesh_emb, ctx_emb, b_emb], dim=1) # (batch, cond_length + ctx_length + 1, d_model)

        B, L, _ = mesh_emb.shape
        mask_mesh = torch.zeros(B, L, dtype=torch.bool, device=mesh_emb.device)
        mask_b = torch.zeros(B, 1, dtype=torch.bool, device=mesh_emb.device)
        cond_mask = torch.cat([mask_mesh, mask_ctx, mask_b], dim=1)

        return cond_emb, cond_mask

    def forward(self, condition, x, global_cond=None):
        cond_emb, cond_mask = self._encode(condition)
        return self.xattn_transformer(cond_emb, x, global_cond=global_cond, cond_mask=cond_mask)

    def generate(self, condition, **kwargs):
        cond_emb, cond_mask = self._encode(condition)
        return self.xattn_transformer.generate(cond_emb, cond_mask=cond_mask, **kwargs)