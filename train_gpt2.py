from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT2(nn.Module):

    def __init__(self, config):
        super.__init__()
        self.config = config
        # config is a dataclass object
        
        # The skeleton of the model
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),  # h stands for head
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # This gives a dictionary of the form: 
        # "wte": nn.Embedding, 
        # "wpe": nn.Embedding, 
        # "h": nn.ModuleList, 
        # "ln_f": nn.LayerNorm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)