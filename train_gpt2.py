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
        