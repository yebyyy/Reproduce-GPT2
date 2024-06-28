from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# The name of the variables are kept same as in the huggingface implementation
# So that we can port the weights easily

class CaulsalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in one batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # B: batch size, T: sequence length, C: channels
        # calculate query, key, value
        qkv = self.c_attn(x)  # Emits 3 * n_embd
        q, k, v = qkv.split(self.n_embd, dim=2)
        # split into multiple heads, n_head act like a batch size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, C//n_head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, C//n_head
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, C//n_head
        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head) ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # attention as probabilities of each token
        y = att @ v  # (B, nh, T, T) * (B, nh T, hs)  # tokens as weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # contiguous() is used to make sure the tensor is stored in a contiguous chunk of memory
        # This is necessary if you want to use view() on the tensor and equivalent to concatenating the tensor
        # output projection
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")  # Gelu is a non-linear activation function
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CaulsalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Attention acts like reducing
        x = x + self.mlp(self.ln_2(x))  # MLP acts like mapping
        return x


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