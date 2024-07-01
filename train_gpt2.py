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
    block_size: int = 1024  # maximum length of the sequence
    vocab_size: int = 50257  # number of tokens = 50000 BPE merges(Each BPE merge will create a new token) + 256 bytes + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
