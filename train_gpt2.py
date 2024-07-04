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
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
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
        super().__init__()
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

        # lm_head and wte share the same weights
        self.lm_head.weight = self.transformer['wte'].weight
        # The shapes of lm_head and wte are different, but weights can be transposed, and they share the data

        # Initialize weights
        self.apply(self._init_weights)  # Applies the _init_weights function to all the modules in the model)

    def _init_weights(self, module):  # module is inside the model
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(model.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # no need to initialize LayerNorm, since pytorch initializes it with zeros and ones

    def forward(self, idx, targets=None):
        # idx is of shape (B, T) since it is a batch of tokens
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward length{T}, block size is exhausted"
        # Token embedding and positional embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos)  # (T, n_embd)
        tok_emb = self.transformer['wte'](idx)  # (B, T, n_embd)
        idx = tok_emb + pos_emb                   # (B, T, n_embd)
        # Transformer
        for block in self.transformer['h']:
            idx = block(idx)                        # (B, T, n_embd)
        # Final layer norm
        idx = self.transformer['ln_f'](idx)         # (B, T, n_embd)
        # Language model head
        logits = self.lm_head(idx)                  # (B, T, vocab_size)
        # Loss calculation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # This gives around 11, which is ok for random initialization given that -ln(1/50257) = 10.8
        return logits, loss

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
        config = GPT2Config(**config_args)
        model = GPT2(config)
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

# -----------------------------------------------------------------------------------
# Load Data
import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        # Not moving to GPU here since don't want to waste GPU memory
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = {buf[:-1].view(B, T)}  # inputs
        y = {buf[1: ].view(B, T)}  # targets
        # move the position
        self.current_position += B * T
        # reset the position if we reach the end
        if self.current_position + B * T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y


# Auto Detect GPU
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("device: " + device)

# Train
train_loader = DataLoaderLite(4, 32)
model = GPT2(GPT2Config())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW optimizer is a bug fix of Adam, it has a weight decay fix, which is a normalization of the gradient
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss {loss.item()}")

# Sampling
    # from transformers import pipeline, set_seed
    # set_seed(42)
    # generator = pipeline('text-generation', model="gpt2")
    # generator("Hello, I'm a Language Model,", max_length=30, num_return_sequences=5)
  # We need to implement the sampling function in the GPT2 class similar to the huggingface implementation
num_return_sequences = 5
max_length = 30

# model = GPT2.from_pretrained('gpt2')  # Load the model
model = GPT2(GPT2Config())
model.eval()  # Set the model to evaluation mode()
model.to('cuda')

  # prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a Language Model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, T)
# torch.Tensor.repeat() repeats the tensor along the specified dimensions
x = tokens.to("cuda")  # (5, T)

  # generate
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # take the logits of the last token
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
        # do topk sampling of 50, which is the huggingface default
        # only keep the top 50 tokens with the highest probability
        # anything after the top 50 will have its probability set to 0
        topk_probs, topk_idx = torch.topk(probs, 50, dim = -1)  # (B, 50)
        # select a token from the top 50
        ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_idx, -1, ix)  # (B, 1)
        # torch.gather() gathers values along an axis dim from the input tensor
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)