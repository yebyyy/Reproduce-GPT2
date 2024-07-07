from dataclasses import dataclass
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import inspect


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
        self.c_proj.GPT_SCALE_INIT = 1  # GPT uses 1/sqrt(n_embd) for scaling
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
        # att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head) ** 0.5)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)  # attention as probabilities of each token
        # y = att @ v  # (B, nh, T, T) * (B, nh T, hs)  # tokens as weighted sum of values
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        self.c_proj.GPT_SCALE_INIT = 1
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
    n_embd: int = 768  # 128 * 6

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
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5  # for each block we have 2 residual layers 1 for attention and 1 for MLP
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with the parameters that requires grad
        param_dict = {pn : p for pn, p in self.named_parameters()}
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}  # pn is parameter name, p is parameter
        # weight decay all the 2D parameters
        # this means weight tensors in matmul and embeddings decay, biases and layernorms do not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print("using fused adam: %s" % use_fused)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **extra_args)
        return optimizer

# -----------------------------------------------------------------------------------
# Load Data
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, process_count, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.process_count = process_count
        assert split in {'train', 'val'}
        
         # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank  # for example process_rank = 0 we start from 0, process_rank = 1 we start from 8192, process_rank = 2 we start from 16384...

        # with open("input.txt", "r") as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # # Not moving to GPU here since don't want to waste GPU memory
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1: ]).view(B, T)  # targets
        # move the position
        self.current_position += B * T * self.process_count  # B*T as a batch, then have to move process_count batches
        # reset the position if we reach the end
        if self.current_position + (B * T * self.process_count + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        # if self.current_position + (B * T * self.process_count + 1) >= len(self.tokens):
        #     self.current_position = self.B * self.T * self.process_rank
        return x, y

# DDP and Device
# Use command torchrun --standalone --nproc_per_node=4 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group
import os
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Train
torch.set_float32_matmul_precision("high")

# Gradient Accumulation for simulating large batch size
total_batch_size = 524288  # 2^19
B = 4
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_acc_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total batch size: {total_batch_size}, gradient accumulation steps: {grad_acc_steps}")

train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, process_count=ddp_world_size, split='train')
model = GPT2(GPT2Config(vocab_size=50304))
model.to(device)
# model = torch.compile(model)  # does not work with python 3.12
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank], output_device=device)
    # DDP is a wrapper that wraps the model and distributes it across the GPUs
    # device_ids is a list of GPU ids that the model will be distributed to
    # output_device is the device where the output will be gathered
    # DDP will take care of the communication between the GPUs
raw_model = model.module if ddp else model

# Learning Rate Schedule
import math
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715  # 375M / 2^19 = 715 we warmup for the first 375M tokens and 2^19 tokens per batch
max_steps = 19073  # 10e9 / 2^19 = 19073  we have 10B tokens and 2^19 tokens per batch
def get_lr(it):
    # 1. Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps   # it + 1 so avoid starting from 0
    # 2. Constant min_lr
    if it > max_steps:
        return min_lr
    # 3. Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts from 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


import time
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)  # AdamW optimizer is a bug fix of Adam, it has a weight decay fix, which is a normalization of the 
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t1 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    # gradient accumulation to simulate large batch size
    for mini_step in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_acc_steps  # divide by steps to average the loss
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (mini_step == grad_acc_steps - 1)
        loss.backward()
    if ddp:
        torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
    # clip the global norm of the gradients to 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine learning rate
    lr = get_lr(step)
    # set the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    t = t2 - t1
    tksec = train_loader.B * train_loader.T * grad_acc_steps * ddp_world_size / t
    if master_process:
        print(f"step {step}, loss {loss_accum.item()}, norm {norm: .4f}, time {t:.2f}s, tokens/sec {tksec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

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