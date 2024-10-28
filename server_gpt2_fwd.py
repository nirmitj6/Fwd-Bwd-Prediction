## to execute allocate N GPUs from server and then execute ###
## torchrun --standalone --nproc_per_node=N server_gpt2.py
## So for 4 GPUs:
## torchrun --standalone --nproc_per_node=N server_gpt2.py
## To activate env1, first go inside the directory and execute
## source env1/bin/activate

from dataclasses import dataclass
import torch.backends
from transformers import GPT2LMHeadModel # type: ignore
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int= 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head==0
        self.c_attn=nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1

        self.n_head=config.n_head
        self.n_embd=config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B, T, C = x.size()
        qkv=self.c_attn(x)
        q,k,v= qkv.split(self.n_embd,dim=-1) # each (B,T,C)
        k=k.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        q=q.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        v=v.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        # att= (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))  # (B,nh,T,T)
        # att=att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # att=F.softmax(att, dim=-1)
        # y= att @ v # (B,nh,T,T)*(B,nh,T,hs) = (B,nh,T,hs)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc= nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu= nn.GELU(approximate='tanh')
        self.c_proj= nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
        wte=nn.Embedding(config.vocab_size, config.n_embd),
        wpe=nn.Embedding(config.block_size, config.n_embd),
        h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # we want the weight sharing scheme
        self.transformer.wte.weight=self.lm_head.weight

        # init parameters
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx, targets=None):
        B, T =idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size {B}"
        pos=torch.arange(0,T,dtype=torch.long, device=idx.device) # size T
        pos_emb=self.transformer.wpe(pos) # (T, n_embd)
        tok_emb=self.transformer.wte(idx) # (B,T, n_embd)
        x=tok_emb+pos_emb
        # forward the block of transformer
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
        
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # override_args = override_args or {} # default to empty dict
        # # only dropout can be overridden see more notes below
        # assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        #print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        #config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        # if 'dropout' in override_args:
        #     print(f"overriding dropout rate to {override_args['dropout']}")
        #     config_args['dropout'] = override_args['dropout']
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
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params= [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim()<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer


    
### loading data ####
import tiktoken
allowed_special={'<|endoftext|>'}
enc = tiktoken.get_encoding('gpt2')
class DataLoaderLite:
    def __init__(self,B,T, process_rank, num_processes, split):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes= num_processes
        self.split=split
        assert split in {'train', 'val'}
        

        if self.split=='train':
            with open('../Data/TinyStoriesTrain.txt','r') as f:
                text=f.read()
            if master_process:
                print('loading train data....')
        if self.split=="val":
            with open('../Data/TinyStoriesTest.txt','r') as f:
                    text=f.read()
            if master_process:
                print('loading validation data....')
                
        tokens=enc.encode(text, allowed_special=allowed_special)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch ={len(self.tokens) // (B*T)} batches")

        self.current_position=self.B * self.T * self.process_rank


    def next_batch(self):
        B,T=self.B, self.T
        split=self.split
        assert split in {'train', 'val'}
        if split=='train':
            if self.current_position + (B*T* self.num_processes+1) > len(self.tokens):
                self.current_position= self.B * self.T * self.process_rank
            buf =self.tokens[self.current_position : self.current_position+B*T+1]
            x= buf[:-1].view(B,T) # inputs
            y=buf[1:].view(B,T) # targets
            self.current_position += B*T*self.num_processes

        
        if split=='val':
            num_tokens=len(self.tokens)
            self.current_position=random.randint(0,num_tokens-(B*T+1))
            buf =self.tokens[self.current_position : self.current_position+B*T+1]
            x= buf[:-1].view(B,T) # inputs
            y=buf[1:].view(B,T) # targets
        return x,y


import time
### attempt to autodetect the device ###
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device="mps"
print(f"using device: {device}")

#### DDP ####
from torch.distributed import init_process_group, destroy_process_group
import os
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    # seed_offset = ddp_rank # each process gets a different seed
    # # world_size number of processes will be training simultaneously, so we can scale
    # # down the desired gradient accumulation iterations per process proportionally
    # assert gradient_accumulation_steps % ddp_world_size == 0
    # gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
# print(f"tokens per iteration will be: {tokens_per_iter:,}")




##### training part #####
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B=4
T=1024
assert total_batch_size % (B*T) == 0
grad_accum_steps =  total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"==> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')        
config=GPTConfig(vocab_size=50304)
model=GPT(config)
model.to(device)
#model=torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
test_loader=DataLoaderLite(B=4, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
train_loader=DataLoaderLite(B=4, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size,split='train')

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps=2000

def get_lr(it):
    if (it<warmup_steps):
        return max_lr * (it+1)/warmup_steps
    
    if (it>max_steps):
        return min_lr
    
    decay_ratio=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr + coeff * (max_lr-min_lr)

#optimizer=torch.optim.AdamW(model.parameters(), betas=(0.9,0.95),lr=3e-4, eps=1e-3)
store_train_loss=torch.zeros(max_steps)
store_val_loss=torch.zeros(max_steps)
optimizer=model.module.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0=time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x,y=x.to(device), y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss=model(x,y)
        #import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    optimizer.step()
    torch.cuda.synchronize()
    t1=time.time()
    dt=(t1-t0)*1000
    tokens_processed = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)
    tokens_per_sec=tokens_processed/(t1-t0)
    store_train_loss[step]=loss_accum
    x_test,y_test=test_loader.next_batch()
    x_test,y_test=x_test.to(device), y_test.to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss=model(x_test,y_test)
            store_val_loss[step]=loss
    if master_process:
        print(f"step {step} | Train loss: {store_train_loss[step].item():.6f} | Val loss: {store_val_loss[step].item():.6f} | learning rate: {lr:.6f}| norm: {norm:.4f} | dt: {dt:.2f} ms, tok/sec: {tokens_per_sec}")



if master_process:
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))#, gridspec_kw={'width_ratios': [1.5, 2]})
    ax.plot(np.arange(max_steps),store_train_loss, linestyle='-', alpha = 0.8, label=r'Training Loss')
    ax.plot(np.arange(max_steps),store_val_loss, linestyle='-', alpha = 0.8, label=r'Test Loss')
    ax.grid(alpha=0.4)
    ax.set_xlabel(r'iteration', fontsize =14)
    ax.set_ylabel(r'Cross Entropy Loss', fontsize =14)
    plt.title("Forward Model Evaluation")
    plt.setp(ax.get_xticklabels(),fontsize=12)
    plt.setp(ax.get_yticklabels(),fontsize=12)
    ax.tick_params(direction='out', length=5, width=1)
    ax.legend(loc='upper right', fontsize=12, framealpha = 1)
    fig.tight_layout()
    plt.savefig('Fwd_Loss.pdf')

    #import sys; sys.exit(0)
    num_return_sequences=5
    max_length=30
    model.eval()

    tokens=enc.encode("Hello, I am a language model,")
    tokens=torch.tensor(tokens,dtype=torch.long)
    tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)
    x=tokens
    x=tokens.to(device)
    torch.manual_seed(42)
    #torch.cuda.manual_seed(42)
    while x.size(1)< max_length:
        with torch.no_grad():
            logits, loss=model(x) #(B,T,vocab_size)
            print(logits.size())
            logits=logits[:,-1,:] # take logits at the last position (B,vocab_size)
            probs=F.softmax(logits, dim=-1)
            ## do top-k sampling
            topk_probs, topk_indices=torch.topk(probs,10,dim=-1)
            ix=torch.multinomial(topk_probs, 1) #(B,1)
            xcol=torch.gather(topk_indices,-1,ix) #(B,1)
            x= torch.cat([x,xcol],dim=1)

    for i in range(num_return_sequences):
        tokens=x[i,:max_length].tolist()
        decoded=enc.decode(tokens)
        print(">",decoded)

    print("didn't crash, yay!")

if ddp:
    destroy_process_group()
