import time

import torch
import torch.nn as nn
from flash_attn import flash_attn_func  # need to install flash-attn

# Parameters
batch_size = 4
seq_len = 2048  # Long sequence to highlight the difference
num_heads = 8
head_dim = 64
embed_dim = num_heads * head_dim

# Random inputs
q = torch.randn(batch_size, seq_len, embed_dim).cuda()
k = torch.randn(batch_size, seq_len, embed_dim).cuda()
v = torch.randn(batch_size, seq_len, embed_dim).cuda()


# Standard attention (scaled dot-product attention)
def standard_attention(q, k, v):
    attn_weights = torch.softmax(q @ k.transpose(-2, -1) / head_dim**0.5, dim=-1)
    return attn_weights @ v


# FlashAttention wrapper
def flash_attention(q, k, v):
    # Reshape inputs to (batch, seq_len, num_heads, head_dim)
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, num_heads, head_dim)
    v = v.view(batch_size, seq_len, num_heads, head_dim)

    # FlashAttention expects (batch, seq, heads, dim)
    output = flash_attn_func(q, k, v)

    # Reshape output back to original dimension
    return output.view(batch_size, seq_len, embed_dim)


# Timing helper
def benchmark(func, q, k, v, label):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    output = func(q, k, v)

    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    print(f"{label} took {elapsed:.2f} ms")
    return output


if __name__ == "__main__":
    q, k, v = q.half(), k.half(), v.half()  # FlashAttention works best with fp16

    # Benchmark standard attention
    try:
        output_std = benchmark(standard_attention, q, k, v, "Standard Attention")
    except RuntimeError as e:
        print("Standard attention failed (likely due to memory limits):", e)

    # Benchmark FlashAttention
    try:
        output_flash = benchmark(flash_attention, q, k, v, "FlashAttention")
    except Exception as e:
        print("FlashAttention failed:", e)
