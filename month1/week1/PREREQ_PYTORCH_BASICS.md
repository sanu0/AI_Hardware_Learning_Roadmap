# PyTorch Basics — Read Before Week 1 Coding
## Everything You Need to Know (and Nothing More)

This document teaches you just enough PyTorch to run ALL code in Week 1.
It assumes you know basic Python (variables, loops, functions, lists).
Read it once, try every code snippet on Google Colab, and you're ready.

**Time needed:** 30-45 minutes
**Setup:** Open Google Colab (colab.research.google.com) → Runtime → Change runtime type → GPU

---

# 1. WHAT IS PYTORCH?

PyTorch is a Python library for doing math on GPUs. That's it.

```
NumPy:    Math on CPU (fast, but limited to CPU)
PyTorch:  Math on CPU OR GPU (can use Tensor Cores, CUDA, etc.)
```

Everything in PyTorch revolves around one thing: **Tensors**.

---

# 2. TENSORS — The Only Data Structure You Need

A tensor is just an array of numbers. That's all it is.
Think of it as a supercharged NumPy array that can live on a GPU.

## 2.1 Creating Tensors

```python
import torch

# --- SCALAR (just one number) ---
a = torch.tensor(5.0)
print(a)          # tensor(5.)
print(a.shape)    # torch.Size([])  ← no dimensions, just a number

# --- VECTOR (a list of numbers) ---
b = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(b)          # tensor([1., 2., 3., 4.])
print(b.shape)    # torch.Size([4])  ← 4 elements
print(len(b))     # 4

# --- MATRIX (a 2D grid of numbers) ---
c = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
print(c)
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
print(c.shape)    # torch.Size([2, 3])  ← 2 rows, 3 columns

# --- 3D TENSOR (a "stack" of matrices) ---
d = torch.tensor([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])
print(d.shape)    # torch.Size([2, 2, 2])  ← 2 matrices, each 2×2
```

**The `.shape` tells you the size of each dimension.** This is the MOST important
property you'll check constantly. In LLMs:
```
token_embeddings.shape = [batch_size, sequence_length, hidden_dim]
                         e.g., [32, 512, 4096]
                         = 32 sequences, each 512 tokens, each a 4096-dim vector
```

## 2.2 Common Ways to Create Tensors

```python
import torch

# Random numbers (normal distribution, mean=0, std=1)
x = torch.randn(3, 4)       # 3×4 matrix of random numbers
print(x)
# tensor([[ 0.2135, -1.0478,  0.3217,  0.8431],
#         [-0.5921,  0.1234,  1.5678, -0.2345],
#         [ 0.9876, -0.4567,  0.2345,  0.6789]])

# All zeros
z = torch.zeros(2, 3)       # 2×3 matrix of zeros
print(z)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# All ones
o = torch.ones(2, 3)        # 2×3 matrix of ones

# A range of numbers
r = torch.arange(0, 10)     # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
r = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]  (step of 2)

# Identity matrix (1s on diagonal, 0s elsewhere)
eye = torch.eye(3)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# Full of a specific value
f = torch.full((2, 3), 7.0)  # 2×3 matrix, all 7.0

# Same shape as another tensor
x = torch.randn(3, 4)
y = torch.zeros_like(x)      # same shape as x, filled with zeros
z = torch.randn_like(x)      # same shape as x, filled with random
```

## 2.3 Data Types (dtype)

Every tensor has a data type. This matters A LOT for GPU performance:

```python
# Default is float32 (FP32)
a = torch.randn(3, 3)
print(a.dtype)    # torch.float32  ← 4 bytes per number

# Explicitly set data type
b = torch.randn(3, 3, dtype=torch.float32)   # FP32: 4 bytes (default)
c = torch.randn(3, 3, dtype=torch.float16)   # FP16: 2 bytes (half precision)
d = torch.randn(3, 3, dtype=torch.bfloat16)  # BF16: 2 bytes (brain float)
e = torch.randint(0, 10, (3, 3), dtype=torch.int32)  # INT32: 4 bytes
f = torch.randint(0, 10, (3, 3), dtype=torch.int8)   # INT8: 1 byte

# Convert between types
a_fp32 = torch.randn(3, 3)                    # starts as FP32
a_fp16 = a_fp32.half()                         # convert to FP16
a_fp16 = a_fp32.to(torch.float16)              # same thing, different syntax
a_bf16 = a_fp32.bfloat16()                     # convert to BF16
a_back = a_fp16.float()                        # convert back to FP32

# Check size in memory
print(a_fp32.element_size())  # 4 (bytes per element)
print(a_fp16.element_size())  # 2
print(a_fp32.nelement())      # 9 (total elements in 3×3)
print(f"FP32 memory: {a_fp32.element_size() * a_fp32.nelement()} bytes")  # 36
print(f"FP16 memory: {a_fp16.element_size() * a_fp16.nelement()} bytes")  # 18
```

**When to use which:**
```
FP32:  Default. Used when you need full precision.
FP16:  Half memory, faster on Tensor Cores. Used for inference.
BF16:  Half memory, same range as FP32. Used for training.
INT8:  Quarter memory. Used for quantized inference.
```

---

# 3. TENSOR OPERATIONS — The Math

## 3.1 Basic Math

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations (same shape required)
c = a + b          # [5.0, 7.0, 9.0]     ← add each element
c = a - b          # [-3.0, -3.0, -3.0]
c = a * b          # [4.0, 10.0, 18.0]   ← multiply each element (NOT matrix multiply!)
c = a / b          # [0.25, 0.4, 0.5]

# Scalar operations (applies to every element)
c = a + 10         # [11.0, 12.0, 13.0]
c = a * 3          # [3.0, 6.0, 9.0]
c = a ** 2         # [1.0, 4.0, 9.0]     ← square each element

# Common math functions
c = torch.sqrt(a)       # [1.0, 1.414, 1.732]
c = torch.exp(a)        # [2.718, 7.389, 20.086]  ← e^x (used in softmax!)
c = torch.log(a)        # [0.0, 0.693, 1.099]     ← natural log
c = torch.abs(a)        # absolute value
c = torch.sin(a)        # sine (used in RoPE positional encoding!)
c = torch.cos(a)        # cosine

# Reductions (collapse a dimension)
c = a.sum()             # tensor(6.)        ← sum all elements
c = a.mean()            # tensor(2.)        ← average
c = a.max()             # tensor(3.)        ← maximum value
c = a.min()             # tensor(1.)
c = a.argmax()          # tensor(2)         ← INDEX of the maximum (used to pick the predicted token!)
```

## 3.2 Matrix Multiplication — THE Most Important Operation

```python
# The @ operator does matrix multiplication
A = torch.randn(2, 3)     # 2×3 matrix
B = torch.randn(3, 4)     # 3×4 matrix
C = A @ B                  # 2×4 matrix  (inner dimensions must match: 3 == 3)
print(C.shape)             # torch.Size([2, 4])

# Same thing with torch.matmul
C = torch.matmul(A, B)    # identical to A @ B

# This is EXACTLY what happens inside every Transformer layer:
# Q = input @ W_Q   where input is [batch, seq_len, 4096] and W_Q is [4096, 4096]
```

**Rule for matrix multiply shapes:**
```
[M × N] @ [N × K] = [M × K]
         ↑   ↑
    These must match!

Examples:
  [32 × 4096] @ [4096 × 4096] = [32 × 4096]     ✓ (LLM: Q/K/V projection)
  [32 × 4096] @ [4096 × 11008] = [32 × 11008]   ✓ (LLM: FFN up-projection)
  [32 × 4096] @ [11008 × 4096] = ERROR!          ✗ (4096 ≠ 11008)
```

## 3.3 Reshaping Tensors

```python
x = torch.randn(2, 3, 4)    # shape: [2, 3, 4]  (24 total elements)

# Reshape: change shape but keep same data (total elements must match)
y = x.reshape(6, 4)          # [2×3, 4] = [6, 4]
y = x.reshape(2, 12)         # [2, 3×4] = [2, 12]
y = x.reshape(24)            # flatten to 1D
y = x.reshape(-1)            # -1 means "figure it out" → same as 24

# View: same as reshape but shares memory (no copy)
y = x.view(6, 4)             # same as reshape, but more efficient

# Transpose: swap two dimensions
x = torch.randn(3, 4)        # [3, 4]
y = x.T                      # [4, 3]  (shortcut for 2D transpose)
y = x.transpose(0, 1)        # [4, 3]  (same thing, explicit)

# For 3D+ tensors:
x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)        # [2, 4, 3]  (swap dim 1 and dim 2)

# Permute: reorder any number of dimensions
y = x.permute(2, 0, 1)       # [4, 2, 3]  (move dim 2 to front)

# Squeeze/Unsqueeze: add or remove dimensions of size 1
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()               # [3, 4]    (remove all dims of size 1)
y = x.squeeze(0)              # [3, 1, 4] (remove only dim 0 if it's size 1)

x = torch.randn(3, 4)
y = x.unsqueeze(0)            # [1, 3, 4]  (add dim at position 0)
y = x.unsqueeze(1)            # [3, 1, 4]  (add dim at position 1)
```

**When you see shapes in LLM code:**
```
input:        [batch, seq_len, hidden]     = [32, 512, 4096]
unsqueeze:    [batch, 1, seq_len, hidden]  = adding a dimension for multi-head attention
transpose:    [batch, hidden, seq_len]     = needed for certain matrix multiplies
reshape:      [batch, seq_len, heads, head_dim] = splitting hidden into multiple heads
```

## 3.4 Indexing and Slicing

```python
x = torch.tensor([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Single element
print(x[0, 0])    # tensor(1)       ← row 0, col 0
print(x[1, 2])    # tensor(7)       ← row 1, col 2

# Entire row
print(x[0])        # tensor([1, 2, 3, 4])   ← all of row 0
print(x[0, :])     # same thing (: means "all")

# Entire column
print(x[:, 0])     # tensor([1, 5, 9])      ← all rows, col 0

# Slicing
print(x[0:2, :])   # rows 0 and 1 (2 is excluded)
print(x[:, 1:3])   # columns 1 and 2

# Last element
print(x[-1])        # tensor([9, 10, 11, 12])  ← last row
print(x[:, -1])     # tensor([4, 8, 12])        ← last column
```

---

# 4. GPU OPERATIONS — Moving Data to the GPU

This is the KEY difference between PyTorch and NumPy.

## 4.1 CPU vs GPU Tensors

```python
# By default, tensors live on CPU
a = torch.randn(3, 3)
print(a.device)    # device(type='cpu')

# Move to GPU
a_gpu = a.cuda()           # copy CPU → GPU
print(a_gpu.device)        # device(type='cuda', index=0)

# Alternative syntax
a_gpu = a.to('cuda')       # same thing
a_gpu = a.to('cuda:0')     # explicitly GPU 0 (if you have multiple)

# Create directly on GPU (faster — no copy needed)
b_gpu = torch.randn(3, 3, device='cuda')
print(b_gpu.device)        # device(type='cuda', index=0)

# Move back to CPU
b_cpu = b_gpu.cpu()        # copy GPU → CPU
b_cpu = b_gpu.to('cpu')    # same thing
```

## 4.2 The Golden Rule: Both Tensors Must Be on Same Device

```python
a_cpu = torch.randn(3, 3)              # on CPU
b_gpu = torch.randn(3, 3, device='cuda')  # on GPU

# This CRASHES:
# c = a_cpu + b_gpu    ← ERROR! Can't mix CPU and GPU tensors

# Fix: move both to same device
c = a_cpu.cuda() + b_gpu    # both on GPU now ✓
# OR
c = a_cpu + b_gpu.cpu()     # both on CPU now ✓ (but slower)
```

**In practice, for AI you put EVERYTHING on GPU and never move back to CPU
until the very end (when you need to print results or save to file).**

## 4.3 GPU Memory Management

```python
# Check how much GPU memory is being used
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

# Allocated = memory actively used by your tensors
# Reserved  = memory PyTorch has claimed from the GPU (includes freed but not returned)

# Free up memory
del a_gpu                       # delete the Python variable
torch.cuda.empty_cache()        # tell PyTorch to release unused reserved memory

# Check total GPU memory
total = torch.cuda.get_device_properties(0).total_memory
print(f"Total GPU Memory: {total / 1024**3:.1f} GB")
```

## 4.4 Why torch.cuda.synchronize() Matters

This is confusing at first but crucial for timing:

```python
import time

a = torch.randn(4096, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')

# WRONG way to time GPU operations:
start = time.time()
c = a @ b
elapsed = time.time() - start
print(f"Time: {elapsed*1000:.2f} ms")    # Shows ~0.01 ms — WRONG! Too fast!

# Why wrong? Because the GPU runs ASYNCHRONOUSLY:
#   Python says "hey GPU, multiply these" and IMMEDIATELY continues
#   The GPU hasn't finished yet when time.time() runs!

# CORRECT way:
torch.cuda.synchronize()    # wait for any previous GPU work to finish
start = time.time()
c = a @ b
torch.cuda.synchronize()    # wait for THIS operation to finish
elapsed = time.time() - start
print(f"Time: {elapsed*1000:.2f} ms")    # Shows ~2.5 ms — correct!
```

**Think of it like ordering food:**
```
WRONG timing:
  You: "I'll have a pizza"           ← start timer
  You: "How long did that take?" ← stop timer (0 seconds!)
  (Pizza isn't ready yet, you just ORDERED it)

CORRECT timing:
  You: "I'll have a pizza"           ← start timer
  You: *waits for pizza to arrive*   ← synchronize()
  You: "How long did that take?" ← stop timer (15 minutes)
```

---

# 5. GETTING GPU INFORMATION

```python
# Is GPU available?
print(torch.cuda.is_available())          # True or False

# How many GPUs?
print(torch.cuda.device_count())          # 1 (usually on Colab)

# GPU name
print(torch.cuda.get_device_name(0))      # "Tesla T4" (on Colab free)

# Detailed properties
props = torch.cuda.get_device_properties(0)
print(f"Name: {props.name}")
print(f"Compute Capability: {props.major}.{props.minor}")
print(f"Total Memory: {props.total_global_mem / 1024**3:.1f} GB")
print(f"SMs: {props.multi_processor_count}")

# Memory summary
print(torch.cuda.memory_summary())        # detailed memory breakdown
```

---

# 6. COMMON PATTERNS YOU'LL SEE IN WEEK 1 CODE

## Pattern 1: Create data → move to GPU → compute → measure time

```python
import torch
import time

N = 4096

# Create on CPU, move to GPU
A = torch.randn(N, N, dtype=torch.float16, device='cuda')
B = torch.randn(N, N, dtype=torch.float16, device='cuda')

# Warmup (first run is always slow due to GPU initialization)
for _ in range(3):
    _ = A @ B
torch.cuda.synchronize()

# Timed run
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    C = A @ B
torch.cuda.synchronize()
avg_time = (time.time() - start) / 10

print(f"Average time: {avg_time*1000:.2f} ms")
```

## Pattern 2: Compare different precisions

```python
A32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
A16 = A32.half()        # .half() = convert to FP16
A_bf = A32.bfloat16()   # .bfloat16() = convert to BF16
```

## Pattern 3: Watch memory as you allocate

```python
def show_memory(label):
    used = torch.cuda.memory_allocated() / 1024**2
    print(f"{label}: {used:.0f} MB used")

show_memory("Before")
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
show_memory("After creating 4096×4096 FP16 tensor")
# Expected: 4096 × 4096 × 2 bytes = 32 MB
```

## Pattern 4: Clean up memory

```python
del x
torch.cuda.empty_cache()
show_memory("After cleanup")
```

---

# 7. PYTORCH vs NUMPY — Quick Translation

If you know NumPy, here's the mapping:

```
NumPy                          PyTorch
─────────────────────────────────────────────────
import numpy as np              import torch
np.array([1,2,3])              torch.tensor([1,2,3])
np.zeros((3,4))                torch.zeros(3, 4)
np.ones((3,4))                 torch.ones(3, 4)
np.random.randn(3,4)           torch.randn(3, 4)
a.shape                        a.shape  (same!)
a.dtype                        a.dtype  (same!)
a.reshape(6, 2)                a.reshape(6, 2) or a.view(6, 2)
a.T                            a.T  (same!)
a @ b                          a @ b  (same!)
np.sum(a)                      torch.sum(a) or a.sum()
np.exp(a)                      torch.exp(a)
np.sqrt(a)                     torch.sqrt(a)
a[0, :]                        a[0, :]  (same!)

ONLY DIFFERENCE:
np.array → lives on CPU only
torch.tensor → can live on CPU or GPU (.cuda())
```

If you don't know NumPy either, don't worry — you now know PyTorch directly.

---

# 8. THINGS YOU DON'T NEED YET

These PyTorch features are important but NOT needed until later weeks:

```
WEEK 1 DAY 5 (you'll learn then):
  ❌ torch.nn.Module          (defining neural networks)
  ❌ torch.nn.Linear          (layers)
  ❌ torch.optim              (optimizers like Adam)
  ❌ requires_grad / backward (automatic differentiation)
  ❌ DataLoader / Dataset     (loading training data)

WEEK 2+:
  ❌ torch.nn.functional       (activation functions, loss)
  ❌ torch.save / torch.load   (saving models)
  ❌ torch.compile              (optimizing models)
  ❌ torch.cuda.amp            (mixed precision training)

You'll learn each of these WHEN you need them, with full explanation.
For now, just tensors + GPU operations + matrix multiply.
```

---

# 9. PRACTICE EXERCISE (10 minutes on Colab)

Try this yourself — type every line, don't copy-paste:

```python
import torch

# 1. Check if GPU is available
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

# 2. Create a matrix and move it to GPU
x = torch.randn(1000, 1000)
print("x is on:", x.device)

x_gpu = x.cuda()
print("x_gpu is on:", x_gpu.device)

# 3. Check memory
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
# Expected: 1000 × 1000 × 4 bytes = ~4 MB

# 4. Matrix multiply on GPU
y_gpu = torch.randn(1000, 1000, device='cuda')
z_gpu = x_gpu @ y_gpu
print(f"Result shape: {z_gpu.shape}")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

# 5. Compare FP32 vs FP16 size
big_fp32 = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
print(f"FP32 tensor: {big_fp32.element_size() * big_fp32.nelement() / 1024**2:.0f} MB")

big_fp16 = big_fp32.half()
print(f"FP16 tensor: {big_fp16.element_size() * big_fp16.nelement() / 1024**2:.0f} MB")
# FP16 is exactly half the memory!

# 6. Time a matrix multiply
import time
A = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
B = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(3):
    _ = A @ B
torch.cuda.synchronize()

# Measure
torch.cuda.synchronize()
start = time.time()
C = A @ B
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"\n4096×4096 FP16 matmul: {elapsed*1000:.2f} ms")
print("This is the EXACT operation inside every LLM layer!")

# 7. Clean up
del big_fp32, big_fp16, A, B, C
torch.cuda.empty_cache()
print(f"\nAfter cleanup: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print("\nDone! You know enough PyTorch for Week 1. 🎉")
```

---

*Time to complete: ~30-45 minutes*
*After this, you can confidently run all Day 2-5 code exercises.*
