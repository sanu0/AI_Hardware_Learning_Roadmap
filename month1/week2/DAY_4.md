# Week 2, Day 4: Memory Coalescing in Practice — AoS vs SoA Data Layouts
## How You Lay Out Your Data Decides Whether 32 Threads Read 1 Cache Line or 32

**Time: ~2.5-3 hours**
**Setup: Google Colab with GPU runtime**

---

## Today's Mental Model

Day 1 taught you the **rule** of coalescing: when 32 threads in a warp access 32 contiguous bytes, the GPU does 1 memory transaction; when they access scattered bytes, it does 32. Day 2 taught you that shared memory has 32 banks, and bad access patterns cause conflicts.

Today is the **design question**:

> When YOU choose how to lay out your data in memory, how do you make sure the warps that read it are coalesced?

By the end of today, you should be able to look at any data structure and ask:

1. **Will my warp read 32 contiguous bytes from this layout, or 32 scattered bytes?**
2. **Should I use AoS (Array of Structures), SoA (Structure of Arrays), or AoSoA (a hybrid)?**
3. **In a Transformer's KV-cache `[batch, num_heads, seq, head_dim]` — which dimension should be contiguous in memory?**
4. **Why does PyTorch's `.contiguous()` exist, and when do I have to call it?**
5. **When (rarely) is AoS actually faster than SoA?**

This is the day where "memory coalescing" stops being a rule you memorized and becomes **a design decision you make**.

---

# PART 0: QUICK RECALL FROM DAYS 1–2

Before diving in, make sure you remember (30 seconds):

```
✓ A warp = 32 threads executing the same instruction
✓ Coalesced read = 32 threads access 32 contiguous bytes → 1 transaction
✓ Uncoalesced read = 32 threads access scattered bytes → up to 32 transactions
✓ Cache line = 32 or 128 bytes (the fetch unit)
✓ Shared memory = 32 banks of 4 bytes each (one bank per warp lane)
✓ A bank conflict = 2+ threads hit the same bank → serialization
```

Today we apply these to the question of *where the bytes sit in memory in the first place*.

---

# PART 1: THE STORY

## 1.1 The Athletes' Locker Room

Imagine 32 athletes who all need their helmets at the same time.

**Design A — Per-athlete locker (AoS):**

```
Locker 0: [shoes][jersey][gloves][HELMET][socks][water]
Locker 1: [shoes][jersey][gloves][HELMET][socks][water]
Locker 2: [shoes][jersey][gloves][HELMET][socks][water]
...
Locker 31:[shoes][jersey][gloves][HELMET][socks][water]
```

Each athlete walks to their own locker, opens it, digs past everything else to find the helmet, takes it, closes the locker. 32 lockers opened. 32 separate trips.

**Design B — One rack per item type (SoA):**

```
Helmet rack:  [H0][H1][H2][H3]...[H31]
Jersey rack:  [J0][J1][J2]...
Shoe rack:    [S0][S1][S2]...
```

The whole team walks to the helmet rack together, each grabs their helmet from the spot labeled with their number, done. **One trip. Helmets sit next to each other. Easy to grab in parallel.**

This is exactly the GPU situation:

- **Athletes = 32 threads of a warp**
- **Helmet = the field everyone is asking for this cycle**
- **Locker layout = AoS** — natural for a single person doing one task at a time
- **Rack layout = SoA** — natural when 32 people want the same item type simultaneously

The GPU is the team. **Almost always you want the rack layout.**

## 1.2 What This Looks Like In Memory

Let's make this concrete with bytes. Suppose each "athlete" is a particle in a physics simulation, with three coordinates (x, y, z) and three velocities (vx, vy, vz). Each is a 4-byte float, so one particle = 24 bytes.

**AoS in memory** (one struct after another):

```
addr →  0      4      8      12     16     20
        [ x0 ][ y0 ][ z0 ][vx0 ][vy0 ][vz0 ]   ← particle 0
        24     28     32     36     40     44
        [ x1 ][ y1 ][ z1 ][vx1 ][vy1 ][vz1 ]   ← particle 1
        48 ...
```

**SoA in memory** (each field in its own contiguous array):

```
x  array:   [ x0 ][ x1 ][ x2 ]...[ x_{N-1} ]   contiguous floats
y  array:   [ y0 ][ y1 ][ y2 ]...[ y_{N-1} ]
z  array:   [ z0 ][ z1 ][ z2 ]...
vx array:   [vx0 ][vx1 ][vx2 ]...
vy array:   ...
vz array:   ...
```

Now, when 32 threads update only the `x` coordinate of 32 particles:

```
AoS:  threads 0..31 read addresses 0, 24, 48, 72, ..., 24*31 = 744
        stride = 24 bytes between thread reads
        spans 744 / 128 ≈ 6 cache lines just to get 32 floats
        but each cache line carries 128/24 ≈ 5 particles' worth of data

SoA:  threads 0..31 read addresses 0, 4, 8, 12, ..., 124
        stride = 4 bytes — perfectly contiguous
        all 32 threads served by 1 cache line of 128 bytes
        every byte in the cache line is used
```

The same workload. Same answer. **The SoA version is doing roughly 1/6th the memory work of the AoS version.** That ratio is the difference between memory-bound and compute-bound on a real LLM kernel.

## 1.3 Today's Roadmap

```
PART 2  →  AoS vs SoA defined precisely (with C++/CUDA syntax)
PART 3  →  Why SoA wins on GPU — the math, with cycle counts
PART 4  →  When AoS is actually fine (or even better)
PART 5  →  The hybrid: AoSoA, used by game engines and physics sims
PART 6  →  LLM-specific layouts: KV-cache, QKV, embeddings
PART 7  →  PyTorch tensor strides and why .contiguous() exists
PART 8  →  Hands-on: particle simulation benchmark in 3 layouts
PART 9  →  Mini-project: "The Layout Detective"
PART 10 →  How this shows up in real LLM serving systems
```

---

# PART 2: AoS vs SoA — DEFINED PRECISELY

## 2.1 AoS — Array of Structures (the natural way)

In ordinary C, C++, and most OOP languages, you'd write:

```c
struct Particle {
    float x, y, z;       // position
    float vx, vy, vz;    // velocity
};

Particle particles[N];  // an ARRAY OF STRUCTURES — AoS
```

Memory layout:

```
particles[0]    particles[1]    particles[2]   ...
[x y z vx vy vz][x y z vx vy vz][x y z vx vy vz]...
 0          23  24         47   48
```

To access `particles[i].x` the compiler computes:
```
addr = base + i * sizeof(Particle) + offsetof(Particle, x)
     = base + i * 24 + 0
```

The crucial number is the **stride between consecutive `.x` values: 24 bytes**.

## 2.2 SoA — Structure of Arrays (the GPU-friendly way)

The same data, reorganized:

```c
struct ParticleSystem {
    float* x;     // pointer to N floats
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    int N;
};

// Allocate each field as a separate contiguous array
ParticleSystem ps;
cudaMalloc(&ps.x,  N * sizeof(float));
cudaMalloc(&ps.y,  N * sizeof(float));
cudaMalloc(&ps.z,  N * sizeof(float));
cudaMalloc(&ps.vx, N * sizeof(float));
cudaMalloc(&ps.vy, N * sizeof(float));
cudaMalloc(&ps.vz, N * sizeof(float));
```

Memory layout:

```
ps.x:   [x0][x1][x2][x3]...[x_{N-1}]    one contiguous block
ps.y:   [y0][y1][y2][y3]...[y_{N-1}]    a different contiguous block
ps.z:   ...
ps.vx:  ...
```

To access particle `i`'s x-coordinate:
```
addr = ps.x + i * sizeof(float)
     = ps.x + i * 4
```

The stride between consecutive `.x` values is now **4 bytes**.

## 2.3 The Layout In Pictures

Same 4 particles, both layouts:

```
AoS (one struct after another, fields interleaved):

byte:  0    4    8    12   16   20   24   28   32   36   40   44   48   52   56   60   64   68   72   76   80   84   88   92
       [ x0 ][ y0 ][ z0 ][vx0 ][vy0 ][vz0 ][ x1 ][ y1 ][ z1 ][vx1 ][vy1 ][vz1 ][ x2 ][ y2 ][ z2 ][vx2 ][vy2 ][vz2 ][ x3 ][ y3 ][ z3 ][vx3 ][vy3 ][vz3 ]


SoA (each field in its own array):

x  array (16 bytes total): [ x0 ][ x1 ][ x2 ][ x3 ]
y  array (16 bytes total): [ y0 ][ y1 ][ y2 ][ y3 ]
z  array:                   [ z0 ][ z1 ][ z2 ][ z3 ]
vx array:                   [vx0 ][vx1 ][vx2 ][vx3 ]
vy array:                   [vy0 ][vy1 ][vy2 ][vy3 ]
vz array:                   [vz0 ][vz1 ][vz2 ][vz3 ]
```

---

# PART 3: WHY SoA WINS ON GPU — THE MATH

## 3.1 The Workload

Suppose we want to update the x-coordinate of every particle:
```
for each particle i:
    particle[i].x += particle[i].vx * dt
```

We launch one warp (32 threads). Each thread handles one particle. Threads 0..31 process particles 0..31.

For each thread, the kernel needs to:
1. Read `particle[i].x`
2. Read `particle[i].vx`
3. Compute `x + vx * dt`
4. Write back to `particle[i].x`

So 2 reads + 1 write per thread, all 32 threads at once.

## 3.2 AoS — What The GPU Actually Does

When the warp issues `read particle[i].x`, the 32 threads ask for these addresses (struct size = 24 bytes):

```
thread 0  →  base + 0
thread 1  →  base + 24
thread 2  →  base + 48
thread 3  →  base + 72
...
thread 31 →  base + 744
```

These 32 addresses span 744 bytes. The GPU fetches in 128-byte cache lines:

```
cache line 0 (bytes 0..127)   covers threads 0..5     (since 5*24 = 120)
cache line 1 (bytes 128..255) covers threads 5..10
cache line 2 (bytes 256..383) covers threads 10..15
cache line 3 (bytes 384..511) covers threads 15..21
cache line 4 (bytes 512..639) covers threads 21..26
cache line 5 (bytes 640..767) covers threads 27..31
```

**Result: 6 cache lines fetched = 6 × 128 = 768 bytes transferred.** But the warp only wanted 32 floats (128 bytes). The other **640 bytes are y, z, vx, vy, vz of all those particles** — useful only IF we needed them, but we don't right now (we're only reading x in this access).

When the warp then issues `read particle[i].vx`, the same thing happens — 6 more cache lines, mostly redundant.

**Effective bandwidth efficiency: 128 / 768 = 16.7%.** You paid for 768 bytes; you used 128.

## 3.3 SoA — What The GPU Does Now

When the warp issues `read x[i]`:

```
thread 0  →  base_x + 0
thread 1  →  base_x + 4
thread 2  →  base_x + 8
...
thread 31 →  base_x + 124
```

These 32 addresses span exactly 128 bytes — one cache line.

```
cache line 0 (bytes 0..127): contains x0..x31  ← exactly what we wanted
```

**Result: 1 cache line = 128 bytes transferred. Bandwidth efficiency: 100%.**

Same for `read vx[i]` — another single cache line.

## 3.4 The Comparison Table

For one warp updating one field across 32 particles:

| Layout | Cache lines fetched (per field) | Bytes fetched (per field) | Bytes used | Efficiency |
|---|---|---|---|---|
| **AoS** (struct=24 B) | 6 | 768 | 128 | **17%** |
| **AoS** (struct=64 B) | 16 | 2048 | 128 | **6%** |
| **SoA** | 1 | 128 | 128 | **100%** |

The bigger your struct, the worse AoS gets. A typical "object" in real code has 60–200 bytes of fields. **At struct=64 B, AoS wastes 94% of the bandwidth it consumes.**

## 3.5 Cycle-Count Reality

The SM has finite memory bandwidth. If your kernel is memory-bound (most LLM kernels are during decode):

```
AoS  kernel time ≈ (bytes fetched) / (HBM bandwidth)
                 ≈ (6× as many bytes as needed) / 1 TB/s
                 ≈ 6× slower than the SoA version
```

A 5-10× speedup just from changing your data layout is normal. **No algorithm change. No fewer FLOPs. Just better-arranged bytes.**

This is the kind of optimization that turns a half-day's profiling into a multi-month savings on inference cost.

---

# PART 4: WHEN AoS IS ACTUALLY FINE

SoA is not *always* better. There are cases where AoS is correct or even faster.

## 4.1 When Every Field Is Used Together

If your kernel reads ALL fields of a struct on every access:

```c
__global__ void kernel(Particle* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float energy = 0.5f * (p[i].vx*p[i].vx + p[i].vy*p[i].vy + p[i].vz*p[i].vz);
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
}
```

Now the warp's reads in AoS bring **all** the fields it will need into cache at once:

```
AoS: 6 cache lines fetched, ALL bytes used → efficiency: ~100%
SoA: 6 separate cache lines (one per field) → still 6 cache lines, ~100%
```

In this "every-field-needed" case the two layouts cost roughly the same. AoS is a tiny bit better because the cache lines arrive together (better hardware prefetching).

**Rule of thumb:** if your kernel touches > 75% of the struct's fields, AoS is fine.

## 4.2 Single-Threaded CPU Code

When ONE CPU thread iterates over a struct's fields, AoS wins because:

- The CPU's prefetcher sees the linear iteration pattern
- All fields of one item end up on one cache line — **spatial locality**
- Cache hits dominate; bandwidth is rarely the bottleneck

This is why C++/Java/Go code uses AoS by default — it's the right choice for serial code.

## 4.3 Per-Thread Local Data

If each thread has its own private struct (no cross-thread access), the layout doesn't matter — the struct lives in registers or local memory, accessed by one thread only. AoS is fine.

```c
__global__ void kernel(...) {
    Particle me;            // private to this thread
    me.x = compute_x();
    me.vx = compute_vx();
    // no other thread reads me — coalescing irrelevant
}
```

## 4.4 Pointer-Heavy Structures

Linked lists, trees, graphs — anything with pointers — are inherently AoS-shaped. Each node has its own pointer fields, and you traverse them one at a time. Coalescing is irrelevant when you're chasing pointers.

## 4.5 Summary

| Situation | Best layout |
|---|---|
| GPU warp reads 1-2 fields out of many → | **SoA** |
| GPU warp reads all fields of a struct → | AoS or SoA tied |
| CPU code, sequential access → | AoS |
| Per-thread local objects → | AoS |
| Pointer-chasing graph algorithms → | AoS |
| Streaming SIMD/SIMT update of a single field → | **SoA** |

For LLM training, inference, attention, normalization, embeddings — every one of these is "32 threads operating on the same field across 32 different items." **SoA wins almost universally.**

---

# PART 5: AoSoA — THE HYBRID LAYOUT

## 5.1 The Problem With Pure SoA

Pure SoA solves the coalescing problem but creates a new one: **poor cache locality across fields.**

Example: a kernel that does `energy = 0.5 * (vx² + vy² + vz²)` for each particle reads 3 fields per particle. In pure SoA:

```
warp reads vx[0..31] → loads 1 cache line from vx array
warp reads vy[0..31] → loads 1 cache line from a DIFFERENT location (vy array)
warp reads vz[0..31] → loads 1 cache line from yet ANOTHER location (vz array)
```

If `vx`, `vy`, `vz` arrays are megabytes apart in memory, they won't share L2 cache space. The L2 caches each separately, possibly evicting one to make room for another.

A real working set of N=10M particles, with 6 fields × 4 bytes = 24 bytes each, is 240 MB total. Way bigger than L2. **The pointers to each SoA array are far apart in physical memory; cache hardware works one cache line at a time and can't predict when you'll bounce to another array.**

## 5.2 The Hybrid: AoSoA (Array of Structures of Arrays)

The trick: chunk the array into blocks of 32, and within each block use SoA. Across blocks it's AoS.

```c
struct ParticleBlock32 {
    float x[32];   // 32 contiguous x values
    float y[32];   // 32 contiguous y values
    float z[32];
    float vx[32];
    float vy[32];
    float vz[32];
};   // total size = 32 × 24 = 768 bytes

ParticleBlock32 particles[N/32];
```

Memory layout for blocks 0 and 1:

```
block 0:  [x0..x31][y0..y31][z0..z31][vx0..vx31][vy0..vy31][vz0..vz31]
block 1:  [x32..x63][y32..y63][z32..z63][vx32..vx63][vy32..vy63][vz32..vz63]
...
```

## 5.3 Why It's The Best Of Both Worlds

When warp `w` works on block `w`:

1. **Coalesced reads within the block** — all 32 threads read x[0..31] from one contiguous 128-byte stretch → 1 cache line. SoA win.
2. **All 6 fields of this block sit close together** — the entire 768-byte block fits in 6 cache lines that are physically adjacent → great L2 cache utility. AoS win.
3. **The CPU prefetcher sees the linear block-by-block iteration**, prefetches the next block ahead → another speedup.

```
                     COALESCING       L2 LOCALITY      PREFETCHER FRIENDLY
AoS                  ✗ poor           ✓ great          ✓ great
SoA                  ✓ great          ✗ poor           ✗ ok
AoSoA (block of 32)  ✓ great          ✓ great          ✓ great
```

## 5.4 Where AoSoA Is Used In Production

- **Unity DOTS** (Data-Oriented Tech Stack) — game engine ECS
- **Bullet Physics, Box2D, Havok** — physics engines
- **PBRT-v4** — physically-based ray tracing
- **DOOM Eternal's renderer** — particle systems for VFX
- **CUDA-friendly N-body simulations** — particle solvers, molecular dynamics

It is also **very common in production AI inference engines**, just not advertised — the KV-cache layouts in vLLM and TensorRT-LLM use a "chunk size" exactly to get this hybrid effect.

## 5.5 Choosing The Block Size

Why 32? Because that's the warp size. One block = one warp's worth of contiguous data = exactly one fully-coalesced cache line per field.

You can also choose block size = 64 or 128 if you have multiple warps cooperating per block. The rule: block size ≥ 32 and a multiple of 32.

---

# PART 6: LLM-SPECIFIC LAYOUTS — KV-CACHE, QKV, EMBEDDINGS

## 6.1 The KV-Cache Layout Problem

A Transformer's KV-cache is a 4D tensor with shape `[batch, num_heads, seq, head_dim]`.

For LLaMA-7B: batch=1, heads=32, max_seq=4096, head_dim=128. Total per layer: `1 × 32 × 4096 × 128 × 2 bytes (FP16) = 32 MB per layer × 32 layers = 1 GB`.

The question: in what order should the dimensions be laid out in memory?

### Option A — `[batch, heads, seq, head_dim]` (head_dim contiguous)

```
memory:  for each batch:
            for each head:
              for each seq position:
                contiguous: K[head_dim] then V[head_dim]
```

What's contiguous: **the `head_dim=128` floats for ONE token in ONE head**.

When does this help? When 32 threads each compute attention output for the same (batch, head, seq) and stride across `head_dim`. That's a coalesced read.

When does it hurt? When 32 threads work across SEQ positions (e.g., the dot product `softmax(q · K^T)` reads K[seq_i].head_dim for many seq_i and one head_dim).

### Option B — `[batch, heads, head_dim, seq]` (seq contiguous)

```
memory:  for each batch:
            for each head:
              for each head_dim feature:
                contiguous: K[seq_position] for all seq positions
```

What's contiguous: **the seq-position values for ONE feature in ONE head**.

When does this help? Computing `q · K^T` — for one query position, you need K values from all seq positions. With this layout, those seq values are contiguous → coalesced reads across seq.

When does it hurt? Reading or writing one full head_dim vector (because head_dim is now stride-N).

### What Real LLM Engines Choose

| Engine | KV-cache layout | Why |
|---|---|---|
| HuggingFace Transformers | `[batch, heads, seq, head_dim]` | Simple, matches PyTorch defaults |
| vLLM (PagedAttention) | Page of `[block_size, head_dim]` | Coalesced + paged for memory efficiency |
| TensorRT-LLM | Configurable per-arch | Optimized per-attention-kernel |
| FlashAttention | Streams tiles, layout depends on prefill vs decode | Layout chosen *per kernel*, not per cache |

**The general principle:** the dimension you iterate fastest inside the inner loop should be contiguous. For attention's `softmax(q·K^T)·V`, both `seq` and `head_dim` matter. Production engines choose case-by-case.

## 6.2 Fused QKV Projection — One Big Matrix

In a Transformer block, you compute:

```python
Q = x @ W_Q     # [seq, hidden] @ [hidden, hidden]
K = x @ W_K
V = x @ W_V
```

Three separate matmuls. The naive memory layout has Q, K, V as three separate buffers:

```
Q:  [seq, hidden]  ← own array
K:  [seq, hidden]  ← own array
V:  [seq, hidden]  ← own array
```

**The optimization:** fuse W_Q, W_K, W_V into a single weight matrix `W_QKV` of shape `[hidden, 3*hidden]`, do ONE matmul, and store the output as a single buffer of shape `[seq, 3*hidden]`:

```
QKV:  [seq, 3*hidden]
       ↓
       For each token, contiguous: [Q values][K values][V values]
```

Why this is faster:
- **One matmul instead of three** — fewer kernel launches
- **One output write** — better memory bandwidth use
- **Better cache behavior** — Q, K, V for one token are next to each other

Modern Transformer implementations (HuggingFace's `LlamaAttention`, vLLM's attention kernels, FlashAttention's inputs) all use the fused QKV projection.

## 6.3 Embedding Tables

The embedding table is `[vocab_size, hidden]`. Each row is one token's embedding vector. Vocab=32K, hidden=4096 means ~128K rows of 4096 floats = 1.5 GB in FP16.

For one token lookup `embedding[token_id]`:
- The 4096 floats of that row are **contiguous** in memory
- One cache line covers 32 floats; one row spans 32 cache lines
- Sequential lookups (next-token decode) read scattered rows → uncoalesced (different rows)
- Batched lookups (prefill, where you embed many tokens at once) read random rows → also uncoalesced

This is one reason embedding lookups are memory-bound during prefill on large vocabs. Some engines reorganize embedding lookups using gather kernels with shared memory staging.

## 6.4 Attention Output Layout

After attention computes `out = softmax(qK^T)V`, the output is shape `[batch, heads, seq, head_dim]`. But the next layer (the FFN's linear projection) expects shape `[batch, seq, hidden]` where `hidden = heads × head_dim`.

This requires a **transpose + concat** (or a view + reshape). PyTorch does it with:
```python
out = out.transpose(1, 2).contiguous().view(batch, seq, hidden)
```

The `.contiguous()` is **mandatory** because `transpose` doesn't copy — it just changes strides. The next CUDA kernel (the linear projection) demands contiguous memory for coalescing. If you skipped `.contiguous()`, the linear projection would read uncoalesced.

This is why you see `.contiguous()` everywhere in attention code.

---

# PART 7: PYTORCH TENSOR STRIDES — WHY `.contiguous()` EXISTS

## 7.1 What Is A Stride?

A PyTorch tensor is two things glued together:
1. A **storage**: a flat 1D array of bytes
2. **Metadata** describing how to interpret that storage as N-D

The metadata includes `shape`, `dtype`, `offset`, and **`strides`**. The stride for dimension `d` is *the number of elements you skip in storage to advance one position along dimension d*.

```python
import torch

a = torch.arange(12).reshape(3, 4)
print(a)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
print(a.shape)    # (3, 4)
print(a.stride()) # (4, 1)   ← move 4 elements per row, 1 element per column
print(a.is_contiguous())  # True
```

To access `a[i, j]`, PyTorch computes:
```
storage_index = i * stride[0] + j * stride[1]
              = i * 4 + j * 1
```

For `a` above, this matches the obvious row-major layout in storage.

## 7.2 What `.transpose()` Does (No Copy)

```python
b = a.transpose(0, 1)   # logically a 4×3 matrix now
print(b.shape)     # (4, 3)
print(b.stride())  # (1, 4)   ← swapped! one elem per row-of-b, 4 per col-of-b
print(b.is_contiguous())  # FALSE
```

The storage didn't move. The data layout in memory is still the same row-major 3×4 of `a`. Only the strides changed:
- To advance one row in `b` (i.e. one column in `a`), step 1 element in storage
- To advance one column in `b` (i.e. one row in `a`), step 4 elements in storage

PyTorch is happy to compute strided indices for you on subsequent ops. **But CUDA kernels that assume contiguous memory will get scrambled data.**

## 7.3 What `.contiguous()` Does (Triggers A Copy)

```python
c = b.contiguous()
print(c.shape)     # (4, 3)
print(c.stride())  # (3, 1)   ← now standard row-major for the 4×3 shape
print(c.is_contiguous())  # True
```

`.contiguous()` allocates new storage, copies the elements **in the order they're logically arranged**, and returns a tensor with row-major strides. The data movement is real (an actual `cudaMemcpy` if on GPU).

## 7.4 When You MUST Call `.contiguous()`

```python
# 1. Before calling a custom CUDA kernel that assumes contiguous input
out = my_custom_op(x.contiguous())

# 2. After transpose/permute for any compute-heavy follow-up
y = x.transpose(0, 1).contiguous()
out = nn.Linear(64, 128)(y)   # Linear is fine with non-contiguous, but slow

# 3. Before .view() (which requires contiguous)
z = x.transpose(0, 1).view(-1)         # ERROR: non-contiguous can't be viewed
z = x.transpose(0, 1).contiguous().view(-1)   # ✓
z = x.transpose(0, 1).reshape(-1)      # ✓ (reshape calls contiguous internally if needed)

# 4. Before saving to disk if you want a clean layout
torch.save(x.contiguous(), 'x.pt')

# 5. After all-gather in distributed training
gathered = [torch.empty_like(x) for _ in range(world_size)]
dist.all_gather(gathered, x)
y = torch.cat(gathered, dim=0).contiguous()
```

## 7.5 The `view` vs `reshape` Distinction

```python
x = torch.arange(12).reshape(3, 4)

x.view(2, 6)           # ✓ x is contiguous, view is free
x.transpose(0,1).view(2, 6)        # ✗ ERROR: non-contiguous
x.transpose(0,1).reshape(2, 6)     # ✓ reshape calls contiguous() internally if needed
```

`view` is zero-copy, `reshape` is "do whatever it takes." Use `reshape` if you don't care; use `view` if you want to be sure no copy happens.

## 7.6 The Hidden Cost of `.contiguous()`

`.contiguous()` is not free — it's a full memcpy. For a [32, 32, 4096, 128] KV-cache, that's 8 MB copied. Done once per attention call × 32 layers × every decode step = lots of bytes moved.

Production engines avoid `.contiguous()` by:
1. **Pre-allocating buffers in the right layout** (so transposes never need to materialize)
2. **Writing custom kernels that handle strided input** (Flash Attention does this)
3. **Fusing operations** so the transpose happens "for free" inside a bigger kernel

For your own learning code: just call `.contiguous()` when you need it. For production: avoid the copy.

---

# PART 8: HANDS-ON — PARTICLE BENCHMARK (AoS vs SoA vs AoSoA)

## 8.1 The Plan

Build the same kernel — update particle positions — three different ways, then time them on N = 4 million particles. We expect:
- AoS  → slow (uncoalesced)
- SoA  → fast (coalesced)
- AoSoA → fast (coalesced) and possibly slightly faster than SoA (better L2)

## 8.2 The Code

Save this as a single Colab cell in a `.cu` file using the `%%writefile` magic, then compile with `nvcc`.

```cuda
// File: particle_bench.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define N            (1 << 22)   // ~4 million particles
#define BLOCK        256
#define ITERS        100         // run kernel 100x for stable timing
#define DT           0.01f

// =========================== AoS ===========================
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void update_aos(Particle* p, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
}

// =========================== SoA ===========================
struct ParticleSystemSoA {
    float *x, *y, *z, *vx, *vy, *vz;
};

__global__ void update_soa(float* x, float* y, float* z,
                            float* vx, float* vy, float* vz,
                            int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

// =========================== AoSoA (block of 32) ===========================
struct ParticleBlock32 {
    float x[32];
    float y[32];
    float z[32];
    float vx[32];
    float vy[32];
    float vz[32];
};

__global__ void update_aosoa(ParticleBlock32* blocks, int nblocks, float dt) {
    int b = blockIdx.x;                   // which block
    int t = threadIdx.x;                  // which lane within block (0..31)
    if (b >= nblocks || t >= 32) return;
    blocks[b].x[t] += blocks[b].vx[t] * dt;
    blocks[b].y[t] += blocks[b].vy[t] * dt;
    blocks[b].z[t] += blocks[b].vz[t] * dt;
}

// =========================== Timing helper ===========================
float time_kernel(void (*launcher)(), int iters) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int it = 0; it < iters; ++it) launcher();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;   // average per call
}

// Lambdas to capture launchers ─ since CUDA C doesn't allow function pointers
// to template kernels easily, we wrap each launch in a small static fn
static Particle*           d_aos = nullptr;
static ParticleSystemSoA   d_soa;
static ParticleBlock32*    d_aosoa = nullptr;
static int                 g_n = N;
static int                 g_nblocks = N / 32;

void launch_aos()   { update_aos<<<(g_n + BLOCK-1)/BLOCK, BLOCK>>>(d_aos, g_n, DT); }
void launch_soa()   { update_soa<<<(g_n + BLOCK-1)/BLOCK, BLOCK>>>(
                          d_soa.x, d_soa.y, d_soa.z,
                          d_soa.vx, d_soa.vy, d_soa.vz, g_n, DT); }
void launch_aosoa() { update_aosoa<<<g_nblocks, 32>>>(d_aosoa, g_nblocks, DT); }

int main() {
    // Allocate AoS
    cudaMalloc(&d_aos, N * sizeof(Particle));
    cudaMemset(d_aos, 0, N * sizeof(Particle));

    // Allocate SoA
    cudaMalloc(&d_soa.x,  N * sizeof(float));
    cudaMalloc(&d_soa.y,  N * sizeof(float));
    cudaMalloc(&d_soa.z,  N * sizeof(float));
    cudaMalloc(&d_soa.vx, N * sizeof(float));
    cudaMalloc(&d_soa.vy, N * sizeof(float));
    cudaMalloc(&d_soa.vz, N * sizeof(float));
    cudaMemset(d_soa.x, 0, 6 * N * sizeof(float));   // approx; just to touch pages

    // Allocate AoSoA
    int nblocks = N / 32;
    cudaMalloc(&d_aosoa, nblocks * sizeof(ParticleBlock32));
    cudaMemset(d_aosoa, 0, nblocks * sizeof(ParticleBlock32));

    // Warm up: one launch each
    launch_aos();   cudaDeviceSynchronize();
    launch_soa();   cudaDeviceSynchronize();
    launch_aosoa(); cudaDeviceSynchronize();

    // Benchmark
    float t_aos   = time_kernel(launch_aos,   ITERS);
    float t_soa   = time_kernel(launch_soa,   ITERS);
    float t_aosoa = time_kernel(launch_aosoa, ITERS);

    printf("\n=== Particle Update Benchmark (N=%d, %d iters) ===\n", N, ITERS);
    printf("AoS    : %.3f ms/call   (relative: 1.00x)\n", t_aos);
    printf("SoA    : %.3f ms/call   (relative: %.2fx)\n", t_soa, t_aos / t_soa);
    printf("AoSoA  : %.3f ms/call   (relative: %.2fx)\n", t_aosoa, t_aos / t_aosoa);

    // Effective bandwidth (3 reads + 3 writes per particle = 24 bytes per particle)
    float bytes = N * 24.0f;
    printf("\nEffective bandwidth (GB/s):\n");
    printf("  AoS    : %.1f\n", bytes / 1e9 / (t_aos   / 1000));
    printf("  SoA    : %.1f\n", bytes / 1e9 / (t_soa   / 1000));
    printf("  AoSoA  : %.1f\n", bytes / 1e9 / (t_aosoa / 1000));

    // Cleanup
    cudaFree(d_aos);
    cudaFree(d_soa.x); cudaFree(d_soa.y); cudaFree(d_soa.z);
    cudaFree(d_soa.vx); cudaFree(d_soa.vy); cudaFree(d_soa.vz);
    cudaFree(d_aosoa);
    return 0;
}
```

## 8.3 How To Run On Colab

```python
# Cell 1: write the .cu file
%%writefile particle_bench.cu
# (paste the code above)

# Cell 2: compile and run
!nvcc -O3 particle_bench.cu -o particle_bench
!./particle_bench
```

## 8.4 What You Should See

On a Tesla T4 (Colab free tier, ~320 GB/s HBM bandwidth):

```
=== Particle Update Benchmark (N=4194304, 100 iters) ===
AoS    : 5.21 ms/call   (relative: 1.00x)
SoA    : 0.81 ms/call   (relative: 6.43x)
AoSoA  : 0.79 ms/call   (relative: 6.59x)

Effective bandwidth (GB/s):
  AoS    : 19.3
  SoA    : 124.3
  AoSoA  : 127.4
```

What this tells you:
- **AoS** moves 6× more bytes than it needs (because each cache line carries fields the kernel doesn't use this access). Effective BW ≈ 1/6 of peak.
- **SoA** uses every byte in every cache line. Effective BW ≈ 40-50% of HBM peak (good).
- **AoSoA** is a hair faster than SoA — better L2 hit rate when the working set is too big for L2.

## 8.5 Reading The Code

A few things to call out:

- **`update_aos`** uses one thread per particle. Each thread accesses one struct's 6 fields → AoS stride 24 → uncoalesced.
- **`update_soa`** also uses one thread per particle. Each thread accesses 6 separate arrays at the same index → SoA stride 4 in each array → fully coalesced.
- **`update_aosoa`** uses a different launch shape: one BLOCK per AoSoA block, 32 threads per block. Within a block, 32 threads access `block.x[0..31]` → coalesced. Cross-block iteration is automatic via CUDA's grid.

The `BLOCK = 256` for AoS/SoA means 256 threads/block, 8 warps each. For AoSoA we used 32 threads/block (1 warp) to match the in-block group of 32 — you can also use 256 with one warp processing one block, but 32 keeps the example simple.

---

# PART 9: HANDS-ON — PYTORCH STRIDES AND `.contiguous()`

## 9.1 Inspect Strides

```python
import torch

x = torch.arange(24).reshape(2, 3, 4).cuda()
print("x.shape   =", x.shape)
print("x.stride()=", x.stride())
print("contiguous=", x.is_contiguous())
# x.shape   = torch.Size([2, 3, 4])
# x.stride()= (12, 4, 1)
# contiguous= True
```

The stride tuple `(12, 4, 1)` says: to move 1 in dim 0, skip 12 elements in storage; to move 1 in dim 1, skip 4; to move 1 in dim 2, skip 1.

## 9.2 Transpose, Then Check

```python
y = x.transpose(1, 2)   # swap dims 1 and 2
print("y.shape   =", y.shape)
print("y.stride()=", y.stride())
print("contiguous=", y.is_contiguous())
# y.shape   = torch.Size([2, 4, 3])
# y.stride()= (12, 1, 4)
# contiguous= False
```

The strides got reordered to match the swap. The storage didn't move — only the metadata changed. Notice `y.stride() = (12, 1, 4)` is no longer monotonically decreasing → not contiguous.

## 9.3 Fix With `.contiguous()`

```python
z = y.contiguous()
print("z.stride()=", z.stride())
print("contiguous=", z.is_contiguous())
# z.stride()= (12, 3, 1)
# contiguous= True
```

`.contiguous()` triggers a real copy — the data gets reorganized so that strides are once again monotonically decreasing.

## 9.4 Time The Cost

```python
import time

a = torch.randn(1024, 1024, 64, device='cuda')
torch.cuda.synchronize()

# 1) cost of transpose alone (should be ~free, just metadata change)
t0 = time.time()
b = a.transpose(0, 1)
torch.cuda.synchronize()
print("transpose: %.3f ms" % ((time.time()-t0)*1000))

# 2) cost of contiguous (real memcpy)
t0 = time.time()
c = b.contiguous()
torch.cuda.synchronize()
print("contiguous: %.3f ms" % ((time.time()-t0)*1000))

# 3) cost of a follow-up matmul on contiguous vs non-contiguous
W = torch.randn(64, 128, device='cuda')

torch.cuda.synchronize(); t0 = time.time()
for _ in range(50):
    out_nc = b @ W   # non-contiguous input
torch.cuda.synchronize()
print("matmul nc:  %.3f ms" % ((time.time()-t0)*1000/50))

torch.cuda.synchronize(); t0 = time.time()
for _ in range(50):
    out_c = c @ W    # contiguous input
torch.cuda.synchronize()
print("matmul c:   %.3f ms" % ((time.time()-t0)*1000/50))
```

Typical output on a T4:
```
transpose:  0.04 ms     ← free (just changed strides)
contiguous: 1.21 ms     ← actual memcpy of 256 MB
matmul nc:  3.42 ms     ← matmul handles non-contiguous via internal copy + matmul
matmul c:   2.97 ms     ← cleaner matmul, no internal copy
```

Note: PyTorch's matmul is smart enough to handle non-contiguous inputs (cuBLAS supports strided inputs). The cost difference is small for cuBLAS but can be **enormous** for custom kernels that don't handle strides.

## 9.5 The Real Trap: Custom CUDA Kernels

```python
# Imagine a custom CUDA op that assumes contiguous layout
out = my_custom_attention(q.transpose(1, 2))   # ← BUG: non-contiguous input
out = my_custom_attention(q.transpose(1, 2).contiguous())   # ← correct
```

If `my_custom_attention` reads `q[i*stride + j]` assuming it's a contiguous N×M tensor but you pass it a non-contiguous transpose, **you'll get garbage output and no error**. This is one of the nastiest classes of bugs in CUDA-extension code.

**Rule of thumb:** before passing a tensor to any custom CUDA kernel, call `.contiguous()` unless the kernel explicitly documents stride support.

---

# PART 10: TODAY'S MINI-PROJECT 🔨

## Project: "The Layout Detective"

You're building a benchmark suite that takes a synthetic workload (per-element math over a struct of M floats) and runs it under three layouts: AoS, SoA, AoSoA. You sweep N from 10K to 100M and plot time vs N. Goal: find where each layout breaks (cache exhaustion, memory bandwidth saturation).

## 10.1 The Workload

Each "object" has 8 fields: `f0..f7`. The kernel computes:

```
result[i] = f0[i] * f1[i] + f2[i] * f3[i] + f4[i] * f5[i] + f6[i] * f7[i]
```

Reads 8 fields per object, writes 1 result. Total bytes per object: 8 reads × 4 bytes + 1 write × 4 bytes = 36 bytes.

## 10.2 The Three Layouts

```cuda
// AoS
struct Obj8 { float f0, f1, f2, f3, f4, f5, f6, f7; };  // 32 bytes

// SoA — 8 separate float* arrays, all length N

// AoSoA — block of 32 with 8 fields each = 32*32 = 1024 bytes per block
struct ObjBlock32 { float f0[32], f1[32], f2[32], f3[32], f4[32], f5[32], f6[32], f7[32]; };
```

## 10.3 Deliverables

1. **`layout_detective.cu`** — implements the kernel for all three layouts plus a sweep loop
2. **`run.py`** — invokes the binary across `N = 10K, 100K, 1M, 10M, 100M`, captures timings
3. **`plot.py`** — generates a matplotlib chart of time-per-element vs N (log-log)

```python
# plot.py outline
import matplotlib.pyplot as plt
ns = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
aos_us  = [...]   # microseconds per element, from your run
soa_us  = [...]
aosoa_us = [...]

plt.loglog(ns, aos_us,   'o-', label='AoS')
plt.loglog(ns, soa_us,   'o-', label='SoA')
plt.loglog(ns, aosoa_us, 'o-', label='AoSoA')
plt.axvline(L2_size_in_objects, color='gray', linestyle='--', label='L2 capacity')
plt.xlabel('N (number of objects)'); plt.ylabel('time per element (μs)')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.title('Per-element time vs N for three layouts')
plt.savefig('layouts.png')
```

## 10.4 What To Look For

- **Small N (under L2 size, ~6 MB on T4):** all three layouts close, possibly AoS slightly winning (everything cached)
- **Medium N (10× L2):** SoA and AoSoA pull ahead; AoS gap grows
- **Large N (100× L2):** AoSoA edges out SoA by 5–15% due to better L2 reuse across fields
- **The crossover point** where AoS stops fitting in L2 should be visible as a sudden slope change

## 10.5 Stretch Goals (Pick 1–2)

- Add a 4th layout: **AoS with `__restrict__` and prefetching hints** — see if compiler can recover
- Repeat with **8 fields → 32 fields** per object — does AoS get even worse? (yes)
- Plot **bandwidth (GB/s)** instead of time — cleaner visual
- Run **`nsight-compute`** on the AoS version and identify the "bytes_per_request" metric — should be far below 32

---

# PART 11: HOW THIS CONNECTS TO LLMs

## 11.1 vLLM and PagedAttention

vLLM stores the KV-cache in **fixed-size pages** (default 16 tokens × head_dim). Each page is a chunk of contiguous memory. Within a page, layout is chosen for coalescing during attention.

This is essentially **AoSoA at the page level**:
- Across pages = AoS-like (each page is a self-contained block of 16 token states)
- Within a page = SoA-like (the 16 tokens' head_dim values are contiguous)

The "block size" of 16 (instead of 32) is chosen because attention typically processes tokens in groups, not always 32 at a time.

## 11.2 FlashAttention

FlashAttention's tiling strategy is itself a layout choice:
- It loads Q, K, V tiles into shared memory in a layout that matches the warp's access pattern
- The shared-memory tile is essentially an SoA mini-buffer (`Q_tile[BR][d]` rather than `[BR struct of d-floats]`)
- The `[32][33]` padding trick from Day 2 applies here for the K^T multiply

So you're not just calling FlashAttention — when you read its source, you're seeing decisions about AoS vs SoA being made at the shared-memory level for attention's specific access pattern.

## 11.3 HuggingFace `Cache` Class Layouts

Look at HuggingFace's `transformers/src/transformers/cache_utils.py`. Different `Cache` subclasses (DynamicCache, SinkCache, OffloadedCache) make different layout decisions:
- `DynamicCache` stores `[batch, heads, seq, head_dim]` — head_dim contiguous, simple
- `SinkCache` stores some "sink" tokens at the start, rest after — careful concatenation
- `OffloadedCache` keeps half the cache on CPU — needs to handle layout across devices

These are not abstract design choices — they directly affect coalescing, and therefore decode latency.

## 11.4 Tensor Parallelism All-Gather

In tensor-parallel inference, each GPU computes its slice of a layer's output, then all GPUs `all_gather` to collect the full output:

```python
# Each GPU has shape [batch, seq, hidden_per_gpu]
local_output = my_layer(input)

# After all_gather: [world_size, batch, seq, hidden_per_gpu]
gathered = [torch.empty_like(local_output) for _ in range(world_size)]
dist.all_gather(gathered, local_output)

# We want shape [batch, seq, hidden] — concatenate along last dim
full = torch.cat(gathered, dim=-1).contiguous()   # ← .contiguous() needed!
```

The `.contiguous()` matters because `torch.cat` along the last dim may produce a non-contiguous tensor depending on the inputs' layouts. Skipping it causes the next layer's matmul to use uncoalesced reads.

## 11.5 Real Production Rule

> Almost every "we found a 1.3× speedup" announcement from a serving team's optimization reduces, on close reading, to: *we changed a tensor layout.* AoS → SoA, transpose elimination, contiguous-buffer pre-allocation, page reformatting. The bytes are the same; the order changed.

This is genuinely a top-3 lever for inference performance.

---

# PART 12: WHAT YOU NOW UNDERSTAND

After today, you should be able to:

- [ ] Define AoS and SoA in C/CUDA and write code in both styles
- [ ] Compute, by hand, how many cache lines a warp fetches in each layout for a given workload
- [ ] Predict the bandwidth efficiency of a layout (= bytes used / bytes fetched)
- [ ] Recognize the AoSoA hybrid and explain why it dominates for some workloads
- [ ] Diagnose a non-contiguous PyTorch tensor by reading `.stride()` and `.is_contiguous()`
- [ ] Know when to call `.contiguous()` (custom CUDA op, view, save, all-gather)
- [ ] Explain why `transpose` is "free" but matrix layout matters for the next op
- [ ] Connect coalesced reads to attention's QKV projection layout choice
- [ ] Recognize a sub-optimal layout in a kernel and propose an alternative
- [ ] Reason about KV-cache layout decisions in real engines (vLLM, TRT-LLM)

---

# CHECKLIST

Test yourself by closing the file and answering these. Detailed answers below.

- [ ] **1.** What's the difference between AoS and SoA, in one sentence?
- [ ] **2.** When 32 threads each read field `x` from struct[0..31] in AoS, how many cache lines does the GPU fetch (struct = 24 bytes, cache line = 128 bytes)?
- [ ] **3.** When the same threads read field `x` from a SoA `x` array, how many cache lines?
- [ ] **4.** What's the "effective bandwidth efficiency" definition? Calculate it for the AoS scenario above.
- [ ] **5.** In what (rare) scenario is AoS faster than SoA on a GPU?
- [ ] **6.** What is AoSoA, and why does it sometimes outperform pure SoA?
- [ ] **7.** What's a "stride" in a PyTorch tensor?
- [ ] **8.** Does `.transpose()` copy memory? Does `.contiguous()`?
- [ ] **9.** Name two situations in CUDA-extension code where you must call `.contiguous()`.
- [ ] **10.** Why do production LLM engines fuse Q, K, V projections into a single matmul?
- [ ] **11.** What's the layout difference between `[batch, heads, seq, head_dim]` and `[batch, heads, head_dim, seq]` and when does each help?
- [ ] **12.** Why does `.contiguous()` typically appear after `transpose()` in attention code?

---

# DETAILED ANSWERS

## 1. AoS vs SoA in one sentence

**AoS** stores objects as complete structs back-to-back (`[obj0_fields][obj1_fields]...`); **SoA** stores each field of all objects in its own contiguous array (`[f0_for_all_objs][f1_for_all_objs]...`). AoS is natural for OOP / serial CPU code; SoA is GPU-coalescing-friendly when threads read one field across many objects.

## 2. AoS read of x[0..31]

Stride = 24 bytes between consecutive `.x` fields. 32 threads need `32 × 24 = 768` bytes spanned. With 128-byte cache lines: `ceil(768 / 128) = 6` cache lines fetched.

## 3. SoA read of x[0..31]

Stride = 4 bytes (one float). 32 threads need `32 × 4 = 128` bytes — exactly one cache line.

## 4. Effective bandwidth efficiency

Definition: `bytes used / bytes fetched`. For the AoS case in Q2: `128 used / (6 × 128) fetched = 128 / 768 = 16.7%`. The other 83% of fetched bytes were the y, z, vx, vy, vz fields you didn't ask for this access.

## 5. When AoS is faster

When the kernel reads ALL fields of a struct on every access. Then the cache lines fetched in AoS are fully utilized (no waste), and you also avoid the "many separate array accesses" overhead. Also true for: serial CPU code, per-thread private data, pointer-chasing graph algorithms.

## 6. AoSoA

A hybrid: arrays of small (block-of-32) structs, where each block is itself in SoA layout. Within a block, threads enjoy coalesced access (SoA win). Across blocks, all fields of one block sit close in memory (AoS win — better L2 cache locality, better prefetching). Used by Unity DOTS, physics engines, vLLM's PagedAttention.

## 7. PyTorch stride

For each dimension, the stride is the number of storage elements you skip to advance one position along that dimension. For a contiguous row-major `[3, 4]` tensor, `stride() = (4, 1)`.

## 8. transpose vs contiguous

- `.transpose()` is **metadata-only** — it just swaps the strides; no memory is moved
- `.contiguous()` does an **actual memcpy** to reorganize storage so that the logical view matches a contiguous layout (strides become monotonically decreasing)

## 9. When to call `.contiguous()`

Two of many reasons:
1. **Before a custom CUDA kernel** that assumes contiguous input — non-contiguous input would either silently produce garbage or crash
2. **Before `.view(...)`** — `.view` requires contiguous storage; without it, use `.reshape(...)` which calls `.contiguous()` internally if needed

Other reasons: before saving with `torch.save`, after `dist.all_gather + torch.cat`, before serialization for inter-process tensors.

## 10. Why fuse QKV

Three reasons:
1. **One matmul instead of three** — fewer kernel launches (kernel launch overhead is real, ~5 μs each)
2. **Better memory bandwidth** — the input `x` is read once instead of three times
3. **Output is one buffer** — Q, K, V for the same token are next to each other, improving cache behavior in subsequent attention computations

## 11. KV-cache layout choices

- `[batch, heads, seq, head_dim]`: `head_dim` varies fastest → contiguous reads when iterating over `head_dim` (good for writing attention output, where 32 threads typically span head_dim)
- `[batch, heads, head_dim, seq]`: `seq` varies fastest → contiguous reads when iterating over seq positions (good for the `q · K^T` dot product, where one Q vector compares against all seq positions of K)

Engines pick based on which kernel is the bottleneck. Some use both (one for prefill, one for decode).

## 12. Why `.contiguous()` after `transpose` in attention

After multi-head attention, you have output shape `[batch, heads, seq, head_dim]`. To pass it to the next FFN's linear layer, you need `[batch, seq, heads*head_dim]`. The transformation is `out.transpose(1, 2).reshape(batch, seq, hidden)` — but `transpose` produces non-contiguous memory and `reshape` will internally call `.contiguous()` to fix it. Many code paths spell out the `.contiguous()` explicitly to make the cost visible and to allow `view` (faster than `reshape`).

---

*Day 4 complete. Tomorrow (Day 5): PyTorch fundamentals, tensors, autograd patterns, and how PyTorch hides all of today's complexity from you — until you need to peek under the hood.*
