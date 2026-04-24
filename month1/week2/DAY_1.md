# Week 2, Day 1: The Story of GPU Memory
## How HBM, Cache Lines, and Coalescing Decide If Your LLM Runs Fast or Slow

**Time: ~2.5-3 hours**
**Setup: Google Colab with GPU runtime**

---

## Today's Mental Model

By the end of today, you should be able to look at a CUDA kernel and ask:

1. **How many useful bytes does each warp need?**
2. **How many bytes does the GPU actually fetch to satisfy that warp?**
3. **Is performance limited by math, HBM bandwidth, PCIe, or bad access pattern?**

This is the core skill behind fast GEMM, Flash Attention, vLLM paging, quantized
inference, and almost every "why is my GPU slow?" debugging session.

---

# PART 1: THE STORY

## 1.1 A Problem That Broke Everything

Imagine it's 2006. NVIDIA has just released CUDA. Researchers are excited — finally,
they can use thousands of GPU threads for general computing, not just graphics.

A physicist writes a simple program: "multiply each element of a large array by 2."

Expected:
- One CPU thread: **1 second**
- 1000 GPU threads: should be 1000x less = **1 ms**

He runs it. The GPU takes **100 ms**. 100x faster than CPU — impressive, but
not 1000x. Where did the other 900x go?

He profiles. The GPU cores are idle **99% of the time**. Why?

**Answer: they're waiting for data.**

```
  What he expected:                 What actually happened:
  ────────────────                  ──────────────────────
  ┌──────────────────┐              ┌──┬──────────────────┐
  │ ALL 1000 cores   │              │C │                   │
  │ ALL working      │              │O │  IDLE             │
  │ ALL the time     │              │M │  Waiting for      │
  │                  │              │P │  memory to        │
  │   ████████████   │              │U │  deliver data...  │
  │   ████████████   │              │T │                   │
  │   ████████████   │              │E │                   │
  │                  │              │  │                   │
  └──────────────────┘              └──┴──────────────────┘
   1 ms, 1000x speedup              100 ms, only 100x speedup
                                     Compute = 1%, Waiting = 99%
```

The compute was done in 1 ms. The remaining 99 ms was spent **moving numbers
from memory to the cores**. Even though the GPU had thousands of arithmetic units,
they were starving.

This is the fundamental problem of GPU computing, and it's the problem every
optimization in the last 20 years has tried to solve. Today you're going to
understand WHY this happens, and HOW to fix it.

## 1.2 Why Memory Is The Enemy

Here's the dirty secret of modern hardware: **compute has been doubling every
2 years (Moore's Law), but memory speed only grows ~7% per year**. The gap keeps
widening.

```
                    THE MEMORY WALL (widening gap)
                    ─────────────────────────────

  10000x │                                              ● Compute
         │                                          ╱
         │                                       ╱
   1000x │                                   ╱
         │                              ╱
         │                          ╱
    100x │                      ╱
         │                  ╱
         │              ╱  ─────────────────────   Memory
     10x │          ╱ ─────────────
         │    ╱ ─────
      1x │─────
         └─────────────────────────────────────────────────
        1980    1990    2000    2010    2020    2024
```

- **1980:** compute ≈ memory speed — balanced system
- **2024:** compute is ~1000x AHEAD of memory speed — massive imbalance

This gap is the #1 performance problem in computing. Every cache, every
prefetcher, every Tensor Core exists to fight it.

### The LLM Reality

GPU ratios look better in isolation, but the GPU also has 10,000+ parallel cores
all demanding data. Do the math for an H100 running Llama:

| Metric | H100 Value |
|---|---|
| Compute (FP16 Tensor Core) | 990 TFLOPS |
| Memory bandwidth | 3.35 TB/s |
| Ratio | **295 FLOPs per byte** |

To keep the cores fed, you need to do **295 operations per byte** read from memory.
But LLM inference does **~1 operation per byte** (read weight, multiply, done).

You're compute-capability 300x over what your memory can feed. The cores sit idle
299 out of 300 cycles. **That's why LLMs are memory-bound.**

Important nuance for later:

| Phase | What happens | Usually limited by |
|-------|--------------|--------------------|
| **Prefill** | Process the whole prompt in parallel | More compute-heavy because attention over many prompt tokens creates large matmuls |
| **Decode, batch=1** | Generate one new token at a time | Very memory-bound because weights are streamed for one token |
| **Decode, large batch** | Generate one token for many users at once | Less memory-bound because the same weights are reused across many sequences |

Today focuses on the **decode, small-batch** case because it exposes the memory
wall most clearly. Later, batching and continuous serving will teach you how
systems like vLLM and TensorRT-LLM push the workload toward better GPU utilization.

Intel learned this lesson in the 90s and invented caches. NVIDIA copied the idea
but had to solve an even harder problem: caches designed for 8 CPU cores don't
scale to 10,000 GPU threads. They needed something new.

That "something new" is what you're learning today.

## 1.3 The Insight That Made GPUs Viable

In 2008, NVIDIA engineers made a radical choice. Instead of one big cache,
they designed for **one specific access pattern**:

> "If 32 threads in a warp read 32 consecutive memory addresses,
> we'll make that one instruction. One transaction. One fetch."

This idea is **memory coalescing**. The GPU doesn't want to service 32 unrelated
thread memory requests one by one. Instead, it looks at all addresses requested by
the warp, groups nearby addresses into the fewest possible memory transactions,
and fetches aligned chunks of memory.

For today's beginner mental model, think "one aligned 128-byte chunk is perfect
for 32 FP32 values." Real NVIDIA GPUs use sectors, caches, and architecture-specific
transaction sizes, but the performance rule is stable:

> Consecutive warp addresses are cheap. Scattered warp addresses waste bandwidth.

**If threads access consecutive addresses (stride 1):**

```
Thread 0:  A[0]   (byte 0)
Thread 1:  A[1]   (byte 4)
Thread 2:  A[2]   (byte 8)
...
Thread 31: A[31]  (byte 124)
```

All within bytes 0-127 = one ideal 128-byte warp access.
32 threads served in the time of 1 transaction. **THIS is what makes GPUs fast.**

**If threads access scattered addresses:**

```
Thread 0:  A[0]      (byte 0)
Thread 1:  A[1000]   (byte 4000)
Thread 2:  A[50000]  (byte 200000)
...
```

Each in a different aligned chunk = many transactions. Much more data fetched than needed.
**Your 2024 H100 suddenly performs like a 2010 GPU.**

This single architectural choice — coalescing — is why NVIDIA dominates AI.
CPUs can't do this because their threads run independently on different cores.
GPU threads in a warp run IN LOCKSTEP, so the hardware can batch their memory
requests. That's the superpower.

Today you'll write code that exploits this superpower. And you'll write code that
violates it, and see the performance crater.

---

# PART 2: THE HARDWARE STORY

## 2.1 Why HBM Exists

CPU memory (DDR5) is designed for CPUs: few cores, pointer-chasing, latency matters.
It uses ~128 wires (a "64-bit bus × 2 channels") to connect to the CPU.

Now ask: how do you feed 10,000 GPU cores? Each core wants to read a float per cycle.
That's 10,000 × 4 bytes × 1.5 GHz = **60 TB/s** of theoretical demand.

DDR5 peaks at ~100 GB/s. You'd need 600 channels. Physically impossible.

The solution came from Hynix in 2013: **High Bandwidth Memory (HBM)**.

**HBM innovation:** stop trying to make memory chips FASTER — make them WIDER and SHORTER.

### Side-by-side physical comparison

```
  CPU SYSTEM (DDR5):                      GPU SYSTEM (HBM3):
  ──────────────────                       ───────────────────

   ┌──────────────┐                       ┌──────────┐ ┌──────────┐
   │     CPU      │                       │   HBM    │ │   HBM    │
   │              │                       │  stack   │ │  stack   │
   │              │                       │ ┌──────┐ │ │ ┌──────┐ │
   │              │                       │ │ die4 │ │ │ │ die4 │ │  ← stacked
   │              │                       │ ├──────┤ │ │ ├──────┤ │    vertically
   └──┬──┬──┬──┬──┘                       │ │ die3 │ │ │ │ die3 │ │
      │  │  │  │                          │ ├──────┤ │ │ ├──────┤ │
      │  │  │  │  (4 long wires           │ │ die2 │ │ │ │ die2 │ │
      │  │  │  │   per channel)           │ ├──────┤ │ │ ├──────┤ │
      │  │  │  │                          │ │ die1 │ │ │ │ die1 │ │
   ┌──┴──┴──┴──┴──┐                       │ └──┬───┘ │ │ └──┬───┘ │
   │              │                       └────┼─────┘ └────┼─────┘
   │  DDR5 DIMM   │                            │            │
   │  (separate   │                            │ (1024      │
   │   card)      │                            │  TINY TSVs │
   │              │                            │  per stack)│
   └──────────────┘                            ▼            ▼
                                          ┌───────────────────────┐
   128 wires total                        │                       │
   ~5 GHz × 128 / 8 = 100 GB/s            │      GPU die         │
                                          │  (compute happens    │
                                          │   RIGHT NEXT to      │
                                          │   memory)            │
                                          │                       │
                                          └───────────────────────┘

                                          5 HBM stacks × 1024 wires each
                                           = 5120 wires TOTAL
                                          ~2.6 GHz × 5120 / 8 = 3350 GB/s
                                          (33x more bandwidth than DDR5)
```

### The key insight: wires, not speed

| Memory | Wires | Clock | Bandwidth |
|--------|-------|-------|-----------|
| DDR5   | 128   | 5.0 GHz | **100 GB/s** |
| HBM3   | 5120  | 2.6 GHz | **3350 GB/s** |

HBM wins despite having a SLOWER clock — it makes up the difference with **40x more
wires in parallel**.

**HBM's secret:** short distance → wide bus → parallel data transfer. Since the memory
chips are stacked on top of each other and bonded to the GPU package, the wires can
be incredibly short and numerous.

DDR5 memory on a stick 10cm away from the CPU can't have 5000 wires — physical
routing limits it to ~128.

This is also why GPU memory is stuck on the card — you CAN'T upgrade GPU RAM like
you can upgrade DDR5 in your laptop. The HBM chips are physically bonded to the
GPU package.

## 2.2 The Cache Hierarchy: Making HBM Bearable

Even at 3.35 TB/s, HBM is **slow** compared to what GPU cores can consume.
So NVIDIA added multiple layers of caches:

```
                        THE GPU MEMORY PYRAMID
                        ──────────────────────

                    ▲ FASTEST  ────────  SMALLEST
                    │
                    │      ┌──────────────┐
                    │      │  REGISTERS   │     1 cycle  ▪ 256 KB / SM
                    │      │ per-thread   │     ↕ 100x   ▪ Private to 1 thread
                    │      └──────┬───────┘
                    │             │ spills if too many
                    │      ┌──────▼───────┐
                    │      │SHARED / L1   │    ~20 cyc   ▪ 228 KB / SM
                    │      │ per-block    │     ↕ 10x    ▪ Shared in block
                    │      └──────┬───────┘
                    │             │ miss
                    │      ┌──────▼───────┐
                    │      │   L2 CACHE   │   ~200 cyc   ▪ 50 MB total (H100)
                    │      │ all-SM shared│     ↕ 2-3x   ▪ 4 MB (T4)
                    │      └──────┬───────┘
                    │             │ miss
                    │      ┌──────▼───────┐
                    │      │GLOBAL MEMORY │   ~500 cyc   ▪ 80 GB (H100)
                    │      │    (HBM)     │              ▪ 16 GB (T4)
                    ▼      └──────────────┘              ▪ Where LLM weights live
                    SLOWEST ────────  LARGEST
```

### What lives where for an LLM (Llama-7B, 13.4 GB)

| Level | Holds |
|-------|-------|
| **Registers** | Current token's activations (a few floats per thread) |
| **Shared memory** | Tile of weights being multiplied RIGHT NOW |
| **L1 / L2 cache** | Recent attention activations, hopefully cached |
| **HBM (global)** | ALL 13.4 GB of weights + KV-cache |

**The killer fact:** NO cache can hold the full model (too big). In batch-1 dense
decode, generating each new token roughly streams the model weights from HBM.
Memory bandwidth is what determines your tokens/sec ceiling.

Reading a float from HBM vs from registers: **500x slower**. So caches matter ENORMOUSLY.

But here's the wrinkle: LLM weights (13.4 GB for Llama-7B FP16) can't fit in any
on-chip cache. They live in HBM. For small-batch decode, the GPU repeatedly streams
those weights layer by layer. Even with excellent caching of activations/KV data,
memory bandwidth is the bottleneck.

## 2.3 Cache Lines — The Unit of Memory Fetch

Your program says "read 4 bytes from address 100." The GPU doesn't actually read 4 bytes.
It reads an entire **cache line** containing that address.

Small but important nuance: CUDA programmers often say "cache line" when they
really mean **memory transaction / cache sector / aligned memory segment**. The
exact size depends on GPU generation, cache level, data type, and instruction.
For this lesson, we'll use a simplified model:

- A warp has 32 threads.
- Each thread reads one FP32 value = 4 bytes.
- A perfectly coalesced warp therefore uses 32 × 4 = **128 bytes**.

That 128-byte number is the mental anchor. Later, Nsight Compute will show you
the exact sector/request counters for your GPU.

### Visual: What a cache line looks like

Global memory is divided into fixed-size chunks called cache lines:

```
  Address:  0         128       256       384       512   ...
            ▼         ▼         ▼         ▼         ▼
            ┌─────────┬─────────┬─────────┬─────────┬────
  Memory:   │LINE 0   │LINE 1   │LINE 2   │LINE 3   │ ...
            │128 bytes│128 bytes│128 bytes│128 bytes│
            └─────────┴─────────┴─────────┴─────────┴────
```

Each L1 cache line holds 32 consecutive floats:

```
            ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  LINE 0:   │f0│f1│f2│f3│f4│f5│f6│f7│f8│f9│...........│f31│
            └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
            byte 0                                    byte 124
```

**The key rule:**
- You want 1 byte? You pay for 128 bytes.
- You want 32 consecutive bytes? You still pay for 128.
- You want bytes from 4 different lines? You pay for 4 × 128 = 512.

### Why different sizes at different cache levels?

| Cache | Line size | Total | Design philosophy |
|-------|-----------|-------|-------------------|
| **L1 / per-SM path** | Often reasoned about as 128-byte warp chunks | 228 KB class on H100 | Big aligned chunks match the ideal 32-thread FP32 warp access. |
| **L2 / sector path** | Often tracked in smaller sectors such as 32 bytes | 50 MB class on H100 | Smaller sectors reduce waste when many SMs request nearby but not identical data. |
| **HBM** (global) | — | 80 GB | The source. Slow but massive. |

The 128-byte number is not arbitrary. It's **exactly the size of a warp's
coalesced FP32 access**: 32 threads × 4 bytes each = 128 bytes. That's why this
pattern appears everywhere in CUDA teaching, profiling, and kernel design.

When you access memory consecutively, you get the cache line for free after the first
access — the next 31 reads hit cache. This is why **spatial locality** wins on GPUs
just like it wins on CPUs.

---

# PART 3: COALESCING — THE MOST IMPORTANT GPU CONCEPT

## 3.1 The Coalescing Rule

Every time a warp (32 threads) issues a memory access, the GPU:

1. Collects the 32 addresses each thread wants
2. Groups them by which cache line each falls into
3. Issues ONE memory transaction per unique cache line
4. Delivers each thread its byte(s) from the fetched lines

**The rule:** minimum transactions = minimum aligned memory chunks touched.

Best case: all 32 FP32 values fit in one 128-byte aligned chunk. FAST.
Worst case: all 32 values live in different chunks. Lots of wasted bandwidth.

This is why profiler metrics often talk about **global load efficiency**,
**sectors per request**, or **memory throughput**. They are all ways of asking:

> How much of the fetched memory did my program actually use?

## 3.2 Real Example With Numbers

Array of 1000 floats, each 4 bytes. Array starts at byte address 0.

### Pattern A: Consecutive (COALESCED — GOOD)
```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
float x = A[i];  // each thread reads A[thread_id]
```

```
VISUAL: WHERE EACH THREAD'S DATA LIVES (Pattern A — Coalesced)

Bytes:  0    32   64   96   128   ...
        │    │    │    │    │
        ▼    ▼    ▼    ▼    ▼
        ┌──────────────────────┬──────────────────────┐
Memory: │   CACHE LINE 0 (128B)│   CACHE LINE 1 (128B)│  ...
        │ ┌──┬──┬──┬──┬ ... ┬──┤ ┌──┬──┬──┬ ...       │
        │ │T0│T1│T2│T3│     │T31│ │  │  │  │           │
        │ └──┴──┴──┴──┴ ... ┴──┘ └──┴──┴──┴ ...       │
        └──────────────────────┴──────────────────────┘
         ↑
         │  All 32 threads' data lives HERE.
         │  GPU says: "I'll grab this whole line — one transaction."
```

| Metric | Value |
|--------|-------|
| Transactions | **1** (one 128-byte fetch) |
| Bytes fetched | 128 |
| Bytes used | 128 (32 threads × 4 bytes) |
| Efficiency | **100%** ✓ |
| Time | 1x (baseline) |

### Pattern B: Stride 32 (UNCOALESCED — BAD)

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
float x = A[i * 32];  // stride of 32
```

```
Bytes:  0        128       256       384  ...  3968
        │         │         │         │        │
        ▼         ▼         ▼         ▼        ▼
        ┌─────────┬─────────┬─────────┬───── ──┬─────────┐
Memory: │ LINE 0  │ LINE 1  │ LINE 2  │        │ LINE 31 │
        │[T0]░░░░░│[T1]░░░░░│[T2]░░░░░│  ...   │[T31]░░░░│
        └─────────┴─────────┴─────────┴────────┴─────────┘
         ▲         ▲         ▲                  ▲
         │         │         │                  │
         4 bytes   4 bytes   4 bytes            4 bytes
         USED      USED      USED               USED
         
         124       124       124                124
         WASTED    WASTED    WASTED             WASTED
```

Each thread hits a DIFFERENT cache line!

| Metric | Value |
|--------|-------|
| Transactions | **32** (one per thread!) |
| Bytes fetched | 4,096 (32 × 128) |
| Bytes used | 128 (same as before) |
| Efficiency | **3%** ✗ |
| Time | 32x slower (in effective bandwidth) |

### Pattern C: Stride 2 (half-bad)

```
Bytes:   0    32   64   96   128  160  192  224
         │    │    │    │    │    │    │    │
         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
         ┌──────────────────────┬──────────────────────┐
Memory:  │   CACHE LINE 0       │   CACHE LINE 1       │
         │T0  T1  T2  T3 ...T15│T16 T17 T18 ... T31   │
         │ ░  ░  ░  ░   ... ░  │ ░  ░  ░   ...  ░     │
         └──────────────────────┴──────────────────────┘
          16 threads in line 0    16 threads in line 1
```

| Metric | Value |
|--------|-------|
| Transactions | **2** (only 2 different lines) |
| Bytes fetched | 256 (2 × 128) |
| Bytes used | 128 |
| Efficiency | **50%** |
| Time | 2x slower |

### The stride penalty — summary table

| Stride | Cache lines touched | Transactions | Efficiency |
|--------|---------------------|--------------|------------|
| 1 | 1 | 1 | **100%** |
| 2 | 2 | 2 | 50% |
| 4 | 4 | 4 | 25% |
| 8 | 8 | 8 | 12% |
| 16 | 16 | 16 | 6% |
| 32 | 32 | 32 | **3%** |

Each stride doubling halves your effective bandwidth. This is THE performance
behavior to remember.

## 3.3 Why The Formula `i = blockIdx.x * blockDim.x + threadIdx.x` Matters

This is the canonical CUDA indexing pattern. For a simple 1D array, **it gives
coalesced access** because neighboring thread IDs read neighboring elements.

```c
// Thread 0 in block 0: i = 0
// Thread 1 in block 0: i = 1
// Thread 2 in block 0: i = 2
// ...
// Thread 0 in block 1: i = blockDim.x (e.g., 256)
// Thread 1 in block 1: i = blockDim.x + 1
```

**Within a warp** (32 threads with consecutive threadIdx.x values), `i` increments by 1.
`A[i]` accesses consecutive addresses. Coalesced.

This is why every CUDA tutorial uses this formula. It's not just convention — it's
the simplest way to get peak bandwidth for a 1D contiguous array.

Two caveats you should remember even this early:

1. Coalescing depends on the **addresses used by threads in the same warp**, not
   just the formula text.
2. In 2D/3D kernels, other indexing formulas can also be perfectly coalesced if
   neighboring threads still touch neighboring memory.

## 3.4 Connection To LLMs

Matrix multiply is THE LLM operation. How you index matters enormously.

### Row-major memory layout

A 3×4 matrix on paper vs. in memory:

```
A 3×4 matrix on paper:                In memory (1D, row-major):
                                      
     col:  0    1    2    3            byte addresses:
     ┌────────────────────┐            0  4  8 12 16 20 24 28 32 36 40 44
row 0│  1    2    3    4  │            ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
row 1│  5    6    7    8  │            │ 1│ 2│ 3│ 4│ 5│ 6│ 7│ 8│ 9│10│11│12│
row 2│  9   10   11   12  │            └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
     └────────────────────┘            └── row 0 ──┘└── row 1 ──┘└── row 2 ──┘
```

**Formula:** `A[row][col]` lives at byte `(row × cols + col) × 4`

### Access pattern 1 — ROW-wise (COALESCED ✓)

All threads read different COLS of the SAME ROW:

```
Thread 0: A[0][0] = byte  0
Thread 1: A[0][1] = byte  4      ← consecutive in memory
Thread 2: A[0][2] = byte  8      ← consecutive
Thread 3: A[0][3] = byte 12      ← consecutive
```

```
┌──┬──┬──┬──┐───────────────
│T0│T1│T2│T3│←── all in 1 cache line
└──┴──┴──┴──┘───────────────
```

**ONE transaction, full efficiency.** ✓

### Access pattern 2 — COLUMN-wise (UNCOALESCED ✗)

All threads read the SAME COL of different ROWS:

```
Thread 0: A[0][0] = byte  0
Thread 1: A[1][0] = byte 16      ← 16 bytes apart!
Thread 2: A[2][0] = byte 32      ← 16 bytes apart!
```

```
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│T0│░░│░░│░░│T1│░░│░░│░░│T2│░░│░░│░░│  ← threads SCATTERED
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘     across multiple lines
```

**Multiple transactions, wasted bandwidth.** ✗

### Why naive matrix multiply is slow

The standard CPU-style algorithm:

```c
for each output C[i][j]:
    for k in 0..K:
        C[i][j] += A[i][k] * B[k][j];
```

- `A[i][k]`: as k grows, bytes are consecutive → **row-wise → COALESCED** ✓
- `B[k][j]`: as k grows, bytes jump by `cols × 4` → **column-wise → UNCOALESCED** ✗

Even though you touch every element of B exactly once, the access pattern is
terrible. Naive matmul hits maybe 20 GB/s on a T4 that peaks at 320 GB/s —
a 10-30x slowdown vs a tiled version.

Flash Attention, cuBLAS, and every fast kernel are designed to make every memory
access coalesced. When you implement tiled matmul (this Saturday) and attention
(Week 11), you'll design layouts specifically so threads read consecutively.

---

# PART 4: THE cudaMemcpy STORY

## 4.1 Why cudaMemcpy Exists

CPU and GPU have **different memory spaces**. The CPU can't directly access GPU
memory (and vice versa) because they're physically separate chips connected by a
thin PCIe wire.

Early CUDA (2007) required explicit copies: `cudaMemcpy(gpu_ptr, cpu_ptr, bytes, H2D)`.
Programmers hated it. Lots of boilerplate, easy to mess up direction.

Over time, NVIDIA added conveniences — but the underlying reality hasn't changed.
PCIe is still the bottleneck between CPU and GPU.

## 4.2 The Bandwidth Reality

Three types of memory bandwidth in a CPU-GPU system, with enormous differences:

```
   ┌───────────────────────────┐         ┌──────────────────────────┐
   │         CPU               │         │           GPU             │
   │                           │         │                           │
   │   ┌──────────────────┐    │   PCIe  │    ┌──────────────────┐   │
   │   │    CPU CORES     │    │         │    │    GPU CORES     │   │
   │   │   ~8 × 4 GHz     │    │  slow!  │    │  ~10,000 cores   │   │
   │   └────────┬─────────┘    │ 25 GB/s │    │    1.5 GHz       │   │
   │            │              │◄───────►│    └─────────┬────────┘   │
   │    ~100    │              │         │              │   ~3350    │
   │    GB/s    ▼              │         │              ▼   GB/s     │
   │   ┌──────────────────┐    │         │    ┌──────────────────┐   │
   │   │    DDR5 RAM      │    │         │    │     HBM3         │   │
   │   │    32-128 GB     │    │         │    │    16-80 GB      │   │
   │   └──────────────────┘    │         │    └──────────────────┘   │
   │                           │         │                           │
   └───────────────────────────┘         └──────────────────────────┘
```

Relative bandwidth comparison:

| Bus | Bandwidth | Relative |
|-----|-----------|----------|
| PCIe 4.0 x16 (CPU ↔ GPU) | 25 GB/s | **1x** (slowest — the bottleneck) |
| DDR5 (CPU RAM) | 100 GB/s | 4x |
| HBM3 (GPU memory) | 3,350 GB/s | **134x** — massive advantage |

### Why this kills performance if you're not careful

**Moving 14 GB (Llama-7B weights) from CPU → GPU:**

`14 GB ÷ 25 GB/s = 560 ms`

**Running the same model once on GPU (reading weights from HBM):**

`14 GB ÷ 3,350 GB/s = 4.2 ms`

**One CPU→GPU copy costs as much as streaming the model weights from HBM 133 times.** If you copy
every token, you're throwing away 99% of your GPU's speed.

This is why when you call `.to('cuda')` in PyTorch, it's slow the first time and
fast forever after (the data stays on GPU). And why `model.cpu().cuda()` in a loop
is a performance disaster.

## 4.3 The Five Directions

```c
cudaMemcpy(dst, src, size, kind);
```

| Direction | Speed | When To Use |
|-----------|-------|-------------|
| `cudaMemcpyHostToDevice` | ~25 GB/s (PCIe) | Loading model weights, input data |
| `cudaMemcpyDeviceToHost` | ~25 GB/s (PCIe) | Reading results back for display |
| `cudaMemcpyDeviceToDevice` | ~3000 GB/s (HBM) | Copying within GPU — FAST |
| `cudaMemcpyHostToHost` | ~50 GB/s (CPU) | Rare, use `memcpy()` instead |
| `cudaMemcpyDefault` | auto-detected | Convenient but slight overhead |

**Modern twist — Unified Memory:**
```c
cudaMallocManaged(&ptr, size);
// One pointer, works on both CPU and GPU.
// Driver migrates pages automatically when needed.
```

Convenient but with hidden costs. When GPU accesses a page the CPU recently wrote,
the driver must copy it over at ~25 GB/s. Can be slower than explicit copies if
you don't understand the page migration pattern.

---

# PART 5: HANDS-ON — PROVING THE THEORY

Theory is worthless without proof. Let's measure coalescing's impact ourselves.

## 5.1 Exercise 1: Prove Coalescing Matters

```c
%%writefile coalesce_demo.cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        printf("CUDA error at %s:%d: %s\n",                     \
               __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(1);                                                 \
    }                                                            \
} while(0)

// GOOD: Each thread reads the address corresponding to its ID.
// Warp 0's threads access bytes 0-127 of A → one cache line.
__global__ void coalesced(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}

// BAD: Thread i reads A[i*stride]. Consecutive threads jump by 'stride'.
// For stride=32, each thread's address is in a different cache line.
__global__ void strided(float *in, float *out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < N) out[i * stride] = in[i * stride];
}

int main() {
    int N = 16 * 1024 * 1024;    // 16 million floats = 64 MB
    size_t bytes = N * sizeof(float);
    
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    
    int bs = 256;
    int blocks = (N + bs - 1) / bs;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("════════════════════════════════════════════════════\n");
    printf("  Coalescing Demo (16M floats = 64 MB)\n");
    printf("════════════════════════════════════════════════════\n");
    printf("  Pattern      │ Time (ms) │ Effective BW (GB/s)\n");
    printf("────────────────────────────────────────────────────\n");
    
    // Warmup + benchmark coalesced
    coalesced<<<blocks, bs>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        coalesced<<<blocks, bs>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_c;
    cudaEventElapsedTime(&ms_c, start, stop);
    ms_c /= 100;
    float bw_c = 2.0f * bytes / (ms_c/1000.0f) / 1e9;
    printf("  Coalesced    │ %8.3f │ %6.0f           \n", ms_c, bw_c);
    
    // Benchmark various strides
    int strides[] = {2, 4, 8, 16, 32};
    for (int s = 0; s < 5; s++) {
        int stride = strides[s];
        int N_eff = N / stride;
        int blocks_eff = (N_eff + bs - 1) / bs;
        
        strided<<<blocks_eff, bs>>>(d_in, d_out, N, stride);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++)
            strided<<<blocks_eff, bs>>>(d_in, d_out, N, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms_s;
        cudaEventElapsedTime(&ms_s, start, stop);
        ms_s /= 100;
        
        // Effective BW = useful bytes moved, not the larger amount fetched/wasted.
        float bw_s = 2.0f * N_eff * sizeof(float) / (ms_s/1000.0f) / 1e9;
        printf("  Stride %2d    │ %8.3f │ %6.0f           \n",
               stride, ms_s, bw_s);
    }
    
    printf("════════════════════════════════════════════════════\n");
    printf("  Lesson: stride=N means N-x bandwidth waste.\n");
    printf("════════════════════════════════════════════════════\n");
    
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

```python
!nvcc coalesce_demo.cu -o coalesce_demo && ./coalesce_demo
```

**What you'll see** (T4 numbers, yours will be similar):

```
Pattern     │ Time (ms) │ Effective BW (GB/s)
Coalesced   │    0.450  │     284
Stride  2   │    0.450  │     142
Stride  4   │    0.450  │      71
Stride  8   │    0.450  │      36
Stride 16   │    0.450  │      18
Stride 32   │    0.450  │       9
```

**Read this carefully.** The TIME stays roughly the same. But the **effective**
bandwidth crashes. Why? The GPU is fetching the same amount of data in all cases
(memory controllers saturated). But at stride 32, only 1/32 of the fetched bytes
are actually used by your kernel. You're paying for bandwidth you don't use.

In a real kernel where you care about USEFUL work per second, stride 32 is 32x
slower than coalesced. Coalescing isn't a "nice to have" — it's the difference
between a GPU feeling like a GPU vs. a GPU feeling like a slow CPU.

### Reading The Code

```c
__global__ void coalesced(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}
```
- `i = blockIdx.x * blockDim.x + threadIdx.x` — standard pattern.
- Warp 0: threadIdx.x = 0..31 → i = 0..31 → bytes 0..127 → ONE cache line.
- Warp 1: threadIdx.x = 0..31, blockIdx.x = 0, blockDim.x = 256 → i = 32..63 → bytes 128..255 → ONE cache line.
- Every warp's 32 threads access one cache line each. Optimal.

```c
__global__ void strided(float *in, float *out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < N) out[i * stride] = in[i * stride];
}
```
- Input index is `i * stride`, not `i`.
- With stride=32: threadIdx.x 0 → in[0]/out[0], threadIdx.x 1 → in[32]/out[32], ...
- Warp 0: threads access indices 0, 32, ..., 992 = bytes 0, 128, 256, ..., 3968.
- 32 different aligned chunks for both loads and stores. Disaster.

## 5.2 Exercise 2: Compute Your GPU's Theoretical Peak

```c
%%writefile peak_bw.cu
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Formula: 2 (DDR) × clock_rate × (bus_width / 8) / 1e9 GB/s
    // Why 2? Double-Data-Rate: memory transfers on both clock edges.
    // clock_rate is in kHz from CUDA → multiply by 1000 for Hz.
    // bus_width is in bits → divide by 8 for bytes.
    
    double clock_hz   = prop.memoryClockRate * 1000.0;
    double bus_bytes  = prop.memoryBusWidth / 8.0;
    double peak_gb_s  = 2.0 * clock_hz * bus_bytes / 1e9;
    
    printf("╔════════════════════════════════════════════╗\n");
    printf("║ GPU: %-38s ║\n", prop.name);
    printf("╠════════════════════════════════════════════╣\n");
    printf("║ Memory clock:     %.2f GHz                 ║\n", clock_hz / 1e9);
    printf("║ Memory bus width: %d bits                   ║\n", prop.memoryBusWidth);
    printf("║ Bytes per cycle:  %.0f                      ║\n", bus_bytes * 2);
    printf("║ Peak bandwidth:   %.1f GB/s                ║\n", peak_gb_s);
    printf("╚════════════════════════════════════════════╝\n");
    
    printf("\nWhat this means for LLMs:\n");
    printf("  Llama-7B FP16 (13.4 GB): max %.0f tokens/sec\n", peak_gb_s / 13.4);
    printf("  Llama-7B INT4 (3.4 GB):  max %.0f tokens/sec\n", peak_gb_s / 3.4);
    printf("  (memory-bound, decode phase)\n");
    
    return 0;
}
```

```python
!nvcc peak_bw.cu -o peak_bw && ./peak_bw
```

On T4 you'll see ~320 GB/s. Remember this number — it's your GPU's hard ceiling.
Every kernel you write should aim for 80-95% of this. Below 60% means something's wrong.

---

# PART 6: TODAY'S MINI-PROJECT 🔨

## "The Bandwidth Roast" — Test Your Own GPU's Memory System

You're going to build a comprehensive memory bandwidth tester that demonstrates
every concept from today: coalesced access, strided access, vectorized (float4) loads,
and the effect of data size. The output is a report you could actually paste in a
blog post or show to your team.

**Why this project matters:** Once you have this tool, you can run it on ANY kernel
you write. It tells you instantly if your kernel is memory-bound and how close to
peak bandwidth it gets. This IS how NVIDIA engineers analyze performance.

```c
%%writefile bw_dashboard.cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        printf("CUDA error: %s\n", cudaGetErrorString(err));      \
        exit(1);                                                 \
    }                                                            \
} while(0)

// Standard coalesced copy — our baseline
__global__ void copy_f32(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}

// Vectorized copy: each thread handles 4 floats at once.
// float4 is a struct of 4 floats stored contiguously (16 bytes aligned).
// Loading one float4 = one 16-byte instruction instead of four 4-byte ones.
__global__ void copy_f4(float4 *in, float4 *out, int N4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N4) out[i] = in[i];
}

// Strided copy — demonstrates uncoalesced penalty
__global__ void copy_stride(float *in, float *out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < N) out[i * stride] = in[i * stride];
}

// The "triad" operation from STREAM benchmark: C = A + s*B
// Reads 2 arrays, writes 1 array. Classic memory-bound pattern.
__global__ void triad(float *A, float *B, float *C, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + s * B[i];
}

float time_kernel(void (*run)(float*, float*, float*, int, int),
                   float *A, float *B, float *C, int N, int extra) {
    int bs = 256;
    
    // Warmup (first launch always slow)
    run(A, B, C, N, extra);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) run(A, B, C, N, extra);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100;
}

// Launchers so we can pass them as function pointers
void run_copy_f32(float *A, float *B, float *C, int N, int unused) {
    int bs = 256;
    copy_f32<<<(N+bs-1)/bs, bs>>>(A, B, N);
}
void run_copy_f4(float *A, float *B, float *C, int N, int unused) {
    int bs = 256;
    int N4 = N / 4;
    copy_f4<<<(N4+bs-1)/bs, bs>>>((float4*)A, (float4*)B, N4);
}
void run_stride(float *A, float *B, float *C, int N, int stride) {
    int bs = 256;
    int N_eff = N / stride;
    copy_stride<<<(N_eff+bs-1)/bs, bs>>>(A, B, N, stride);
}
void run_triad(float *A, float *B, float *C, int N, int unused) {
    int bs = 256;
    triad<<<(N+bs-1)/bs, bs>>>(A, B, C, 2.5f, N);
}

int main() {
    // Query device for theoretical peak
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double peak = 2.0 * prop.memoryClockRate * 1000.0 *
                   (prop.memoryBusWidth / 8.0) / 1e9;
    
    int N = 64 * 1024 * 1024;    // 64M floats = 256 MB
    size_t bytes = N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║              BANDWIDTH DASHBOARD                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ GPU:              %-40s║\n", prop.name);
    printf("║ Peak bandwidth:   %.0f GB/s                              ║\n", peak);
    printf("║ Data size:        %d MB                                  ║\n", (int)(bytes/1024/1024));
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Test             │ Time (ms) │ BW (GB/s)│ %% of peak      ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    
    // Coalesced baseline
    float ms = time_kernel(run_copy_f32, d_A, d_B, d_C, N, 0);
    float bw = 2.0f * bytes / (ms/1000.0f) / 1e9;
    printf("║ Copy (float)     │ %8.3f │ %6.0f  │ %5.1f%% of peak  ║\n",
           ms, bw, 100.0f*bw/peak);
    
    // Vectorized float4 — usually hits higher peak
    ms = time_kernel(run_copy_f4, d_A, d_B, d_C, N, 0);
    bw = 2.0f * bytes / (ms/1000.0f) / 1e9;
    printf("║ Copy (float4)    │ %8.3f │ %6.0f  │ %5.1f%% of peak  ║\n",
           ms, bw, 100.0f*bw/peak);
    
    // Triad — realistic LLM-style workload
    ms = time_kernel(run_triad, d_A, d_B, d_C, N, 0);
    bw = 3.0f * bytes / (ms/1000.0f) / 1e9;   // 3 arrays: read A, read B, write C
    printf("║ Triad (A+s*B)    │ %8.3f │ %6.0f  │ %5.1f%% of peak  ║\n",
           ms, bw, 100.0f*bw/peak);
    
    // Strided — the punishment
    int strides[] = {2, 4, 8, 16, 32};
    for (int s = 0; s < 5; s++) {
        int stride = strides[s];
        ms = time_kernel(run_stride, d_A, d_B, d_C, N, stride);
        float bw_eff = 2.0f * (N/stride) * sizeof(float) / (ms/1000.0f) / 1e9;
        printf("║ Stride %2d         │ %8.3f │ %6.0f  │ %5.1f%% of peak  ║\n",
               stride, ms, bw_eff, 100.0f*bw_eff/peak);
    }
    
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ LLM predictions (decode phase, memory-bound):             ║\n");
    printf("║   Llama-7B FP16 (13.4 GB):  ~%.0f tokens/sec              ║\n",
           peak / 13.4);
    printf("║   Llama-7B INT8 (6.7 GB):   ~%.0f tokens/sec              ║\n",
           peak / 6.7);
    printf("║   Llama-7B INT4 (3.4 GB):   ~%.0f tokens/sec              ║\n",
           peak / 3.4);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

```python
!nvcc bw_dashboard.cu -o bw_dashboard && ./bw_dashboard
```

**Expected output on T4:**
```
╔═══════════════════════════════════════════════════════════╗
║              BANDWIDTH DASHBOARD                           ║
╠═══════════════════════════════════════════════════════════╣
║ GPU:              Tesla T4                                ║
║ Peak bandwidth:   320 GB/s                                ║
║ Data size:        256 MB                                  ║
╠═══════════════════════════════════════════════════════════╣
║ Test             │ Time (ms) │ BW (GB/s)│ % of peak       ║
╠═══════════════════════════════════════════════════════════╣
║ Copy (float)     │    1.800  │    284   │ 88.8% of peak   ║
║ Copy (float4)    │    1.650  │    310   │ 96.9% of peak   ║  ← best!
║ Triad (A+s*B)    │    2.700  │    285   │ 89.0% of peak   ║
║ Stride  2        │    1.800  │    142   │ 44.4% of peak   ║
║ Stride  4        │    1.800  │     71   │ 22.2% of peak   ║
║ Stride  8        │    1.800  │     36   │ 11.2% of peak   ║
║ Stride 16        │    1.800  │     18   │  5.6% of peak   ║
║ Stride 32        │    1.800  │      9   │  2.8% of peak   ║
╠═══════════════════════════════════════════════════════════╣
║ LLM predictions (decode phase, memory-bound):             ║
║   Llama-7B FP16 (13.4 GB):  ~24 tokens/sec                ║
║   Llama-7B INT8 (6.7 GB):   ~48 tokens/sec                ║
║   Llama-7B INT4 (3.4 GB):   ~94 tokens/sec                ║
╚═══════════════════════════════════════════════════════════╝
```

Look at the stride column. Every doubling of stride halves your effective bandwidth.
This is the SINGLE most important perf behavior on GPUs. If someone tells you
their kernel is slow and the GPU is memory-bound, this is where to look first.

And the LLM predictions at the bottom? Treat them as **upper-bound intuition**,
not a promise. Real vLLM/TensorRT-LLM throughput depends on KV-cache traffic,
attention cost, batching, sampling overhead, kernel launch overhead, and serving
scheduler behavior. But the direction is correct: quantization can greatly improve
decode throughput because it reduces the number of bytes read from memory.

The whole quantization industry is built on this memory-bandwidth reality you just
measured.

---

# PART 7: HOW TO READ THIS IN A PROFILER

You do not need Nsight Compute today, but you should know what you will look for
when you start profiling real kernels.

| Question | Profiler clue |
|----------|---------------|
| Am I memory-bound? | High memory throughput, low SM compute utilization |
| Are my loads coalesced? | Good global load efficiency, low sectors/request |
| Am I wasting memory traffic? | Many bytes transferred but low useful work |
| Is PCIe hurting me? | Large host-device memcpy time in timeline |
| Is my kernel launch overhead dominating? | Many tiny kernels, low GPU occupancy over time |

The most important habit: never say "the GPU is slow." Say which resource is
limiting you: **HBM bandwidth, PCIe bandwidth, cache behavior, occupancy, launch
overhead, or compute throughput**.

## Common Mistakes To Avoid

- **Mistake 1: Timing GPU code without synchronization.** CUDA launches are async.
  Use CUDA events or `cudaDeviceSynchronize()` around CPU timers.
- **Mistake 2: Reporting only elapsed time.** Always convert to effective GB/s
  or TFLOPS so you can compare to hardware limits.
- **Mistake 3: Counting only useful bytes.** Useful bandwidth and actual memory
  traffic are different. Uncoalesced access may fetch far more than it uses.
- **Mistake 4: Copying between CPU and GPU inside a loop.** Keep data on the GPU.
- **Mistake 5: Assuming every LLM phase is equally memory-bound.** Decode batch=1
  is the clearest memory-bound case; prefill and batched decode behave differently.

---

# PART 8: WHAT YOU NOW UNDERSTAND

Close this document and try to answer these without looking:

1. **Why is HBM faster than DDR5?** (hint: physical structure, wire count)
2. **What's a cache line and why does size matter?** (hint: spatial locality, warp size)
3. **What's the difference between coalesced and uncoalesced access?** (hint: lockstep threads)
4. **Why is `i = blockIdx.x * blockDim.x + threadIdx.x; A[i]` always coalesced?** (hint: warp thread IDs are consecutive)
5. **Why is LLM inference memory-bound?** (hint: model size vs compute-per-byte)
6. **How do you measure bandwidth efficiency?** (hint: bytes used / bytes fetched)
7. **Why does quantization speed up inference?** (hint: smaller model → fewer bytes → less read time)

If you can answer these, you now understand memory bandwidth deeper than 95% of
ML engineers who write CUDA code. This knowledge will compound into every kernel
you see — Flash Attention, cuBLAS, Triton, vLLM — all of them are exercises in
minimizing memory access and maximizing coalescing.

---

# CHECKLIST

- [x] Understand the memory-compute gap and why it defines GPU performance
- [x] Know HBM's physical structure and why it's ~30x faster than DDR5
- [x] Understand cache lines/sectors and why 32 FP32 warp loads map naturally to 128 bytes
- [x] Can explain coalescing with a specific example (stride 1 vs stride 32)
- [x] Know why `blockIdx.x * blockDim.x + threadIdx.x` is the canonical pattern
- [x] Can calculate peak bandwidth from memory clock and bus width
- [x] Can measure effective bandwidth and compare to peak
- [x] Know when to use each cudaMemcpy direction (H2D, D2H, D2D, default)
- [x] Understand PCIe as the CPU-GPU bottleneck
- [x] Know why decode batch=1 is more memory-bound than prefill or large-batch decode
- [x] Know what profiler clues indicate memory-bound vs compute-bound behavior
- [x] Ran all three code exercises and saw coalescing penalty firsthand
- [x] Built the Bandwidth Dashboard and read the output intelligently
- [x] Can predict LLM inference speed from memory bandwidth measurements

---

# DETAILED ANSWERS

## 1. Why is HBM faster than DDR5?

HBM is faster because it is physically designed for **bandwidth**, not upgradeability
or long-distance signaling. DDR5 DIMMs sit away from the CPU on the motherboard and
connect through a relatively narrow bus. HBM stacks sit on the same package as the
GPU, connected through many short wires called TSVs/interposer connections.

The simplified bandwidth equation is:

```text
bandwidth = transfers_per_second × bus_width_bytes
```

DDR5 runs at a high clock, but the bus is narrow. HBM may run at a lower clock, but
it has thousands of wires in parallel. That huge bus width is the win.

Example intuition:

```text
DDR5: fewer lanes, faster cars
HBM:  many more lanes, slightly slower cars
```

For GPUs, the many-lane highway wins because thousands of CUDA cores are hungry for
data at the same time.

## 2. What's a cache line and why does size matter?

A cache line, sector, or aligned memory segment is the chunk of memory the hardware
fetches when your program asks for data. Your thread may ask for 4 bytes, but the
memory system usually fetches a larger aligned chunk around that address.

For the beginner CUDA mental model:

```text
1 warp = 32 threads
1 FP32 value = 4 bytes
32 threads × 4 bytes = 128 bytes
```

So if a warp reads 32 consecutive FP32 values, those 32 useful values fit naturally
into a 128-byte aligned chunk. That is efficient because almost every fetched byte
is used.

If the warp reads scattered addresses, the GPU may fetch many chunks while using
only a few bytes from each chunk. That wastes bandwidth. Cache-line size matters
because it controls the granularity of memory traffic: you pay for chunks, not
individual floats.

## 3. What's the difference between coalesced and uncoalesced access?

Coalesced access means neighboring threads in a warp access neighboring memory
addresses. The hardware can combine those requests into a small number of memory
transactions.

Example:

```c
float x = A[i];  // thread 0 reads A[0], thread 1 reads A[1], ...
```

For FP32, warp 0 uses:

```text
A[0]..A[31] = bytes 0..127
```

That is ideal.

Uncoalesced access means neighboring threads access far-apart addresses.

Example:

```c
float x = A[i * 32];
```

Warp 0 uses:

```text
thread 0 → A[0]    byte 0
thread 1 → A[32]   byte 128
thread 2 → A[64]   byte 256
...
```

Now each thread lands in a different aligned chunk. The warp still does useful
work, but it wastes most of the memory bandwidth it requested.

## 4. Why is `i = blockIdx.x * blockDim.x + threadIdx.x; A[i]` coalesced?

For a simple 1D contiguous array, this formula maps consecutive thread IDs to
consecutive element indices.

Inside a warp:

```text
threadIdx.x = 0  → i = base + 0
threadIdx.x = 1  → i = base + 1
threadIdx.x = 2  → i = base + 2
...
threadIdx.x = 31 → i = base + 31
```

If the code reads:

```c
A[i]
```

then the warp reads neighboring memory addresses. That is why this indexing pattern
is the canonical CUDA pattern for vector operations.

Important caveat: coalescing depends on the **actual addresses used by the warp**.
Other indexing formulas can also be coalesced, and this formula can become part of
an uncoalesced access if you later do something like `A[i * stride]`.

## 5. Why is LLM inference memory-bound?

In small-batch decode, each generated token requires running through all model
layers. For a dense model, that means the GPU streams a large amount of weight data
from HBM.

For LLaMA-7B FP16:

```text
~6.7B parameters × 2 bytes ≈ 13.4 GB of weights
```

For batch-1 decode, those weights are used for only one new token. That gives low
arithmetic intensity:

```text
many bytes read per token
not enough math per byte to saturate Tensor Cores
```

Prefill and large-batch decode reuse weights across many tokens/sequences, so they
do more math per byte loaded. Batch-1 decode is the clearest memory-bound case.

## 6. How do you measure bandwidth efficiency?

The basic formula for effective bandwidth is:

```text
effective_bandwidth = useful_bytes_moved / time_seconds
```

For a copy kernel:

```c
out[i] = in[i];
```

each element does:

```text
1 read + 1 write = 2 × bytes
```

So:

```c
float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;
```

This reports useful GB/s. To understand efficiency, compare it to theoretical peak:

```text
percent_of_peak = effective_bandwidth / peak_bandwidth × 100
```

For uncoalesced kernels, useful bandwidth may be low even when actual memory traffic
is high, because many fetched bytes are wasted.

## 7. Why does quantization speed up inference?

Quantization stores each weight using fewer bits.

Approximate LLaMA-7B sizes:

```text
FP32: 6.7B × 4 bytes   ≈ 26.8 GB
FP16: 6.7B × 2 bytes   ≈ 13.4 GB
INT8: 6.7B × 1 byte    ≈ 6.7 GB
INT4: 6.7B × 0.5 bytes ≈ 3.35 GB
```

In memory-bound decode, throughput is roughly limited by:

```text
tokens_per_second ≈ memory_bandwidth / bytes_read_per_token
```

If INT4 makes the model about 4× smaller than FP16, the GPU reads about 4× fewer
weight bytes per token. That can greatly increase decode throughput.

Real systems also pay for dequantization, KV-cache traffic, attention work, sampling,
and scheduler overhead, so the speedup is not always exactly 4×. But the core reason
quantization helps is simple:

> fewer bytes per weight → fewer HBM bytes per token → higher memory-bound throughput.

**Tomorrow (Day 2): Shared Memory** — the on-chip SRAM that sidesteps HBM entirely.
Where Flash Attention's magic lives.

---

*Status: ✅ COMPLETED*
*Date completed: 2026-04-25*
