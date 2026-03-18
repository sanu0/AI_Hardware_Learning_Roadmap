# Week 1, Day 2: NVIDIA GPU Physical Architecture
## Deep Dive Into the SM — Where AI Actually Runs

Yesterday you learned the BIG PICTURE: CPU vs GPU philosophy, SIMT, memory hierarchy,
and why LLMs are memory-bandwidth-bound. Today we go INSIDE the GPU chip — you'll
understand every component of a Streaming Multiprocessor, how warps actually execute,
and what makes each NVIDIA GPU generation better for AI.

By the end of this session, you'll:
- Know every component inside an SM and what it does
- Understand how warp schedulers keep the GPU busy
- Know the difference between CUDA Cores and Tensor Cores at the physical level
- Understand registers, shared memory, and the L1/L2 cache split
- Know what changed in each NVIDIA GPU generation for AI
- Write code to query YOUR GPU's capabilities

---

# PART 0: QUICK RECALL FROM DAY 1

Before diving in, make sure you remember these (30 seconds):

```
✓ GPU = thousands of simple cores, optimized for throughput
✓ SIMT = 32 threads (warp) execute same instruction on different data
✓ Thread → Warp (32) → Block (up to 1024) → Grid
✓ Memory hierarchy: Registers → Shared Memory → L1 → L2 → HBM (Global)
✓ LLM inference is memory-bandwidth-bound (reading weights is the bottleneck)
✓ Tensor Cores do matrix multiply ~10-16x faster than regular cores
✓ An SM is one "factory floor" in the GPU "factory"
```

Today we zoom into that "factory floor" — the SM.

---

# PART 1: THE STREAMING MULTIPROCESSOR (SM) — COMPLETE ANATOMY

## 1.1 What Is an SM?

The SM (Streaming Multiprocessor) is the **fundamental compute unit** of an NVIDIA GPU.
Everything happens inside SMs. When you launch a CUDA kernel (a GPU function),
the GPU distributes work to SMs.

Think of it this way:
```
A GPU is like a COMPANY:
├── The CEO (GigaThread Engine): decides which work goes where
├── Factory Floor 0 (SM 0): does actual work
├── Factory Floor 1 (SM 1): does actual work
├── Factory Floor 2 (SM 2): does actual work
├── ...
├── Factory Floor 131 (SM 131): does actual work  (H100 has 132 SMs)
├── Shared Warehouse (L2 Cache): 50 MB, everyone can access
└── External Supply Chain (HBM): 80 GB, bulk storage

Each factory floor (SM) is SELF-CONTAINED:
  - Has its own workers (CUDA Cores, Tensor Cores)
  - Has its own fast storage (Registers, Shared Memory)
  - Has its own manager (Warp Schedulers)
  - Can work independently of other floors
```

**Different GPUs have different numbers of SMs:**
```
GPU              SMs    CUDA Cores    Tensor Cores    HBM
─────────────────────────────────────────────────────────────
RTX 3090 (home)   82     10,496        328          24 GB GDDR6X
RTX 4090 (home)  128     16,384        512          24 GB GDDR6X
A100 (datacenter) 108      6,912        432          80 GB HBM2e
H100 (datacenter) 132     16,896        528          80 GB HBM3
B200 (datacenter) 192     ?,???         ???         192 GB HBM3e

More SMs = more parallel work = faster AI
```

## 1.2 Inside One SM — The Full Breakdown

Let's look at ONE SM from the H100 (Hopper architecture). I'll explain every
component, what it does, and why it matters for LLMs.

```
┌═══════════════════════════════════════════════════════════════════┐
║                        ONE SM (H100 Hopper)                      ║
║                                                                   ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │              WARP SCHEDULERS (×4)                            │  ║
║  │                                                              │  ║
║  │  Scheduler 0    Scheduler 1    Scheduler 2    Scheduler 3   │  ║
║  │  manages        manages        manages        manages       │  ║
║  │  warps 0,4,8..  warps 1,5,9..  warps 2,6,10.. warps 3,7,11.│  ║
║  │                                                              │  ║
║  │  Each scheduler picks ONE ready warp per cycle and           │  ║
║  │  issues ONE instruction from it to the execution units.      │  ║
║  │  4 schedulers = 4 warps can make progress per clock cycle.   │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                           │                                       ║
║              ┌────────────┼────────────┐                          ║
║              ↓            ↓            ↓                          ║
║  ┌──────────────────────────────────────────────────────┐         ║
║  │           EXECUTION UNITS (the workers)               │         ║
║  │                                                        │         ║
║  │  ┌──────────────────┐  ┌──────────────────┐           │         ║
║  │  │  FP32 CUDA Cores │  │  FP32 CUDA Cores │           │         ║
║  │  │   64 units       │  │   64 units       │           │         ║
║  │  │   (partition 0)  │  │   (partition 1)  │           │         ║
║  │  └──────────────────┘  └──────────────────┘           │         ║
║  │        = 128 FP32 Cores Total                          │         ║
║  │        Each does: one multiply OR one add per cycle    │         ║
║  │        For: general math, non-matrix operations        │         ║
║  │                                                        │         ║
║  │  ┌──────────────────┐  ┌──────────────────┐           │         ║
║  │  │  INT32 Cores     │  │  FP64 Cores      │           │         ║
║  │  │   64 units       │  │   2 units        │           │         ║
║  │  │  (integer math)  │  │  (double prec.)  │           │         ║
║  │  └──────────────────┘  └──────────────────┘           │         ║
║  │  INT32: for index calculations, address computation    │         ║
║  │  FP64: scientific computing (not used in AI much)      │         ║
║  │                                                        │         ║
║  │  ┌────────────────────────────────────────────┐        │         ║
║  │  │        TENSOR CORES (×4)                    │        │         ║
║  │  │                                              │        │         ║
║  │  │  TC 0    TC 1    TC 2    TC 3               │        │         ║
║  │  │                                              │        │         ║
║  │  │  Each Tensor Core computes:                  │        │         ║
║  │  │  D[16×16] = A[16×16] × B[16×16] + C[16×16] │        │         ║
║  │  │  in ONE clock cycle!                         │        │         ║
║  │  │                                              │        │         ║
║  │  │  Supported formats:                          │        │         ║
║  │  │  • FP16:  for inference                      │        │         ║
║  │  │  • BF16:  for training                       │        │         ║
║  │  │  • TF32:  automatic FP32 "upgrade"           │        │         ║
║  │  │  • FP8:   fastest, Hopper+ only              │        │         ║
║  │  │  • INT8:  quantized inference                │        │         ║
║  │  │                                              │        │         ║
║  │  │  THIS IS WHERE 90%+ OF LLM COMPUTE HAPPENS  │        │         ║
║  │  └────────────────────────────────────────────┘        │         ║
║  │                                                        │         ║
║  │  ┌──────────────────┐  ┌──────────────────┐           │         ║
║  │  │  SFU × 16        │  │  Load/Store × 32 │           │         ║
║  │  │                   │  │                   │           │         ║
║  │  │  Special Function │  │  Memory access    │           │         ║
║  │  │  Units:           │  │  units:           │           │         ║
║  │  │  • sin(x)         │  │  • Load from      │           │         ║
║  │  │  • cos(x)         │  │    global memory  │           │         ║
║  │  │  • exp(x)         │  │  • Store to       │           │         ║
║  │  │  • sqrt(x)        │  │    global memory  │           │         ║
║  │  │  • 1/x            │  │  • Load/store     │           │         ║
║  │  │                   │  │    shared memory  │           │         ║
║  │  │  Used for:        │  │                   │           │         ║
║  │  │  softmax (exp),   │  │  32 units = can   │           │         ║
║  │  │  RoPE (sin/cos),  │  │  serve one full   │           │         ║
║  │  │  GELU activation  │  │  warp (32 threads)│           │         ║
║  │  └──────────────────┘  └──────────────────┘           │         ║
║  └──────────────────────────────────────────────────────┘         ║
║                                                                   ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │              MEMORY (the fast storage)                       │  ║
║  │                                                              │  ║
║  │  ┌───────────────────────────────────────────┐              │  ║
║  │  │  REGISTER FILE                             │              │  ║
║  │  │  65,536 registers × 32-bit = 256 KB        │              │  ║
║  │  │                                             │              │  ║
║  │  │  • FASTEST storage in the entire GPU        │              │  ║
║  │  │  • Access time: 1 clock cycle               │              │  ║
║  │  │  • Each thread gets its own private set      │              │  ║
║  │  │  • If SM runs 2048 threads: each gets        │              │  ║
║  │  │    65536 / 2048 = 32 registers              │              │  ║
║  │  │  • If SM runs 1024 threads: each gets        │              │  ║
║  │  │    65536 / 1024 = 64 registers              │              │  ║
║  │  │  • More registers per thread = can hold      │              │  ║
║  │  │    more values without going to slow memory  │              │  ║
║  │  │                                             │              │  ║
║  │  │  LLM connection: During matrix multiply,     │              │  ║
║  │  │  each thread keeps its partial sum in a      │              │  ║
║  │  │  register. This is the fastest possible      │              │  ║
║  │  │  accumulation.                               │              │  ║
║  │  └───────────────────────────────────────────┘              │  ║
║  │                                                              │  ║
║  │  ┌───────────────────────────────────────────┐              │  ║
║  │  │  SHARED MEMORY / L1 CACHE: 228 KB          │              │  ║
║  │  │                                             │              │  ║
║  │  │  This 228 KB is SPLIT between two uses:     │              │  ║
║  │  │                                             │              │  ║
║  │  │  Option A: 228 KB shared + 0 KB L1          │              │  ║
║  │  │  Option B: 192 KB shared + 36 KB L1         │              │  ║
║  │  │  Option C: 128 KB shared + 100 KB L1        │              │  ║
║  │  │  (Programmer chooses based on workload)      │              │  ║
║  │  │                                             │              │  ║
║  │  │  SHARED MEMORY:                              │              │  ║
║  │  │  • Programmer-controlled cache               │              │  ║
║  │  │  • ~20-30 cycle access                       │              │  ║
║  │  │  • ALL threads in a block can read/write     │              │  ║
║  │  │  • Used for cooperation between threads      │              │  ║
║  │  │                                             │              │  ║
║  │  │  L1 CACHE:                                   │              │  ║
║  │  │  • Hardware-managed (automatic)              │              │  ║
║  │  │  • Caches data from global memory            │              │  ║
║  │  │  • You don't control it directly             │              │  ║
║  │  │                                             │              │  ║
║  │  │  LLM connection: Flash Attention keeps the   │              │  ║
║  │  │  Q, K, V TILES in shared memory to avoid     │              │  ║
║  │  │  reading from slow HBM. This is why it's     │              │  ║
║  │  │  called "IO-aware" — it minimizes HBM IO.    │              │  ║
║  │  └───────────────────────────────────────────┘              │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════╝
```

## 1.3 Component-by-Component: What Does This Do for My LLM?

Let's connect EVERY SM component to actual LLM operations:

```
SM Component         What It Does                  LLM Operation That Uses It
──────────────────────────────────────────────────────────────────────────────
FP32 CUDA Cores      Regular math (add, multiply)   Bias addition, residual add,
(128 per SM)         one operation per cycle         RMSNorm computation,
                                                     element-wise operations

Tensor Cores         Matrix multiply-accumulate      Q/K/V projections,
(4 per SM)           16×16 tiles in one cycle        attention matmul,
                     FP16/BF16/FP8                   FFN layers — the BIG stuff
                                                     ~90% of total LLM compute

INT32 Cores          Integer arithmetic              Memory address calculation,
(64 per SM)                                          array index computation
                                                     (runs in parallel with FP32!)

FP64 Cores           Double-precision math           Almost NEVER used in LLMs
(2 per SM)           Scientific computing            (AI doesn't need 64-bit precision)

SFU                  sin, cos, exp, sqrt, 1/x        softmax (needs exp()),
(16 per SM)          transcendental functions         RoPE (needs sin/cos),
                                                     GELU/SiLU activation functions

Load/Store Units     Read/write memory               Loading weight matrices,
(32 per SM)          global, shared, local            storing activations,
                                                     reading KV-cache

Registers            Per-thread fast storage          Partial sums during matmul,
(256 KB per SM)      1-cycle access                   intermediate values,
                                                     thread-local variables

Shared Memory        Per-block fast scratchpad        Tiled matrix multiply,
(up to 228 KB)       20-30 cycle access               Flash Attention tiles,
                     shared between threads            inter-thread communication

Warp Schedulers      Pick warps, issue instructions   Keeping all units busy,
(4 per SM)           manage thread execution           hiding memory latency
```

---

# PART 2: WARP SCHEDULERS — The Brain of the SM

## 2.1 What Is a Warp Scheduler?

Remember: a **warp** = 32 threads executing the same instruction. The warp scheduler's
job is to decide WHICH warp gets to execute next.

Each SM has 4 warp schedulers, and each can issue one instruction per clock cycle.
So in ONE clock cycle, 4 different warps can each advance by one instruction.

```
Clock Cycle 1:
  Scheduler 0 → issues "multiply" to Warp 0  (using Tensor Cores)
  Scheduler 1 → issues "load" to Warp 5      (using Load/Store units)
  Scheduler 2 → issues "add" to Warp 10      (using FP32 cores)
  Scheduler 3 → issues "exp" to Warp 15      (using SFU)

Clock Cycle 2:
  Scheduler 0 → Warp 0 still busy... pick Warp 4 → issues "multiply"
  Scheduler 1 → issues "multiply" to Warp 5
  Scheduler 2 → issues "store" to Warp 10
  Scheduler 3 → Warp 15 waiting for memory... pick Warp 19 → issues "add"
```

**The key insight:** Warp schedulers are how GPUs hide latency. While one warp
waits for data from HBM (400+ cycles), the scheduler issues instructions from
OTHER warps that are ready to go. No time wasted.

## 2.2 Occupancy: How Full Is the SM?

**Occupancy** = (active warps on SM) / (maximum warps SM can hold)

An H100 SM can hold up to **64 warps** (= 2048 threads) simultaneously.
But how many actually FIT depends on resource usage:

```
WHAT LIMITS OCCUPANCY:

1. Registers per thread:
   SM has 65,536 registers total.
   If each thread uses 32 registers:
     65,536 / 32 = 2,048 threads = 64 warps → 100% occupancy ✓
   If each thread uses 128 registers:
     65,536 / 128 = 512 threads = 16 warps → 25% occupancy ✗
   
   Complex kernels need more registers → fewer warps fit → lower occupancy

2. Shared memory per block:
   SM has 228 KB of shared memory.
   If each block uses 114 KB:
     228 / 114 = 2 blocks can fit on SM
   If each block uses 228 KB:
     Only 1 block fits

3. Threads per block:
   Max 1024 threads per block.
   If you launch blocks of 256 threads:
     2048 / 256 = 8 blocks could potentially fit
   If you launch blocks of 1024:
     2048 / 1024 = 2 blocks

THE TRADEOFF:
  High occupancy (many warps): 
    → More warps to switch between when waiting for memory
    → Better latency hiding
    → But each thread has fewer registers (might spill to slow memory)
  
  Low occupancy (few warps):
    → Each thread has more registers (faster)
    → But fewer warps means less latency hiding
    → GPU might sit idle when warps are waiting for memory
```

**LLM connection:** When running a Transformer layer:
- The matrix multiply kernels (Q, K, V, FFN) are usually compute-heavy
  and run at moderate occupancy (50-75%) because Tensor Cores are busy enough
- The element-wise kernels (RMSNorm, activation functions, residual add)
  are memory-bound and NEED high occupancy to hide memory latency
- This is why kernel fusion helps — instead of separate low-occupancy
  kernels for norm + activation + residual, fuse them into ONE kernel

## 2.3 How a Warp Actually Executes — Step by Step

Let's trace what happens when one warp (32 threads) executes `c[i] = a[i] + b[i]`:

```
SETUP: Block 0 assigned to SM 3. Contains Warp 0 (threads 0-31).

Cycle 1: Scheduler selects Warp 0, issues instruction: "LOAD a[i]"
         → 32 Load/Store units each request one element of a[]
         → These addresses are consecutive: a[0], a[1], ..., a[31]
         → Memory controller sees consecutive addresses → COALESCED ACCESS
         → ONE 128-byte transaction from HBM (not 32 separate reads!)
         → But this takes ~400 cycles to come back from HBM...

Cycle 2-400: Warp 0 is WAITING for memory. Scheduler gives cycles to other warps.
         → Warp 4 computes something
         → Warp 8 loads something
         → Warp 12 stores something
         → ... the SM is NOT idle, it's working on other warps

Cycle 401: Data for Warp 0 arrives! a[0] through a[31] are now in registers.
         → Scheduler issues: "LOAD b[i]" for Warp 0
         → Again, consecutive addresses → coalesced → one 128-byte transaction

Cycle 402-800: Warp 0 waiting again. Other warps execute.

Cycle 801: b[0] through b[31] arrive in registers.
         → Scheduler issues: "ADD" instruction
         → 32 FP32 CUDA cores simultaneously compute:
           Thread 0:  c[0] = a[0] + b[0]
           Thread 1:  c[1] = a[1] + b[1]
           ...
           Thread 31: c[31] = a[31] + b[31]
         → This takes 1 cycle. ALL 32 additions happen simultaneously.

Cycle 802: Scheduler issues: "STORE c[i]"
         → 32 threads write their results
         → Coalesced write → one 128-byte transaction to HBM

TOTAL: ~800 cycles for this warp, but the SM was busy with OTHER warps
during the ~798 cycles of memory waiting. This is latency hiding.
```

**Key takeaway:** The actual COMPUTE (the add) took 1 cycle. The memory access
took ~800 cycles. This is why memory bandwidth is the bottleneck. The GPU's
strategy is to have SO many warps that there's always something to compute
while other warps wait.

---

# PART 3: TENSOR CORES — The AI Accelerator In Detail

## 3.1 Regular CUDA Core vs Tensor Core

```
REGULAR FP32 CUDA CORE (one per cycle):
  Computes: a × b + c = d
  That's 2 FLOPs (one multiply, one add)
  
  To compute a 16×16 matrix multiply:
  Need: 16 × 16 × 16 = 4,096 multiply-adds
  Time: 4,096 cycles (one at a time)

TENSOR CORE (one per cycle):
  Computes: D[16×16] = A[16×8] × B[8×16] + C[16×16]
  That's: 16 × 16 × 8 = 2,048 multiply-adds = 4,096 FLOPs
  Time: 1 cycle

  SPEEDUP: 4,096 / 1 = 4,096x for this operation!
  
  (In practice, ~8-16x overall speedup because of data movement overhead)
```

## 3.2 How Tensor Cores Work Physically

A Tensor Core is a specialized circuit that does a small matrix multiply-and-accumulate
(MMA) in a single clock cycle. Here's what happens:

```
TENSOR CORE OPERATION (4th gen, Hopper):

Input A: 16×8 matrix (128 elements, in FP16/BF16/FP8)
Input B: 8×16 matrix (128 elements, in FP16/BF16/FP8)
Input C: 16×16 matrix (256 elements, accumulator in FP32)

Operation:
  D = A × B + C

  The Tensor Core has a grid of multiply-add circuits:
  
  ┌────────────────────────────────────────────┐
  │  A[0,0]×B[0,0] + A[0,1]×B[1,0] + ...      │ → D[0,0]
  │  A[0,0]×B[0,1] + A[0,1]×B[1,1] + ...      │ → D[0,1]
  │  ...                                        │
  │  A[15,0]×B[0,15] + A[15,1]×B[1,15] + ...  │ → D[15,15]
  │                                              │
  │  ALL 2,048 multiply-adds happen              │
  │  SIMULTANEOUSLY in hardware.                 │
  │  Not sequentially — in PARALLEL circuits.    │
  └────────────────────────────────────────────┘

Output D: 16×16 matrix (256 elements, in FP32)

Total: 2,048 multiply-adds = 4,096 FLOPs in 1 cycle.
```

## 3.3 Tensor Core Supported Precisions (and what each means for LLMs)

```
Format      Input     Accumulate    TFLOPS (H100)    When to use
──────────────────────────────────────────────────────────────────
FP16        16-bit    FP32          990              General inference
BF16        16-bit    FP32          990              Training (safe range)
TF32        19-bit*   FP32          495              Auto FP32 "upgrade"
FP8 E4M3    8-bit     FP32          1,979            Fast inference (Hopper+)
FP8 E5M2    8-bit     FP32          1,979            Training with FP8
INT8        8-bit     INT32         1,979            Quantized inference
FP64        64-bit    FP64          67               Scientific (not AI)

*TF32 = "TensorFloat-32": uses FP32 inputs but rounds mantissa to 10 bits
  internally. PyTorch uses this by default when you do FP32 matmul on Ampere+.
  You get Tensor Core speed WITHOUT changing your code from FP32!

PERFORMANCE PROGRESSION for H100:
  FP32 (CUDA cores only):    67 TFLOPS
  TF32 (Tensor Cores):      495 TFLOPS  ← 7.4x faster, automatic!
  FP16/BF16:                990 TFLOPS  ← 14.8x faster
  FP8:                    1,979 TFLOPS  ← 29.5x faster
  
  Going from FP32 to FP8 = 29.5x more compute throughput!
```

**LLM connection:**
```
TRAINING typically uses BF16:
  → Tensors stored in BF16 (half memory vs FP32)
  → Tensor Cores compute at 990 TFLOPS
  → Accumulation in FP32 (maintains precision for gradient updates)
  → This is what PyTorch AMP (Automatic Mixed Precision) does

INFERENCE typically uses FP16, INT8, or FP8:
  → FP16: simple, good quality, 990 TFLOPS
  → INT8: quantized, 2x less memory, 1,979 TFLOPS
  → FP8: best speed, requires calibration, 1,979 TFLOPS
  → Transformer Engine (Hopper) automatically manages FP8 scaling

YOUR LAPTOP (if RTX 30/40 series):
  → Also has Tensor Cores! Just fewer of them.
  → RTX 4090: 330 TFLOPS FP16 (vs H100's 990)
  → Still 10x+ faster than CPU for AI
```

## 3.4 How Tensor Cores Map to Matrix Multiplication in LLMs

When PyTorch does `output = input @ weight` for a Transformer layer:

```
input shape:  [batch=32, seq=1, hidden=4096]
weight shape: [hidden=4096, hidden=4096]
output shape: [batch=32, seq=1, hidden=4096]

This is a matrix multiply: [32 × 4096] × [4096 × 4096]

PyTorch calls cuBLAS → cuBLAS calls Tensor Cores:

1. cuBLAS TILES the computation:
   Break the big matrix into 16×16 tiles that fit Tensor Cores.
   
   Output is [32 × 4096] = 32/16 × 4096/16 = 2 × 256 = 512 tiles.
   
   Each tile needs: inner dimension / 8 = 4096/8 = 512 Tensor Core calls.
   
   Total Tensor Core calls: 512 tiles × 512 = 262,144 MMA operations.

2. cuBLAS DISTRIBUTES tiles across SMs:
   H100 has 132 SMs, each with 4 Tensor Cores.
   Total Tensor Cores: 528
   
   262,144 operations / 528 Tensor Cores = ~496 cycles per Tensor Core
   At 1.83 GHz: 496 / 1.83G = 0.27 microseconds
   
   But must also load data from HBM — this is what actually takes time.

3. Each SM processes its assigned tiles:
   - Load tile of input from HBM → shared memory
   - Load tile of weight from HBM → shared memory
   - Feed tiles to Tensor Core: D = A × B + C
   - Accumulate partial results in registers
   - When all inner-dimension tiles are done, write result to HBM
```

---

# PART 4: THE REGISTER FILE — Fastest Storage

## 4.1 Why Registers Matter

Registers are the FASTEST memory in the entire system. Each SM has 65,536
32-bit registers = 256 KB. This is divided among all active threads.

```
WHY THIS MATTERS — A REAL EXAMPLE:

Suppose a thread computes a dot product for one element of a matrix multiply:

  sum = 0
  for k in range(4096):
      sum += A[row][k] * B[k][col]

Where does 'sum' live?
  → In a REGISTER. The fastest possible location.
  → The thread updates it 4096 times without ANY memory access.
  → Only at the end does it write 'sum' to memory.

If 'sum' had to live in shared memory: 20-30 cycles per access × 4096 = 80K-120K cycles
If 'sum' had to live in HBM: 400+ cycles per access × 4096 = 1.6M+ cycles
In a register: 1 cycle per access × 4096 = 4,096 cycles

REGISTERS ARE 100-400x FASTER than the alternatives for this pattern.
```

## 4.2 Register Pressure: The Hidden Bottleneck

```
Each thread WANTS as many registers as possible (to keep data close).
But the SM has a FIXED total (65,536 registers).

More threads on SM → fewer registers per thread:
  2048 threads (100% occupancy): 32 registers each
  1024 threads (50% occupancy):  64 registers each
  512 threads (25% occupancy):  128 registers each

If a kernel needs 48 registers per thread:
  65,536 / 48 = 1,365 threads = ~42 warps = 66% occupancy
  
  Some warps CAN'T fit because there aren't enough registers.
  
What if a kernel needs MORE than 255 registers per thread?
  The excess SPILLS to "local memory" which is actually... HBM (slow!)
  Register spill = terrible performance.
  This is called "register pressure."
  
LLM connection:
  Kernel writers (cuBLAS, Flash Attention) carefully tune register usage.
  Too many registers → low occupancy → poor latency hiding.
  Too few registers → data spills to slow memory.
  Finding the sweet spot is a key optimization skill.
```

---

# PART 5: L2 CACHE AND THE FULL MEMORY PATH

## 5.1 The Complete Memory Journey

When an SM needs data from HBM, it doesn't go directly. There are caches:

```
Thread requests data (e.g., a weight value)
            │
            ↓
┌───────────────────────┐
│ Is it in REGISTERS?    │ → YES: return immediately (1 cycle)
└───────────┬───────────┘
            │ NO
            ↓
┌───────────────────────┐
│ Is it in L1 CACHE?     │ → YES: return (~30 cycles)
│ (inside SM, per-SM)    │
└───────────┬───────────┘
            │ NO (L1 miss)
            ↓
┌───────────────────────┐
│ Is it in L2 CACHE?     │ → YES: return (~200 cycles)
│ (shared, 50 MB on H100)│
└───────────┬───────────┘
            │ NO (L2 miss)
            ↓
┌───────────────────────┐
│ Go to HBM              │ → SLOW: ~400-600 cycles
│ (main GPU memory)      │   But HIGH BANDWIDTH: 3,350 GB/s
└───────────────────────┘

LLM connection:
  Model weights are typically too large for L2 cache (14 GB >> 50 MB).
  So weight reads almost always go to HBM = slow.
  
  BUT activations (intermediate results) are smaller and get reused:
    A 32-token batch × 4096 hidden dim × 2 bytes = 256 KB
    This fits in L2 cache! If the next layer needs it, it's already cached.
    
  KV-cache entries for RECENT tokens may also fit in L2.
  This is one reason larger L2 caches help LLM inference.
```

## 5.2 Cache Line and Memory Transactions

```
When the GPU reads from HBM, it doesn't read single bytes.
It reads in CHUNKS called cache lines:

  L2 cache line = 32 bytes (can hold 16 FP16 values)
  L1 cache line = 128 bytes (can hold 64 FP16 values)

So when Thread 0 reads weight[0] (just 2 bytes), the GPU actually
fetches 128 bytes containing weight[0] through weight[63].

If threads in the same warp need weight[0] through weight[31]:
  → They all get served by that ONE 128-byte transaction
  → This is COALESCED access — efficient!

If threads in the same warp need weight[0], weight[1000], weight[2000]...:
  → Each thread needs a DIFFERENT cache line
  → Could require up to 32 separate 128-byte transactions
  → This is UNCOALESCED access — 32x slower!

LLM connection:
  Weight matrices are stored ROW by ROW in memory.
  When doing matrix multiply, threads in a warp read CONSECUTIVE
  elements of the same row → coalesced → efficient.
  This is why matrix layout (row-major vs column-major) matters
  and why cuBLAS is column-major (optimized for its access pattern).
```

---

# PART 6: NVIDIA GPU GENERATIONS — What Changed for AI

## 6.1 The Evolution That Matters

```
KEPLER (2012) — Where it started for Deep Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • AlexNet (2012) ran on 2 GTX 580s (Fermi) but Kepler made it practical
  • No Tensor Cores yet — everything on FP32 CUDA cores
  • cuDNN library released during Kepler era
  • LLM impact: None yet (LLMs didn't exist)

PASCAL (2016) — P100: First "AI GPU"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • First NVLink (GPU-to-GPU interconnect, 160 GB/s)
  • First HBM2 memory (732 GB/s bandwidth)
  • FP16 compute support (2x throughput vs FP32)
  • LLM impact: Early language models trained on P100 clusters

VOLTA (2017) — V100: THE BREAKTHROUGH ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • FIRST TENSOR CORES ever made
  • 125 TFLOPS FP16 Tensor Core (vs 15 TFLOPS FP32)
  • Mixed-precision training became practical
  • 900 GB/s HBM2 bandwidth
  • LLM impact: BERT, GPT-1 trained on V100s
  • This is when GPU AI training EXPLODED

TURING (2018) — RTX 20 Series (Consumer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 2nd gen Tensor Cores
  • Added INT8 and INT4 Tensor Core support (inference!)
  • RT Cores for ray tracing (not relevant for AI)
  • LLM impact: Brought Tensor Cores to consumer GPUs

AMPERE (2020) — A100: The Workhorse of LLM Training ⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 3rd gen Tensor Cores
  • TF32: automatic Tensor Core speedup for FP32 code
  • BF16 support (crucial for stable training)
  • Structural sparsity: 2:4 sparsity for 2x speedup
  • 2 TB/s HBM2e bandwidth, up to 80 GB
  • MIG (Multi-Instance GPU): split one A100 into 7 small GPUs
  • 3rd gen NVLink: 600 GB/s
  • LLM impact: GPT-3 trained on thousands of A100s
    LLaMA, most open-source LLMs trained on A100 clusters
    THE dominant AI training GPU for 3 years

HOPPER (2022) — H100: Built For Transformers ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 4th gen Tensor Cores
  • FP8 SUPPORT — 2x throughput vs FP16!
  • TRANSFORMER ENGINE: hardware that automatically manages
    FP8 ↔ FP16 conversion with per-tensor scaling
    → You get FP8 speed with FP16 quality, automatically
  • 3,350 GB/s HBM3 bandwidth (67% more than A100)
  • 4th gen NVLink: 900 GB/s
  • DPX instructions: dynamic programming acceleration
  • Thread Block Clusters: new hierarchy level for multi-SM cooperation
  • LLM impact: GPT-4, LLaMA-2, Mixtral trained on H100 clusters
    2-3x faster than A100 for Transformer training

BLACKWELL (2024) — B200: The Next Frontier ⭐⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 5th gen Tensor Cores
  • FP4 support — 2x throughput vs FP8!
  • 2nd gen Transformer Engine
  • 8 TB/s HBM3e bandwidth (2.4x H100!)
  • Up to 192 GB HBM3e
  • 5th gen NVLink: 1,800 GB/s (2x H100)
  • Two GPU dies on one package (connected by 10 TB/s chip-to-chip)
  • NVLink Switch for up to 576 GPUs in one fabric
  • LLM impact: 2.5x H100 for LLM training, 5x for inference
    Enables training of multi-trillion parameter models
```

## 6.2 The Pattern: What Gets Better Each Generation

```
EVERY generation improves these for AI:

1. MORE Tensor Cores → more TFLOPS
   V100(125) → A100(312) → H100(990) → B200(~2,500)  FP16 TFLOPS

2. LOWER precision Tensor Cores → even more TFLOPS
   FP16 → TF32 → BF16 → FP8 → FP4

3. MORE memory bandwidth → faster weight reading
   900 → 2,000 → 3,350 → 8,000 GB/s

4. MORE total memory → bigger models fit
   32 → 80 → 80 → 192 GB

5. FASTER GPU-to-GPU links → better multi-GPU training
   NVLink: 300 → 600 → 900 → 1,800 GB/s

Each generation is specifically targeting the LLM bottlenecks:
  → More compute for training (Tensor Cores)
  → More bandwidth for inference (HBM generations)
  → More memory for bigger models (HBM capacity)
  → Faster interconnect for distributed training (NVLink)
```

---

# PART 7: NGC CATALOG — Your Model and Container Registry

## 7.1 What Is NGC?

NGC (NVIDIA GPU Cloud) is NVIDIA's registry of GPU-optimized containers,
pre-trained models, and tools. Think of it as "Docker Hub but specifically
for GPU/AI stuff, maintained by NVIDIA."

**URL:** https://catalog.ngc.nvidia.com

```
What you'll find on NGC:
├── Containers
│   ├── PyTorch (optimized for NVIDIA GPUs with cuDNN, NCCL pre-configured)
│   ├── TensorFlow
│   ├── TensorRT-LLM
│   ├── NeMo Framework
│   ├── Triton Inference Server
│   └── RAPIDS
├── Models
│   ├── Pre-trained LLMs (LLaMA, Nemotron, etc.)
│   ├── Embedding models (NV-EmbedQA)
│   ├── Speech models (Riva/Parakeet)
│   └── Vision models
├── Helm Charts
│   └── Kubernetes deployments for AI services
└── Resources
    ├── Jupyter notebooks
    ├── Model scripts
    └── Documentation
```

**Why it matters for you:** When you start training models (Week 6+), you'll use
NGC containers instead of installing CUDA/cuDNN/PyTorch manually. One command:
`docker pull nvcr.io/nvidia/pytorch:24.01-py3` gives you a perfectly optimized
PyTorch environment.

## 7.2 build.nvidia.com — Try Models Instantly

**URL:** https://build.nvidia.com

This is NVIDIA's API playground. You can:
- Try any NVIDIA-hosted LLM (LLaMA, Mixtral, etc.) via API
- Test embedding models, rerankers
- Get an API key and use the same models from your code
- Free tier available

```python
# Example: Call an LLM on build.nvidia.com
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-YOUR-KEY-HERE"  # Get free key from build.nvidia.com
)

response = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "What is a GPU?"}],
    max_tokens=200
)
print(response.choices[0].message.content)
```

You can try this TODAY — no GPU needed, just an API key.

---

# PART 8: SELF-TEST (12 Questions)

**Hardware:**
1. **How many SMs does the H100 have?**
   → 132 SMs, each with 128 FP32 cores and 4 Tensor Cores.

2. **What's the difference between FP32 CUDA Cores and Tensor Cores?**
   → CUDA Core: 1 multiply-add per cycle (2 FLOPs).
   Tensor Core: 16×8 × 8×16 matrix multiply in 1 cycle (4,096 FLOPs). ~2000x more per cycle.

3. **What are SFUs and when are they used in LLMs?**
   → Special Function Units compute sin, cos, exp, sqrt. Used for softmax (exp),
   RoPE (sin/cos), and activation functions (GELU uses exp).

4. **What is occupancy and why does it matter?**
   → Ratio of active warps to max warps on an SM. Higher occupancy = more warps
   to switch between = better latency hiding. Limited by register usage,
   shared memory usage, and block size.

5. **Why does register spill hurt performance?**
   → Spilled registers go to "local memory" which is actually HBM (400+ cycle access).
   What was a 1-cycle register read becomes a 400+ cycle memory read.

6. **What's the difference between coalesced and uncoalesced memory access?**
   → Coalesced: threads in a warp access consecutive addresses → one transaction.
   Uncoalesced: threads access scattered addresses → many transactions → much slower.

**LLM connections:**
7. **Why do LLMs use BF16 for training instead of FP16?**
   → BF16 has the same exponent range as FP32 (handles very large and very small numbers)
   so it doesn't overflow during training. FP16 has limited range and can overflow.

8. **What is TF32 and why is it free performance?**
   → TF32 is an internal format: takes FP32 inputs, rounds to 10-bit mantissa,
   runs on Tensor Cores. PyTorch does this automatically on Ampere+. You get 7x speedup
   without changing a single line of code.

9. **Why does Flash Attention use shared memory?**
   → It tiles the Q, K, V matrices into chunks that fit in shared memory (228 KB, ~20 cycles)
   instead of reading/writing the full attention matrix from HBM (400+ cycles).
   This reduces HBM traffic by O(N) where N is sequence length.

10. **What made the V100 a breakthrough for AI?**
    → First GPU with Tensor Cores. Mixed-precision training became practical.
    125 TFLOPS FP16 vs 15 TFLOPS FP32 on the same chip = 8x free speedup for AI.

11. **What did H100 (Hopper) add specifically for Transformers?**
    → FP8 Tensor Cores (2x throughput vs FP16) and the Transformer Engine
    (automatically manages FP8 precision with per-tensor scaling).

12. **How much faster is Blackwell (B200) vs Hopper (H100) for LLM inference?**
    → ~5x faster: FP4 Tensor Cores (2x compute) + 8 TB/s HBM3e (2.4x bandwidth)
    + dual-die design.

---

# PART 9: CODING EXERCISES (Do on Google Colab)

## Exercise 1: Query Your GPU Properties

This is today's main code exercise. Run this on Google Colab (free GPU runtime)
or any machine with a GPU.

```python
import torch

if not torch.cuda.is_available():
    print("No GPU available! Enable GPU in Colab: Runtime → Change runtime type → GPU")
else:
    # Get device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    print("=" * 60)
    print(f"GPU: {props.name}")
    print("=" * 60)
    
    # Compute capability (determines which features are available)
    print(f"\n--- Architecture ---")
    print(f"Compute Capability: {props.major}.{props.minor}")
    capability_names = {
        (7, 0): "Volta (V100) — 1st gen Tensor Cores",
        (7, 5): "Turing (RTX 20xx) — 2nd gen Tensor Cores",
        (8, 0): "Ampere (A100) — 3rd gen Tensor Cores, TF32, BF16",
        (8, 6): "Ampere (RTX 30xx) — Consumer Ampere",
        (8, 9): "Ada Lovelace (RTX 40xx) — Consumer Ada",
        (9, 0): "Hopper (H100) — 4th gen Tensor Cores, FP8",
    }
    arch = capability_names.get((props.major, props.minor), "Unknown")
    print(f"Architecture: {arch}")
    
    # SM count
    print(f"\n--- Streaming Multiprocessors ---")
    print(f"Number of SMs: {props.multi_processor_count}")
    
    # Calculate CUDA cores (approximate, varies by architecture)
    cuda_cores_per_sm = {7: 64, 8: 128, 9: 128}
    cores_per_sm = cuda_cores_per_sm.get(props.major, 64)
    total_cores = props.multi_processor_count * cores_per_sm
    print(f"CUDA Cores per SM: ~{cores_per_sm}")
    print(f"Total CUDA Cores: ~{total_cores}")
    print(f"Tensor Cores per SM: ~4")
    print(f"Total Tensor Cores: ~{props.multi_processor_count * 4}")
    
    # Memory
    print(f"\n--- Memory ---")
    total_mem = props.total_memory / (1024**3)
    print(f"Total GPU Memory: {total_mem:.1f} GB")
    print(f"Memory Bus Width: {props.memory_bus_width} bits")  # Not available in torch
    
    # What LLMs can fit?
    print(f"\n--- What LLMs fit on this GPU? ---")
    usable_mem = total_mem * 0.9  # Leave 10% for overhead
    print(f"Usable memory (~90%): {usable_mem:.1f} GB")
    
    models = [
        ("LLaMA-7B FP16", 13.4),
        ("LLaMA-7B INT4", 3.5),
        ("LLaMA-13B FP16", 26),
        ("LLaMA-13B INT4", 6.5),
        ("LLaMA-70B FP16", 140),
        ("LLaMA-70B INT4", 35),
        ("Mistral-7B FP16", 14),
        ("Mixtral-8x7B FP16", 93),
    ]
    
    for name, size_gb in models:
        fits = "✅ FITS" if size_gb < usable_mem else "❌ Too large"
        kv_cache_room = max(0, usable_mem - size_gb)
        max_tokens = int(kv_cache_room * 1024 / 0.5)  # ~0.5 MB per token for 7B
        extra = f"(~{max_tokens:,} token context)" if kv_cache_room > 0 else ""
        print(f"  {name:25s} {size_gb:6.1f} GB  {fits}  {extra}")
    
    # Performance estimates
    print(f"\n--- Theoretical Performance ---")
    clock_ghz = props.clock_rate / 1e6  # Convert kHz to GHz
    print(f"GPU Clock: {clock_ghz:.2f} GHz")
    
    fp32_tflops = total_cores * clock_ghz * 2 / 1000  # 2 FLOPs per core per cycle
    print(f"FP32 Peak (CUDA cores): {fp32_tflops:.1f} TFLOPS")
    
    tensor_tflops = fp32_tflops * 8  # Rough estimate: Tensor Cores ~8x FP32
    print(f"FP16 Tensor Core (est): ~{tensor_tflops:.0f} TFLOPS")
    
    # Memory bandwidth estimate
    print(f"\n--- LLM Inference Speed Estimate ---")
    # Use a rough bandwidth estimate (actual varies by GPU)
    bandwidths = {
        "T4": 320, "V100": 900, "A100": 2000, "A10": 600,
        "H100": 3350, "RTX 3090": 936, "RTX 4090": 1008,
    }
    bw = None
    for name, val in bandwidths.items():
        if name.lower() in props.name.lower():
            bw = val
            break
    
    if bw:
        print(f"Memory Bandwidth: ~{bw} GB/s")
        for model_name, size_gb in [("LLaMA-7B FP16", 13.4), ("LLaMA-7B INT4", 3.5)]:
            if size_gb < usable_mem:
                tok_per_sec = bw / size_gb
                print(f"  {model_name}: ~{tok_per_sec:.0f} tokens/sec (theoretical max)")
    else:
        print(f"  (Bandwidth not in database for {props.name})")

    print(f"\n{'=' * 60}")
    print(f"DONE! You now know your GPU's capabilities.")
    print(f"{'=' * 60}")
```

## Exercise 2: Tensor Core vs CUDA Core Speed Test

```python
import torch
import time

if not torch.cuda.is_available():
    print("Need GPU!")
else:
    N = 4096  # LLaMA hidden dimension
    
    # Warmup
    a = torch.randn(N, N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, N, device='cuda', dtype=torch.float32)
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()
    
    # FP32 (uses CUDA cores or TF32 Tensor Cores)
    torch.backends.cuda.matmul.allow_tf32 = False  # Force FP32 CUDA cores
    a32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
    b32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(20):
        _ = a32 @ b32
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / 20
    
    # TF32 (uses Tensor Cores automatically on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(20):
        _ = a32 @ b32
    torch.cuda.synchronize()
    tf32_time = (time.time() - start) / 20
    
    # FP16 (uses Tensor Cores)
    a16 = a32.half()
    b16 = b32.half()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(20):
        _ = a16 @ b16
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / 20
    
    # BF16 (uses Tensor Cores, if supported)
    try:
        abf = a32.bfloat16()
        bbf = b32.bfloat16()
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            _ = abf @ bbf
        torch.cuda.synchronize()
        bf16_time = (time.time() - start) / 20
    except:
        bf16_time = None
    
    # Results
    flops = 2 * N**3
    print(f"\nMatrix Multiply: [{N}×{N}] × [{N}×{N}]")
    print(f"(This is the EXACT size of Q/K/V projections in LLaMA-7B)\n")
    
    print(f"{'Precision':<12} {'Time (ms)':<12} {'TFLOPS':<10} {'Speedup':<10} {'Hardware'}")
    print(f"{'-'*60}")
    
    fp32_tflops = flops / fp32_time / 1e12
    print(f"{'FP32':<12} {fp32_time*1000:<12.2f} {fp32_tflops:<10.1f} {'1.0x':<10} {'CUDA Cores'}")
    
    tf32_tflops = flops / tf32_time / 1e12
    tf32_speedup = fp32_time / tf32_time
    print(f"{'TF32':<12} {tf32_time*1000:<12.2f} {tf32_tflops:<10.1f} {tf32_speedup:<10.1f}x {'Tensor Cores (auto!)'}")
    
    fp16_tflops = flops / fp16_time / 1e12
    fp16_speedup = fp32_time / fp16_time
    print(f"{'FP16':<12} {fp16_time*1000:<12.2f} {fp16_tflops:<10.1f} {fp16_speedup:<10.1f}x {'Tensor Cores'}")
    
    if bf16_time:
        bf16_tflops = flops / bf16_time / 1e12
        bf16_speedup = fp32_time / bf16_time
        print(f"{'BF16':<12} {bf16_time*1000:<12.2f} {bf16_tflops:<10.1f} {bf16_speedup:<10.1f}x {'Tensor Cores'}")
    
    print(f"\nKey insight: TF32 is FREE — same FP32 code, just set allow_tf32=True")
    print(f"FP16 gives another ~2x on top of TF32")
    print(f"These speedups are EXACTLY what happens inside every LLM")
```

## Exercise 3: Visualize GPU Memory During Model Loading

```python
import torch

if not torch.cuda.is_available():
    print("Need GPU!")
else:
    def print_mem(label):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_global_mem / 1024**3
        free = total - allocated
        bar_len = 40
        used_bar = int(allocated / total * bar_len)
        print(f"{label:<40} | {'█' * used_bar}{'░' * (bar_len - used_bar)} | {allocated:.2f}/{total:.1f} GB")
    
    print("Watching GPU memory as we simulate loading model components:\n")
    print_mem("Empty GPU")
    
    # Simulate embedding table (32000 vocab × 4096 dim)
    embedding = torch.randn(32000, 4096, device='cuda', dtype=torch.float16)
    print_mem("+ Embedding table (32K × 4096)")
    
    # Simulate one attention layer's weights
    W_Q = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    W_K = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    W_V = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    W_O = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    print_mem("+ 1 attention layer (Q,K,V,O)")
    
    # Simulate one FFN layer's weights
    W_up = torch.randn(4096, 11008, device='cuda', dtype=torch.float16)
    W_gate = torch.randn(4096, 11008, device='cuda', dtype=torch.float16)
    W_down = torch.randn(11008, 4096, device='cuda', dtype=torch.float16)
    print_mem("+ 1 FFN layer (up, gate, down)")
    
    # Show per-layer cost
    attn_mb = 4 * 4096 * 4096 * 2 / 1024**2
    ffn_mb = (2 * 4096 * 11008 + 11008 * 4096) * 2 / 1024**2
    layer_mb = attn_mb + ffn_mb
    total_32_layers = 32 * layer_mb / 1024
    
    print(f"\n--- Per-Layer Memory Breakdown ---")
    print(f"Attention (Q,K,V,O): {attn_mb:.0f} MB")
    print(f"FFN (up,gate,down):  {ffn_mb:.0f} MB")
    print(f"Total per layer:     {layer_mb:.0f} MB")
    print(f"× 32 layers:         {total_32_layers:.1f} GB")
    print(f"+ embedding + output: ~0.5 GB")
    print(f"= Total model:       ~{total_32_layers + 0.5:.1f} GB (≈ LLaMA-7B at FP16)")
    
    # Cleanup
    del embedding, W_Q, W_K, W_V, W_O, W_up, W_gate, W_down
    torch.cuda.empty_cache()
    print_mem("\nAfter cleanup")
```

---

# PART 10: TODAY'S MINI-PROJECT 🔨

## Project: "GPU Detective" — Know Your Hardware

**Goal:** Build a Python script that acts as a "GPU detective" — it queries your GPU,
runs quick benchmarks, and tells you exactly what LLM workloads your GPU can handle.

**What to build:**
```
OUTPUT SHOULD LOOK LIKE:

╔══════════════════════════════════════════════════════╗
║                   GPU DETECTIVE REPORT                ║
╠══════════════════════════════════════════════════════╣
║  GPU: Tesla T4                                        ║
║  Architecture: Turing (Compute 7.5)                   ║
║  SMs: 40 | CUDA Cores: ~2560 | Tensor Cores: ~320    ║
║  Memory: 15.0 GB | Bandwidth: ~320 GB/s               ║
╠══════════════════════════════════════════════════════╣
║                                                       ║
║  BENCHMARK RESULTS:                                   ║
║  FP32 GEMM [4096×4096]: 8.2 TFLOPS (53% of peak)     ║
║  FP16 GEMM [4096×4096]: 51.3 TFLOPS (79% of peak)    ║
║  Tensor Core speedup: 6.3x                            ║
║  Memory bandwidth: 298 GB/s (93% of theoretical)      ║
║                                                       ║
║  LLM COMPATIBILITY:                                   ║
║  ✅ LLaMA-7B INT4  (3.5 GB) — ~91 tok/s              ║
║  ✅ LLaMA-7B FP16  (13.4 GB) — ~24 tok/s             ║
║  ❌ LLaMA-13B FP16 (26 GB) — doesn't fit             ║
║  ✅ LLaMA-13B INT4 (6.5 GB) — ~49 tok/s              ║
║  ❌ LLaMA-70B (any) — doesn't fit                    ║
║                                                       ║
╚══════════════════════════════════════════════════════╝
```

**Steps:**
1. Combine Exercise 1 (GPU properties) + Exercise 2 (benchmarks) into one script
2. Add a nice formatted output with the box drawing characters
3. Add the LLM compatibility check
4. Calculate tokens/sec estimates based on your actual measured bandwidth

**Why this is a good first project:**
- You learn the PyTorch CUDA API
- You verify your GPU understanding with REAL numbers
- You get a tool you'll use throughout the roadmap to check GPU status
- It's small enough to finish in 30-60 minutes
- The output looks impressive (show it to your team!)

**Bonus challenges (if you have time):**
- Add a bandwidth test: allocate large tensor, copy it, measure GB/s
- Compare your measured TFLOPS with the official spec sheet — how close are you?
- Save the report to a text file

---

# WHAT TO DO NEXT

After Day 2:
- [ ] You know every component inside an SM and what it does for LLMs
- [ ] You understand warp scheduling and occupancy
- [ ] You know the difference between CUDA Cores, Tensor Cores, and SFUs
- [ ] You understand registers, shared memory, L1/L2 cache
- [ ] You know what changed in each GPU generation for AI (Volta → Blackwell)
- [ ] You've explored NGC Catalog and build.nvidia.com
- [ ] You've queried your GPU properties with code
- [ ] You've measured Tensor Core vs CUDA Core speedup
- [ ] You've built the "GPU Detective" mini-project

**Day 3** shifts to MATH FOUNDATIONS: linear algebra, calculus, probability —
the mathematical tools you'll need to understand neural networks and Transformers.
It'll connect back to hardware: "this math operation maps to THIS hardware unit."

---

*Status: ⬜ NOT YET COMPLETED*
*Date completed: ___________*
