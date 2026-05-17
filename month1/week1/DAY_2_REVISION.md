# W1D2 Revision — NVIDIA GPU Physical Architecture
## ⏱️ 10-minute speed-revision sheet

> **One-line summary:** A GPU is a collection of SMs (factory floors); each SM has CUDA Cores, Tensor Cores, registers, and shared memory; warp schedulers keep multiple warps in-flight to hide memory latency; Tensor Cores do almost all LLM compute.

---

## 🧠 30-Second Mental Model

```
GPU
├── GigaThread Engine (CEO — assigns blocks to SMs)
├── L2 Cache (shared warehouse, ~50 MB on H100)
├── HBM (external storage, 80 GB on H100)
└── SMs × N  (N = 132 on H100, 192 on B200)
       │
       └── ONE SM = self-contained factory floor:
             ├── 4 Warp Schedulers (the managers)
             ├── 128 FP32 CUDA Cores  (general math, 1 op/cycle each)
             ├── 64 INT32 Cores       (index/address math)
             ├── 4 Tensor Cores       (matrix multiply — 90%+ of LLM work)
             ├── 16 SFUs              (sin, cos, exp, sqrt, 1/x)
             ├── 32 Load/Store units  (memory ops)
             ├── Register File: 65,536 × 32-bit = 256 KB
             └── Shared Memory + L1: 228 KB (programmer-split)
```

If you remember **only one thing**: an SM has **way more threads than cores**, and the warp scheduler hides slow memory by switching between ready warps every cycle.

---

## 1️⃣ SM Anatomy — Numbers to Memorize (90 sec)

### Per-SM hardware (H100):
| Component | Count / Size | What it does |
|---|---|---|
| **Warp schedulers** | 4 | Pick 1 ready warp/cycle, issue 1 instruction |
| **FP32 CUDA cores** | 128 | One multiply OR add per cycle (general math) |
| **INT32 cores** | 64 | Index/address calculations |
| **FP64 cores** | 2 | Scientific (basically unused for AI) |
| **Tensor Cores** | 4 | Whole 16×16 matmul in **1 cycle** |
| **SFUs** | 16 | Transcendentals: sin, cos, exp, sqrt, 1/x |
| **Load/Store** | 32 | One full warp's worth per cycle |
| **Register file** | 65,536 × 32-bit = **256 KB** | Fastest storage, 1-cycle access |
| **Shared mem + L1** | **228 KB** total (configurable split) | ~20-30 cycle access |

### Whole-GPU comparison:
| GPU | SMs | CUDA Cores | Tensor Cores | HBM |
|---|---|---|---|---|
| RTX 3090 | 82 | 10,496 | 328 | 24 GB |
| RTX 4090 | 128 | 16,384 | 512 | 24 GB |
| A100 | 108 | 6,912 | 432 | 80 GB |
| **H100** | **132** | **16,896** | **528** | **80 GB HBM3** |
| B200 | 192 | — | — | 192 GB HBM3e |

**More SMs ⇒ more parallel work ⇒ faster AI.**

---

## 2️⃣ Warp Schedulers + Occupancy (90 sec)

### How a warp executes (the cycle):
1. Scheduler looks at ~16 in-flight warps; finds one whose operands are ready
2. Issues **one instruction** for that warp → 32 threads execute in lockstep
3. While that warp's data fetch is in-flight, scheduler picks a *different* warp
4. **This switching is FREE** — no cost like a CPU context switch

### Key terms:
- **Warp** = 32 threads executing in lockstep (SIMT)
- **Block** = up to 1024 threads (= up to 32 warps), runs entirely on ONE SM
- **Occupancy** = `(active warps on SM) / (max warps SM can hold)`. H100 max ≈ 64 warps/SM
- **Higher occupancy → more warps to switch between → better latency hiding**
- Limits on occupancy: **registers per thread** (most common), shared memory per block, threads per block

### The "2048 threads, 128 cores" puzzle:
- An SM holds ~2048 threads but only has 128 FP32 cores → **most threads are SLEEPING (waiting on memory)**
- This is intentional: warps cycle through cores; the schedulers keep cores busy by always finding a ready warp
- A core does ~one op/cycle for whichever warp's instruction is currently issued

### 🍳 Memorable analogy:
- **Restaurant kitchen**: 4 chefs (cores), 32 prep stations (one warp), 64 dishes in progress (warps), 1 head chef (scheduler) deciding who cooks next. While dish A waits for the oven, head chef switches to dish B. No idle chefs.

---

## 3️⃣ Tensor Cores — Where 90%+ of LLM Compute Happens (90 sec)

### CUDA Core vs Tensor Core:
| | CUDA Core | Tensor Core |
|---|---|---|
| Operation | scalar (1 multiply OR add) | **D = A · B + C** for 16×16 matrices |
| Per cycle | 1 op | **256 multiply-accumulates** in parallel |
| For LLMs | non-matrix ops, normalization, activations | **all matmuls** (attention, FFN) |
| Speedup | baseline | ~10-16× faster on the same matmul |

### Supported precisions (memorize: which generation introduced what):
| Format | Bits | Use case | Generation |
|---|---|---|---|
| FP16 | 16 | Inference, training | Volta (V100) |
| BF16 | 16 | Training (same range as FP32) | Ampere (A100) |
| TF32 | 19 | Auto FP32 "upgrade" | Ampere (A100) |
| INT8 | 8 | Quantized inference | Volta+ |
| **FP8** | 8 | Fastest, training+inference | **Hopper (H100)** |
| **FP4 / FP6** | 4-6 | Fastest yet | **Blackwell (B100/B200)** |

**Why low precision matters:** halving bits → roughly doubles Tensor Core throughput (e.g., FP8 ≈ 2× FP16 throughput).

### LLM connection:
- Attention's `Q·Kᵀ`, `softmax(...)·V`, FFN's `x·W₁`, `x·W₂` → **all Tensor Core work**
- LayerNorm, RoPE, GELU, softmax exp → CUDA Cores + SFUs

---

## 4️⃣ Register File + Memory Hierarchy (90 sec)

### Per-SM memory ladder (latency, fastest first):
| Tier | Size (per SM) | Latency | Who controls |
|---|---|---|---|
| **Registers** | 256 KB | **~1 cycle** | Compiler |
| **Shared memory** | up to 228 KB | ~20-30 cycles | Programmer (`__shared__`) |
| **L1 cache** | shares the 228 KB pool | ~30 cycles | Hardware (auto) |
| **L2 cache** | 50 MB (whole-GPU shared) | ~200 cycles | Hardware |
| **HBM (global)** | 80 GB (H100) | **~400-800 cycles** | DRAM |

### Register pressure (the hidden bottleneck):
- 65,536 registers shared by all threads on the SM
- **2048 threads → 32 regs/thread**. **1024 threads → 64 regs/thread**.
- A kernel that needs 64 regs/thread can only run 1024 threads/SM → lower occupancy
- Worse: if a kernel needs >255 regs, the compiler **spills** to "local memory" (which is actually in slow HBM!)

### Cache lines and memory transactions:
- HBM reads happen in **32-byte (sector) or 128-byte (cache line) chunks** — never 1 byte
- **Coalesced read**: 32 threads of a warp read 32 contiguous floats → 1 transaction → fast
- **Strided read**: 32 threads read scattered locations → 32 transactions → 32× slower
- **This is why memory layout matters** for performance

---

## 5️⃣ GPU Generation Evolution — What Changed for AI (60 sec)

| Gen | Year | Big Innovation | Why It Matters |
|---|---|---|---|
| **Volta** (V100) | 2017 | First Tensor Cores (FP16) | Birth of "AI GPU" |
| **Turing** (T4) | 2018 | INT8 Tensor Cores | Inference acceleration |
| **Ampere** (A100) | 2020 | BF16, TF32, sparsity 2:4 | Training scale-up |
| **Hopper** (H100) | 2022 | **FP8**, Transformer Engine, TMA, WGMMA, NVLink-Switch | Made GPT-class training feasible |
| **Blackwell** (B100/B200) | 2024 | **FP4/FP6**, dedicated decompression engine, 5th-gen Tensor Cores | 1.5-1.85× over H100 |

**The pattern:** every gen adds (1) lower-precision Tensor Core formats, (2) more memory bandwidth, (3) better inter-GPU links (NVLink), (4) bigger HBM.

---

## 6️⃣ NGC + build.nvidia.com (15 sec)

- **NGC Catalog** (`catalog.ngc.nvidia.com`): NVIDIA's container/model registry — pre-optimized PyTorch, TensorFlow, Triton, NeMo containers
- **build.nvidia.com**: free API playground — try Llama, Mistral, embedding/reranker models without deploying anything

---

## 🔍 Quick Recall (close the file, answer in your head — 60 sec)

1. How many SMs in H100? *(132)*
2. What's a warp size? *(32 threads, lockstep)*
3. What does ONE Tensor Core compute per cycle? *(16×16 matmul fused with add)*
4. What's the latency difference between registers and HBM? *(~1 cycle vs ~400-800 cycles)*
5. Which precision did Hopper add that Ampere didn't? *(FP8)*
6. Why does an SM hold more threads than it has cores? *(latency hiding via warp switching)*
7. What is "register pressure"? *(too many regs/thread → low occupancy → fewer warps to switch between)*
8. Coalesced vs strided read difference? *(1 vs 32 transactions for the same 32 floats)*
9. Where is the L2 cache? *(whole-GPU shared, between SMs and HBM, ~50 MB on H100)*
10. What's NGC? *(NVIDIA's container/model registry)*

---

## 🎯 If You Remember Only Three Things

1. **An SM is the unit.** Everything you study about CUDA happens inside an SM. H100 = 132 SMs.
2. **Tensor Cores are why LLMs run on GPUs at all.** They do 16×16 matmuls in 1 cycle. ~10-16× faster than scalar cores. Each generation adds smaller float formats (FP16 → BF16 → FP8 → FP4).
3. **Latency hiding via warps is the killer trick.** Each SM holds ~64 warps; while one waits on memory, schedulers switch to another. This is why memory-bound code can still be fast.

---

*Revision file generated from `DAY_2.md`. For deep dive, see the original DAY_2.md (1602 lines).*
