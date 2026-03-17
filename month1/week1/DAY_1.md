# Week 1, Day 1: CPU vs GPU Architecture
## Why This Matters for LLMs

Every time someone runs ChatGPT, Llama, or any LLM — thousands of GPUs are crunching
matrix multiplications in parallel. Understanding WHY GPUs are used (not CPUs) and
HOW they work at the hardware level is what separates you from 99% of AI engineers.

By the end of this study session, you'll understand:
- How a CPU executes instructions at the hardware level
- How a GPU is fundamentally different in design philosophy
- Why GPUs are 100-1000x faster for AI workloads
- What actually happens inside the chip when an LLM generates a token

---

# PART 0: JARGON BUSTER — Read This First

Before we dive in, here's every term you need, explained simply. Come back to this
section anytime you see a word you don't recognize.

## Numbers & Precision

**Bit:** The smallest unit of data. Either 0 or 1.

**Byte:** 8 bits. Can represent a number from 0 to 255.

**Floating-point number:** How computers store decimal numbers (like 3.14159).
The "floating point" means the decimal point can move (float) to represent
very large or very small numbers.

**FP32 (Float 32-bit):** Uses 32 bits (4 bytes) to store one number.
Can represent numbers with ~7 digits of precision.
Example: 3.1415927. This is the "default" precision in computing.

**FP16 (Float 16-bit):** Uses 16 bits (2 bytes) — HALF the space of FP32.
Only ~3 digits of precision. Less accurate, but takes half the memory
and can be processed 2x faster. Good enough for most AI!

**BF16 (BFloat16):** Also 16 bits, but arranged differently than FP16.
It can handle the same RANGE of numbers as FP32 (very big and very small)
but with less precision. Preferred for AI training because it doesn't overflow.

**FP8 (Float 8-bit):** Only 1 byte per number. Very low precision but
4x smaller than FP32. Used on newest GPUs (H100) for even faster AI.

**INT4 / INT8:** Integer formats (whole numbers only, no decimals).
INT4 uses half a byte (4 bits) per number — can only represent 16 values!
INT8 uses 1 byte — can represent 256 values.
Used for "quantization" (making models smaller and faster).

**Why this matters:** A model with 7 billion numbers at FP32 = 28 GB.
Same model at FP16 = 14 GB. At INT4 = 3.5 GB. Smaller = faster to read
from memory = faster inference. This is the core idea of quantization.

## Math Concepts

**Scalar:** Just one number. Like 42 or 3.14.

**Vector:** A list of numbers. Like [0.5, -0.3, 0.8, 0.1].
In AI, a vector often represents the "meaning" of a word or token.
LLMs typically use vectors of 4096 numbers (called "4096-dimensional").

Think of it like GPS coordinates but instead of 2 dimensions (latitude, longitude),
you have 4096 dimensions. Each dimension captures some aspect of meaning.
"King" and "Queen" would have similar coordinates (nearby in 4096-D space).

**Matrix:** A grid of numbers. Like a spreadsheet with rows and columns.
```
Matrix A (3 rows, 4 columns) = "3×4 matrix":
┌                     ┐
│  1.0  2.0  3.0  4.0 │  ← row 0
│  5.0  6.0  7.0  8.0 │  ← row 1
│  9.0 10.0 11.0 12.0 │  ← row 2
└                     ┘
```
In LLMs, weight matrices are HUGE. Like 4096 rows × 4096 columns = 16.7 million numbers.

**Matrix Multiplication:** The core operation in ALL of AI. Here's what it does:

```
A (2×3) × B (3×2) = C (2×2)

┌         ┐     ┌      ┐     ┌                            ┐
│ 1  2  3 │  ×  │ 7  8 │  =  │ 1×7+2×9+3×11  1×8+2×10+3×12 │
│ 4  5  6 │     │ 9 10 │     │ 4×7+5×9+6×11  4×8+5×10+6×12 │
└         ┘     │11 12 │     └                            ┘
                └      ┘
                              ┌           ┐
                           =  │  58   64  │
                              │ 139  154  │
                              └           ┘

Each element of the result = dot product of a row from A with a column from B.
A dot product is: multiply corresponding elements, then add them all up.

For a 4096×4096 × 4096×4096 multiply:
  Each element = 4096 multiplications + 4096 additions = 8192 operations
  Total elements = 4096 × 4096 = 16.7 million
  Total operations = 16.7M × 8192 = ~137 BILLION operations

THIS IS WHY GPUs EXIST FOR AI. This one operation, done billions of times,
is what makes language models work. And it's perfectly parallel — every
element of the output can be computed independently.
```

**Why matrix multiply is the #1 operation in LLMs:**
Every single layer in a Transformer does multiple matrix multiplications.
A 7-billion-parameter model does ~14 billion multiply-add operations
just to generate ONE word. ChatGPT generates ~50 words per second.
That's 700 billion operations per second, sustained.

**Softmax:** Converts a list of numbers into probabilities that sum to 1.0.
```
Input:  [2.0, 1.0, 0.1]
           ↓ softmax
Output: [0.659, 0.242, 0.099]    (sums to 1.0)

The biggest input gets the biggest probability.
Used in attention ("how much should I focus on each word?")
and at the end ("which next word is most likely?").
```

## AI / LLM Concepts

**Parameter / Weight:** A single number that the model learned during training.
When we say "LLaMA-7B has 7 billion parameters", it means there are 7 billion
floating-point numbers stored in the model file. These numbers were learned
by training on trillions of words of text.

**Token:** A piece of text — could be a word, part of a word, or a punctuation mark.
LLMs don't read letters or words — they read tokens.
```
"Hello, how are you?" → ["Hello", ",", " how", " are", " you", "?"]
                         token 0  token 1  token 2  token 3  token 4  token 5

Why not just use words? Because:
- "unhappiness" is rare as a whole word, but "un" + "happiness" are common
- This way the model needs a smaller vocabulary (32,000 tokens vs millions of words)
- It can handle ANY text, even made-up words like "GPT-ify"
```

**Embedding:** Converting a token (an integer ID) into a vector (a list of numbers).
```
Token "cat" → ID 2581 → look up row 2581 in embedding table → [0.12, -0.34, 0.78, ...]
                                                                 (4096 numbers)
```
This vector captures the "meaning" of "cat" in a way math can work with.

**Inference:** Using a trained model to generate output (like ChatGPT answering you).
The model weights are FIXED — you're just reading them and doing math.

**Training:** Teaching the model by adjusting its weights based on data.
Show it "The capital of France is ___", it guesses "London" (wrong!),
calculate how wrong it was, nudge all 7 billion weights slightly to make
"Paris" more likely next time. Repeat trillions of times.

**Attention:** The mechanism that lets each word "look at" all other words
to understand context. "It" in "The cat sat on the mat. It was tired."
— attention is how the model figures out "It" refers to "cat", not "mat".

Think of it as a lookup system:
- **Query (Q):** "I'm the word 'It' and I need to know what I refer to" (the question)
- **Key (K):** Each previous word advertises: "I'm 'cat', I'm an animal subject" (the labels)
- **Value (V):** Each previous word's actual information: "here's everything I know about cats" (the content)
- The model compares the Query against all Keys, finds the best match ("cat"),
  and retrieves that word's Value.

This Q/K/V lookup is implemented as... matrix multiplications!

## Hardware Jargon

**FLOP:** Floating-Point Operation. One add or one multiply of decimal numbers.

**GFLOPS:** Billion (Giga) FLOPs per second. Your laptop CPU does ~100 GFLOPS.

**TFLOPS:** Trillion (Tera) FLOPs per second. An H100 GPU does 990 TFLOPS
with Tensor Cores. That's 990,000,000,000,000 operations per second.

**Bandwidth:** How fast data moves. Measured in GB/s (gigabytes per second).
Like the width of a highway — wider = more cars (data) can flow.
- CPU RAM (DDR5): ~300 GB/s (country road)
- GPU HBM3 (H100): 3,350 GB/s (superhighway)

**Memory-bound vs Compute-bound:** The two types of bottleneck.

Analogy — a restaurant:
```
COMPUTE-BOUND (too few cooks):
  Kitchen has 2 cooks but 100 orders.
  Ingredients arrive instantly from the pantry.
  Cooks are always busy. Pantry is waiting.
  Solution: hire more cooks (more TFLOPS).

MEMORY-BOUND (pantry too slow):
  Kitchen has 100 cooks but the pantry is tiny
  with a narrow door. Cooks wait for ingredients.
  Most cooks are idle.
  Solution: bigger pantry door (more bandwidth).

LLM inference = MEMORY-BOUND (pantry too slow)
  The "cooks" (Tensor Cores) could compute much faster,
  but they're waiting for "ingredients" (model weights)
  to arrive from GPU memory (HBM).

LLM training = COMPUTE-BOUND (too few cooks)
  Large batches mean lots of work per weight read.
  The "cooks" are always busy.
```

**HBM (High Bandwidth Memory):** Special memory chips stacked on top of the GPU.
Much faster than regular RAM. "HBM" is like having the pantry RIGHT NEXT
to the kitchen instead of down the hall.

**Tensor Core:** Specialized hardware inside NVIDIA GPUs that can do a small
matrix multiply (16×16) in a single clock cycle. Regular cores do one
multiply per cycle. So Tensor Cores are ~256x more efficient for matrix math.
They only work with reduced precision (FP16, BF16, FP8) — not FP32.

**SM (Streaming Multiprocessor):** A self-contained processing unit inside the GPU.
Each SM has its own CUDA cores, Tensor Cores, shared memory, and registers.
A modern GPU has ~100-132 SMs. Think of each SM as one "factory floor"
in a massive manufacturing plant.

---

Now you're ready. Every term used in the rest of this document is explained above.
Flip back to this section anytime you need a refresher.

---

# PART 1: THE CPU — A Deep Thinker

## 1.1 Von Neumann Architecture

Every modern CPU follows the Von Neumann architecture (1945). The core idea:

```
┌─────────────────────────────────────────────┐
│                    CPU                       │
│  ┌───────────┐    ┌───────────────────────┐ │
│  │  Control   │    │   Arithmetic Logic    │ │
│  │   Unit     │    │     Unit (ALU)        │ │
│  │  (CU)      │    │   + − × ÷ AND OR     │ │
│  └─────┬─────┘    └──────────┬────────────┘ │
│        │                     │               │
│        └──────┬──────────────┘               │
│               │                              │
│        ┌──────┴──────┐                       │
│        │  Registers  │  (tiny, fastest)      │
│        └──────┬──────┘                       │
│               │                              │
│        ┌──────┴──────┐                       │
│        │   L1 Cache  │  (32-64 KB, ~1ns)     │
│        └──────┬──────┘                       │
│               │                              │
│        ┌──────┴──────┐                       │
│        │   L2 Cache  │  (256KB-1MB, ~3ns)    │
│        └──────┬──────┘                       │
│               │                              │
│        ┌──────┴──────┐                       │
│        │   L3 Cache  │  (8-64MB, ~10ns)      │
│        └──────┬──────┘                       │
└───────────────┼──────────────────────────────┘
                │
         ┌──────┴──────┐
         │  Main RAM   │  (16-128GB, ~100ns)
         └─────────────┘
```

**The fetch-decode-execute cycle:**

Every CPU instruction goes through these steps:

1. **Fetch**: Read the next instruction from memory (pointed to by the Program Counter)
2. **Decode**: Figure out what the instruction means (is it an add? a multiply? a branch?)
3. **Execute**: Actually do the operation in the ALU
4. **Write back**: Store the result

This happens BILLIONS of times per second (a 4 GHz CPU does 4 billion cycles/sec).

## 1.2 Why CPUs Are Smart But Slow (for parallelism)

A modern CPU core is an engineering marvel of COMPLEXITY. Here's what a single core has:

### Branch Prediction
When code has an if/else:
```c
if (x > 0) {
    y = x * 2;    // Branch A
} else {
    y = x + 1;    // Branch B
}
```
The CPU GUESSES which branch will be taken BEFORE it knows the answer (using historical
patterns). If it guesses right (~95% of the time), execution continues at full speed.
If wrong, it throws away the speculative work (pipeline flush = ~15-20 cycles wasted).

**Why this matters for LLMs:** GPUs don't have branch prediction. They handle branches
completely differently (all threads execute both paths, mask inactive ones). This is
why GPU code should minimize branching.

### Out-of-Order Execution
```c
a = load(memory[100])    // Takes ~100 cycles to get from RAM
b = 5 + 3                // This doesn't depend on 'a'
c = a * 2                // This depends on 'a'
```
A smart CPU will execute `b = 5 + 3` WHILE WAITING for `a` to load from memory.
It analyzes dependencies and reorders instructions to keep the ALU busy.

A CPU has a huge "reorder buffer" (hundreds of entries) to track all this.

**Why this matters for LLMs:** GPUs don't do out-of-order execution per thread.
Instead, they hide latency by switching to OTHER threads (thousands of them).
Completely different strategy, same goal: keep the compute units busy.

### Speculative Execution
The CPU starts executing instructions that it MIGHT need, before knowing for sure.
Combined with branch prediction, it often has the answer ready before it's asked.

### Large Caches
A modern CPU dedicates ~50% of its transistor budget to caches:
- L1: 32-64 KB per core (~1 nanosecond access)
- L2: 256 KB - 1 MB per core (~3 ns)
- L3: 8-64 MB shared across cores (~10 ns)
- Compare: Main RAM is ~100 ns (100x slower than L1!)

The CPU BETS that you'll access nearby memory soon (spatial locality) or the same
memory again (temporal locality). For sequential code, this bet pays off hugely.

**Why this matters for LLMs:** GPUs have much SMALLER caches per thread. They rely
on massive parallelism and high memory bandwidth instead of caching. Understanding
this difference is key to writing fast GPU code.

## 1.3 CPU Die Layout

If you look at a CPU die photo:
```
┌─────────────────────────────────────────────────┐
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│ │Core │ │Core │ │Core │ │Core │ │Core │ ...    │
│ │  0  │ │  1  │ │  2  │ │  3  │ │  4  │ (8-64 │
│ │     │ │     │ │     │ │     │ │     │ cores) │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
│                                                  │
│         ┌──────────────────────────┐             │
│         │    Large L3 Cache        │             │
│         │    (up to 64 MB)         │             │
│         └──────────────────────────┘             │
│                                                  │
│ ┌────────┐  ┌────────┐  ┌────────┐              │
│ │Memory  │  │  PCIe  │  │  I/O   │              │
│ │Control │  │Control │  │        │              │
│ └────────┘  └────────┘  └────────┘              │
└─────────────────────────────────────────────────┘
```

Notice: MOST of the die is cache and control logic. Only a SMALL fraction is actual
compute (ALUs). This is by design — the CPU optimizes for LATENCY (how fast can I
finish ONE task) rather than THROUGHPUT (how many tasks per second).

**Key stat: Only ~5% of a CPU die area is ALUs (actual compute).**

---

# PART 2: THE GPU — A Massive Parallel Army

## 2.1 The GPU Design Philosophy

The GPU takes the OPPOSITE approach to the CPU:

**CPU philosophy:** Make ONE thread go as fast as possible.
**GPU philosophy:** Run THOUSANDS of threads, each simple, to maximize total work done.

```
CPU (8 cores):                    GPU (thousands of cores):
┌─────────────────┐              ┌─────────────────────────┐
│ ┌──┐┌──┐┌──┐┌──┐│              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│ │C0││C1││C2││C3││              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│ │  ││  ││  ││  ││              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│ └──┘└──┘└──┘└──┘│              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│ ┌──┐┌──┐┌──┐┌──┐│              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│ │C4││C5││C6││C7││              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│ │  ││  ││  ││  ││              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│ └──┘└──┘└──┘└──┘│              │ ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪│
│   BIG caches     │              │ Each ▪ = small core      │
│   Complex logic  │              │ 10,000+ cores!           │
└─────────────────┘              │ Tiny caches              │
                                  │ Simple control           │
                                  └─────────────────────────┘
8 powerful cores                  Thousands of simple cores
Optimized for LATENCY            Optimized for THROUGHPUT
```

**Key stat: ~80% of a GPU die area is ALUs (actual compute).** Compare with CPU's ~5%.

## 2.2 Why Parallelism Matters for AI

Here's the fundamental insight. Consider what happens during LLM inference:

When a model like LLaMA-7B generates ONE token, it needs to:
1. Look up the token embedding (simple table lookup)
2. Pass through 32 transformer layers, each involving:
   - Matrix multiply for Q, K, V projections: [4096 × 4096] × [4096 × 1]
   - Attention computation across all positions
   - Matrix multiply for output projection: [4096 × 4096] × [4096 × 1]
   - Matrix multiply for FFN up-projection: [4096 × 11008] × [11008 × 1]
   - Matrix multiply for FFN down-projection: [11008 × 4096] × [4096 × 1]
3. Final linear layer to vocabulary: [4096 × 32000] × [32000 × 1]

**Total: ~14 BILLION multiply-and-add operations per token.**

At 4 GHz, a single CPU core doing one multiply per cycle would take:
14,000,000,000 / 4,000,000,000 = **3.5 seconds per token.**

A GPU with 10,000 cores running at 1.5 GHz:
14,000,000,000 / (10,000 × 1,500,000,000) = **~0.001 seconds per token.**

That's the difference: **3.5 seconds vs 1 millisecond.** 3,500x faster.

(Real numbers are more complex due to memory bandwidth, but the principle holds.)

## 2.3 SIMT: Single Instruction, Multiple Threads

This is the GPU's execution model. Understanding SIMT is critical for everything ahead.

On a CPU, each core runs its own independent instruction stream:
```
Core 0: load A[0]    Core 1: branch if x>0    Core 2: multiply C[2]*D[2]
         add A[0]+B[0]        call function()          store result
         store C[0]           load Y[7]                load E[5]
```
Each core does completely different things at the same time.

On a GPU, groups of threads (called a **warp** = 32 threads) execute the SAME
instruction but on DIFFERENT data:

```
Warp (32 threads executing the SAME instruction simultaneously):

Instruction: "add A[i] + B[i]"

Thread 0:  A[0] + B[0]
Thread 1:  A[1] + B[1]
Thread 2:  A[2] + B[2]
Thread 3:  A[3] + B[3]
...
Thread 31: A[31] + B[31]

All 32 threads execute "add" at the SAME clock cycle.
Only the index [i] is different.
```

This is SIMT: **Single Instruction, Multiple Threads.**

Every thread in a warp is in lockstep. They all execute the same instruction
at the same time. The only thing that differs is the data each thread operates on.

**Why this is perfect for LLMs:** Matrix multiplication is EXACTLY this pattern.
Every element of the output matrix is computed the same way (multiply and accumulate),
just on different data. There are no branches, no dependencies between elements.
It's embarrassingly parallel.

## 2.4 The GPU Thread Hierarchy

GPUs organize work in a hierarchy. This is CRITICAL to understand:

```
GRID (the entire job)
├── Block (0,0)        Block (1,0)        Block (2,0)
│   ├── Warp 0         ├── Warp 0         ├── Warp 0
│   │   ├── Thread 0   │   ├── Thread 0   │   ├── Thread 0
│   │   ├── Thread 1   │   ├── Thread 1   │   ├── Thread 1
│   │   ├── ...        │   ├── ...        │   ├── ...
│   │   └── Thread 31  │   └── Thread 31  │   └── Thread 31
│   ├── Warp 1         ├── Warp 1         ├── Warp 1
│   │   ├── Thread 32  │   ├── Thread 32  │   ├── Thread 32
│   │   ├── ...        │   ├── ...        │   ├── ...
│   │   └── Thread 63  │   └── Thread 63  │   └── Thread 63
│   └── ...            └── ...            └── ...
├── Block (0,1)        Block (1,1)        Block (2,1)
│   └── ...            └── ...            └── ...
└── ...
```

**Thread:** The smallest unit. One thread does one piece of work.

**Warp (32 threads):** The actual unit of execution. All 32 threads in a warp
execute in lockstep (SIMT). You never launch less than 32 threads effectively.
This is a HARDWARE constraint, not a software choice.

**Block (up to 1024 threads = up to 32 warps):** A group of threads that can
cooperate. Threads within a block can:
- Synchronize with each other (__syncthreads())
- Share data through fast "shared memory" (on-chip SRAM)
- A block runs on ONE Streaming Multiprocessor (SM)

**Grid (many blocks):** The entire computation. Blocks are distributed across
all SMs on the GPU. Blocks are INDEPENDENT — they cannot communicate directly.

**LLM Connection:** When you do a matrix multiply for an attention layer:
- Each block might compute a TILE of the output matrix
- Threads within a block cooperate to load data into shared memory
- The grid covers the entire output matrix

## 2.5 Latency Hiding: The GPU's Secret Weapon

Here's the GPU's genius trick. Memory is slow (~400-600 clock cycles to load from
global memory). On a CPU, you'd stall and wait. On a GPU:

```
Time →
Warp 0: [Compute] [WAITING for memory...................................] [Compute]
Warp 1:           [Compute] [WAITING for memory........................] [Compute]
Warp 2:                     [Compute] [WAITING for memory..............] [Compute]
Warp 3:                               [Compute] [WAITING for memory....] [Compute]
Warp 4:                                         [Compute] [WAITING.....] [Compute]
...
         ^--- GPU switches between warps instantly (zero cost!)
```

The GPU keeps HUNDREDS of warps ready to go. When one warp is waiting for memory,
the GPU instantly switches to another warp that's ready to compute. No cost to switch
(because each warp has its own registers — nothing to save/restore).

**This is why GPUs need MASSIVE parallelism to be fast.** If you only launch 32
threads (1 warp), the GPU will be idle 99% of the time waiting for memory.
Launch 10,000+ threads, and the GPU is always busy.

**LLM Connection:** A matrix multiply in a Transformer layer involves millions of
operations. This creates enough parallelism to keep the GPU fully occupied.
Small operations (like adding a bias vector) are actually inefficient on GPU because
there isn't enough work to hide the memory latency. This is why "kernel fusion" matters
(combining small operations into one big one).

## 2.6 Throughput vs Latency — The Core Tradeoff

```
Task: Add two arrays of 1 million elements

CPU (8 cores, 4 GHz):
- Each core processes ~125,000 elements
- Each element takes ~1ns (fully optimized with SIMD)
- Total time: ~125,000 ns = 0.125 ms
- But: each INDIVIDUAL element is done in ~1ns (LOW LATENCY)

GPU (10,000 cores, 1.5 GHz):
- Thousands of threads, each does a few elements
- Launch overhead: ~5-10 μs
- Actual compute: ~0.01 ms
- Total time: ~0.02 ms (including overhead)
- But: no single element is done faster than CPU (HIGHER per-element LATENCY)
```

**CPU wins when:**
- Few operations with complex logic (web servers, databases, OS)
- Lots of branching and conditional logic
- Low latency for individual operations matters
- Sequential dependencies (step N depends on step N-1)

**GPU wins when:**
- Same operation on MANY data elements (matrix multiply!)
- Minimal branching
- Total THROUGHPUT matters more than individual latency
- Operations are independent (parallelizable)

**LLMs are the PERFECT GPU workload because:**
1. Matrix multiplies are uniform operations on massive data → parallel
2. Attention computation is independent across heads → parallel
3. Batch processing: multiple sequences at once → more parallelism
4. Most operations are multiply-accumulate (what Tensor Cores are built for)

---

# PART 3: THE NUMBERS THAT MATTER

## 3.1 Real Hardware Comparison

Let's compare a top CPU vs a top GPU with actual numbers:

```
                        Intel Xeon w9-3595X    NVIDIA H100 SXM
                        (Top Server CPU)        (Top AI GPU)
─────────────────────────────────────────────────────────────────
Cores                   60                      16,896 FP32 CUDA cores
                                                528 Tensor Cores
Clock Speed             2.5 GHz (boost 4.8)    1.83 GHz (boost)
Transistors             ~55 billion             80 billion
Die Area                ~800 mm²               814 mm²
TDP (Power)             600W                    700W

FP32 Performance        ~4.6 TFLOPS             67 TFLOPS
FP16 Performance        ~9.2 TFLOPS            134 TFLOPS
FP16 Tensor Core        N/A                    990 TFLOPS
FP8 Tensor Core         N/A                    1,979 TFLOPS

Memory                  DDR5, up to 6TB        HBM3, 80 GB
Memory Bandwidth        ~300 GB/s              3,350 GB/s

Price                   ~$10,000               ~$30,000
```

**Key takeaways:**

1. **FP32 compute: GPU is ~15x faster** (67 vs 4.6 TFLOPS)
2. **With Tensor Cores at FP16: GPU is ~215x faster** (990 vs 4.6)
3. **With FP8 Tensor Cores: GPU is ~430x faster** (1,979 vs 4.6)
4. **Memory bandwidth: GPU is ~11x faster** (3,350 vs 300 GB/s)

The Tensor Core numbers are STAGGERING. This is why NVIDIA GPUs dominate AI:
they have specialized hardware for exactly the operations LLMs need.

## 3.2 What Are TFLOPS?

TFLOPS = Tera (trillion) Floating-Point Operations Per Second.

1 TFLOPS = 1,000,000,000,000 operations per second.

When we say H100 does 990 TFLOPS at FP16 with Tensor Cores, it means:
**990 trillion multiply-add operations per second.**

For context, generating one token with LLaMA-7B requires ~14 billion operations.
At 990 TFLOPS, the H100 can theoretically do: 990,000 / 14 = ~70,000 tokens per second.

(In practice it's much less due to memory bandwidth bottleneck, but the compute
is there.)

## 3.3 Memory Bandwidth: The Real Bottleneck

Here's a crucial insight most people miss:

**For LLM inference, the bottleneck is usually MEMORY BANDWIDTH, not compute.**

Why? During inference (generating tokens one at a time):
- You need to READ the entire model weights for each token
- LLaMA-7B at FP16 = 14 GB of weights
- To generate 1 token, you read 14 GB from memory
- At 3,350 GB/s bandwidth (H100): 14 / 3,350 = 0.004 seconds = 4 ms per token
- That's ~250 tokens/second

But the COMPUTE for 14B operations at 990 TFLOPS would take:
14,000,000,000 / 990,000,000,000,000 = 0.000014 seconds = 0.014 ms

So the compute takes 0.014 ms but reading the weights takes 4 ms.
**The GPU is idle 99.6% of the time waiting for memory!**

This is called being "memory-bandwidth bound" and it's the #1 performance
constraint for LLM inference. Everything in the inference optimization world
(quantization, speculative decoding, batching) is trying to solve this problem.

```
LLM Inference Reality:

Compute capacity:  ████████████████████████████████████████ 100%
Compute used:      █                                        ~0.4%
Memory bandwidth:  ████████████████████████████████████████ 100% (bottleneck!)
```

**This is why quantization matters so much:** If you quantize from FP16 (2 bytes)
to INT4 (0.5 bytes), the model is 4x smaller, so you read 4x less data,
so inference is ~4x faster. The compute is negligibly affected.

**This is why batching matters:** If you process 64 sequences at once, you
read the weights ONCE but do 64x the compute. This moves you from memory-bound
to compute-bound, which is where the GPU is happy.

## 3.4 Arithmetic Intensity: The Key Metric

**Arithmetic Intensity** = FLOPs / Bytes transferred

This tells you whether an operation is compute-bound or memory-bound.

```
                    Low arithmetic intensity        High arithmetic intensity
                    (memory-bound)                  (compute-bound)
                    ─────────────────────           ──────────────────────
Example:            Vector addition                 Matrix multiplication
                    c[i] = a[i] + b[i]             C = A × B

Operations:         1 add per element               2*N operations per element
Data movement:      3 values per element            2 values per element (amortized)
                    (read a, b, write c)

Intensity:          1/12 = 0.08 FLOP/byte           Very high for large N
                    (3 floats × 4 bytes)

GPU happy?          NO (waiting for memory)          YES (compute units busy)
```

**For LLMs:**
- Single-token inference: LOW arithmetic intensity → memory-bound → GPU sad
- Training (large batches): HIGH arithmetic intensity → compute-bound → GPU happy
- Batched inference (many users): MODERATE-HIGH → GPU happier

This is why the ENTIRE field of LLM serving is about finding ways to increase
arithmetic intensity: batch more requests, fuse more operations, use Tensor Cores.

---

# PART 4: THE GPU MEMORY HIERARCHY

## 4.1 Types of GPU Memory

The GPU has its own memory hierarchy, completely separate from the CPU:

```
FASTEST ──────────────────────────────────────────── SLOWEST
SMALLEST ─────────────────────────────────────────── LARGEST

┌─────────────┐
│  Registers  │  Per thread, ~255 registers
│  (~1 cycle) │  Total: 65,536 per SM × 32-bit
└──────┬──────┘
       │
┌──────┴──────┐
│   Shared    │  Per block, 48-228 KB per SM
│   Memory    │  (~20-30 cycles)
│   (SRAM)    │  On-chip, programmer-managed cache
└──────┬──────┘
       │
┌──────┴──────┐
│  L1 Cache   │  Per SM, configurable with shared memory
│ (~30 cycles)│  Hardware-managed
└──────┬──────┘
       │
┌──────┴──────┐
│  L2 Cache   │  Shared across ALL SMs
│(~200 cycles)│  40-50 MB on H100
└──────┬──────┘
       │
┌──────┴──────┐
│   Global    │  HBM (High Bandwidth Memory)
│   Memory    │  40-80 GB on modern GPUs
│(~400-600    │  3,350 GB/s bandwidth (H100)
│  cycles)    │  This is where model weights live!
└─────────────┘
```

**LLM Connection:**
- **Model weights** live in Global Memory (HBM) — this is why memory bandwidth matters
- **Activations during computation** should be in Shared Memory or Registers
- **KV-cache** (stored past attention keys/values) lives in Global Memory
- The art of GPU programming is moving data UP this hierarchy (closer to compute)

## 4.2 HBM: The GPU's Main Memory

HBM (High Bandwidth Memory) is physically STACKED on top of the GPU die:

```
    ┌──────────────────────────────────────┐
    │         GPU Package (top view)        │
    │                                      │
    │  ┌────┐  ┌────┐  ┌────────┐  ┌────┐ │
    │  │HBM │  │HBM │  │  GPU   │  │HBM │ │
    │  │ #1 │  │ #2 │  │  Die   │  │ #3 │ │
    │  │    │  │    │  │        │  │    │ │
    │  └────┘  └────┘  └────────┘  └────┘ │
    │  ┌────┐                      ┌────┐ │
    │  │HBM │                      │HBM │ │
    │  │ #4 │                      │ #5 │ │
    │  └────┘                      └────┘ │
    │                                      │
    └──────────────────────────────────────┘
```

Each HBM stack is made of layers of DRAM chips stacked vertically with thousands
of tiny wires (called Through-Silicon Vias / TSVs) connecting them. This gives
MASSIVE bandwidth because there are so many parallel data paths.

- HBM2e (A100): 2 TB/s bandwidth, up to 80 GB
- HBM3 (H100): 3.35 TB/s bandwidth, 80 GB
- HBM3e (B200): 8 TB/s bandwidth, up to 192 GB

Compare with CPU's DDR5: ~300 GB/s. HBM is 10-25x faster in bandwidth.

---

# PART 5: GPU DIE LAYOUT AND STREAMING MULTIPROCESSORS

## 5.1 What's Inside an NVIDIA GPU

Let's look at the H100 (current generation for AI) die layout:

```
┌────────────────────────────────────────────────────────────┐
│                        H100 GPU Die                         │
│                                                             │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │
│  │ SM 0 ││ SM 1 ││ SM 2 ││ SM 3 ││ SM 4 ││ SM 5 │ ...     │
│  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘         │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │
│  │ SM 6 ││ SM 7 ││ SM 8 ││ SM 9 ││SM 10 ││SM 11 │ ...     │
│  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘         │
│                      ...                                    │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐         │
│  │SM 126││SM 127││SM 128││SM 129││SM 130││SM 131│         │
│  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘         │
│                                                             │
│              ┌──────────────────────┐                       │
│              │     L2 Cache         │                       │
│              │     (50 MB)          │                       │
│              └──────────────────────┘                       │
│                                                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │ Memory │ │ Memory │ │  NVLink │ │ PCIe  │ │  GigaT  │   │
│  │ Ctrl 0 │ │ Ctrl 1 │ │ Ctrl   │ │ Gen5  │ │  Engine │   │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
│                                                             │
│  132 SMs × 128 FP32 cores = 16,896 CUDA cores              │
│  132 SMs × 4 Tensor Cores = 528 Tensor Cores               │
│  80 GB HBM3 @ 3,350 GB/s                                   │
└────────────────────────────────────────────────────────────┘
```

## 5.2 Inside One Streaming Multiprocessor (SM)

The SM is the fundamental building block. Here's what's inside ONE SM on H100:

```
┌─────────────────────────────────────────────────────────┐
│                    One SM (H100)                         │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 Warp Scheduler × 4                   │ │
│  │   (Each scheduler can issue 1 instruction/cycle)    │ │
│  │   (4 schedulers = 4 warps can progress per cycle)   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  FP32 CUDA Cores │  │  FP32 CUDA Cores │             │
│  │    (64 units)     │  │    (64 units)     │             │
│  └──────────────────┘  └──────────────────┘             │
│         = 128 FP32 CUDA Cores per SM                     │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  INT32 Units      │  │  FP64 Units      │             │
│  │    (64 units)     │  │    (2 units)      │             │
│  └──────────────────┘  └──────────────────┘             │
│                                                          │
│  ┌──────────────────────────────────────────┐            │
│  │  Tensor Cores × 4 (4th Generation)       │            │
│  │  Each does: 16×16 matrix multiply-add     │            │
│  │  Supports: FP16, BF16, TF32, FP8, INT8  │            │
│  │  THIS is what makes AI fast!              │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  SFU × 16        │  │  Load/Store × 32 │             │
│  │  (sin,cos,exp,   │  │  (memory access   │             │
│  │   sqrt,reciprocal)│  │   units)          │             │
│  └──────────────────┘  └──────────────────┘             │
│                                                          │
│  ┌──────────────────────────────────────────┐            │
│  │  Register File: 65,536 × 32-bit          │            │
│  │  = 256 KB of registers (FASTEST storage)  │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  ┌──────────────────────────────────────────┐            │
│  │  Shared Memory / L1 Cache: 228 KB         │            │
│  │  (Configurable split between the two)     │            │
│  └──────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## 5.3 Tensor Cores: The AI Accelerator

This is the most important hardware for LLMs. Regular CUDA cores do ONE
multiply per clock cycle. A Tensor Core does an ENTIRE MATRIX multiply-accumulate:

```
Regular CUDA Core (1 operation per cycle):
    a × b + c = d         (1 FMA = 2 FLOPs)

Tensor Core (one cycle):
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  4×4 Matrix │ × │  4×4 Matrix │ + │  4×4 Matrix │ = │  4×4 Matrix │
    │      A      │   │      B      │   │      C      │   │      D      │
    └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘

    D = A × B + C

    That's 4 × 4 × 4 = 64 multiply-adds = 128 FLOPs in ONE cycle!
    (Actually on H100 4th gen: 16×16 matrix at FP16)
```

On H100, each Tensor Core processes 16×16 matrices:
- 16 × 16 × 16 = 4,096 multiply-adds = 8,192 FLOPs per Tensor Core per cycle
- 4 Tensor Cores per SM × 132 SMs = 528 Tensor Cores
- At 1.83 GHz: 528 × 8,192 × 1.83 GHz = ~7.9 PFLOPS (FP16)

**This is WHY LLMs use FP16/BF16 for training and FP8 for inference:**
Tensor Cores only work with reduced precision. If you use FP32, you fall back
to regular CUDA cores and lose 10-16x performance.

---

# PART 6: THE NVIDIA GPU LINEAGE

Understanding the generations helps you understand what improved and why:

```
Architecture   Year   Key AI Innovation
──────────────────────────────────────────────────────────────
Tesla          2006   First CUDA-capable GPU. General purpose computing.
Fermi          2010   First GPU with L1/L2 caches, ECC memory.
Kepler         2012   Dynamic parallelism, Hyper-Q. AlexNet moment!
Maxwell        2014   Energy efficiency improvements.
Pascal         2016   First NVLink, HBM2. P100 for deep learning.
Volta          2017   FIRST TENSOR CORES (V100). Mixed precision training.
                      ← This is when GPU AI training exploded
Turing         2018   RT cores + 2nd gen Tensor Cores. INT8 inference.
Ampere         2020   3rd gen Tensor Cores, TF32, BF16, sparsity.
                      A100: the workhorse of LLM training (GPT-3, etc.)
Hopper         2022   4th gen Tensor Cores, FP8, Transformer Engine,
                      H100: built specifically for Transformers!
Blackwell      2024   5th gen Tensor Cores, FP4, 2nd gen Transformer
                      Engine, B200: 2.5x H100 for AI.
```

**Key inflection points:**
- **Volta (2017):** Tensor Cores invented → mixed precision training viable → LLM training becomes practical
- **Ampere (2020):** A100 becomes THE GPU for training LLMs (GPT-3 trained on thousands of A100s)
- **Hopper (2022):** Transformer Engine + FP8 → specifically designed for Transformer workloads
- **Blackwell (2024):** FP4 + massive memory → enables even larger models

---

# PART 7: CONNECTING IT ALL TO LLMs

## 7.1 What Happens When You Run `model.generate("Hello")`

Let's trace what physically happens in the hardware:

```
Step 1: Python calls model.generate("Hello")
        ↓
Step 2: PyTorch converts "Hello" to token IDs [15496]
        (this happens on CPU)
        ↓
Step 3: Token IDs are copied from CPU RAM to GPU HBM
        (cudaMemcpy over PCIe: ~12 GB/s for PCIe 4.0)
        ↓
Step 4: Embedding lookup on GPU
        - Thread reads token ID 15496 from HBM
        - Looks up row 15496 in embedding table (4096-dim vector)
        - This vector goes to registers/shared memory
        ↓
Step 5: For each of 32 Transformer layers:
        a) Attention:
           - Matrix multiply: input × W_Q, input × W_K, input × W_V
           - Each is a [4096 × 4096] × [4096 × 1] operation
           - Dispatched to Tensor Cores (FP16/BF16)
           - Thousands of threads across dozens of SMs
           - Result stays in GPU memory (no CPU round-trip!)
           
        b) Softmax(Q × K^T / sqrt(d)) × V
           - Computed entirely on GPU
           - Flash Attention keeps this in shared memory (SRAM)
           
        c) FFN: Two matrix multiplies
           - [4096 × 11008] and [11008 × 4096]
           - Again on Tensor Cores
        ↓
Step 6: Final linear layer [4096 × 32000]
        - Produces logits for ALL 32,000 vocabulary tokens
        - On Tensor Cores
        ↓
Step 7: Softmax + sampling (temperature, top-k)
        - Simple operations on GPU
        - Select one token ID
        ↓
Step 8: Copy generated token ID back to CPU
        - cudaMemcpy: just 4 bytes
        ↓
Step 9: Decode token ID to text, e.g., " World"
        - On CPU
        ↓
Step 10: Repeat from Step 4 for next token
         (but now using KV-cache to avoid recomputing
          attention for all previous tokens)
```

**Key insight:** Steps 4-7 happen ENTIRELY on the GPU. The data never leaves
GPU memory between layers. The CPU just launches the work and waits.

## 7.2 Why This Architecture Understanding Matters

Knowing this, you can now understand:

1. **Why quantization helps:** Smaller weights = less data to read from HBM = faster
2. **Why batching helps:** Read weights once, use for N sequences = higher arithmetic intensity
3. **Why KV-cache matters:** Without it, you'd recompute attention for ALL past tokens every step
4. **Why Flash Attention matters:** Keeps attention computation in SRAM instead of reading/writing HBM
5. **Why Tensor Cores matter:** 10-16x more FLOPS than regular CUDA cores for matrix multiply
6. **Why model parallelism matters:** One GPU has 80GB, a 70B FP16 model needs 140GB → split across GPUs
7. **Why NVLink matters:** GPUs need to communicate during model parallelism, PCIe is too slow

---

# PART 8: LLM DEEP DIVE — Every Operation Mapped to Hardware

This is the most important part. Let's walk through EXACTLY what an LLM does
and map EVERY operation to the hardware concepts you just learned.

## 8.1 What IS an LLM, Physically?

An LLM is just a HUGE collection of numbers (called "parameters" or "weights")
stored in GPU memory, plus a fixed set of mathematical operations to apply to them.

```
LLaMA-7B "physically" is:
├── Embedding table:     32,000 × 4,096 = 131 million numbers
├── 32 Transformer layers, each containing:
│   ├── Attention weights:
│   │   ├── W_Q:  4,096 × 4,096 = 16.8 million numbers
│   │   ├── W_K:  4,096 × 4,096 = 16.8 million numbers
│   │   ├── W_V:  4,096 × 4,096 = 16.8 million numbers
│   │   └── W_O:  4,096 × 4,096 = 16.8 million numbers
│   ├── FFN weights:
│   │   ├── W_up:   4,096 × 11,008 = 45 million numbers
│   │   ├── W_gate: 4,096 × 11,008 = 45 million numbers
│   │   └── W_down: 11,008 × 4,096 = 45 million numbers
│   └── Normalization: 4,096 numbers (tiny)
└── Output projection:   4,096 × 32,000 = 131 million numbers

Total: ~6.7 billion numbers
At FP16 (2 bytes each): 6.7B × 2 = ~13.4 GB sitting in GPU HBM
At INT4 (0.5 bytes each): 6.7B × 0.5 = ~3.4 GB (this is quantization!)
```

**That's it.** An LLM is 13.4 GB of numbers in GPU memory. There is no "intelligence"
stored anywhere — the intelligence emerges from applying matrix multiplications
to these numbers in a specific order.

## 8.2 The Life of a Single Token — Complete Hardware Trace

Let's say you type "The capital of France is" and the model needs to predict "Paris".

### Step 1: Tokenization (CPU, no GPU)
```
Input text: "The capital of France is"
                ↓ (tokenizer on CPU)
Token IDs:  [450, 7483, 310, 3444, 338]
            "The" "capital" "of" "France" "is"

Each token is just an integer. This is a lookup in a vocabulary table.
The tokenizer runs on CPU because it's sequential text processing.
```

**Hardware:** CPU only. RAM only. No GPU involved yet.

### Step 2: Token IDs → GPU (PCIe Transfer)
```
CPU RAM:  [450, 7483, 310, 3444, 338]  (5 integers = 20 bytes)
                    ↓ cudaMemcpy (PCIe bus)
GPU HBM:  [450, 7483, 310, 3444, 338]

Transfer: 20 bytes over PCIe Gen4 (32 GB/s) = essentially instant
```

**Hardware:** PCIe bus. This is a tiny transfer. For a batch of 64 sequences
of 2048 tokens each, it'd be 64 × 2048 × 4 = 512 KB — still trivial.

### Step 3: Embedding Lookup (GPU, Memory-Bound)
```
Embedding table in GPU HBM:
┌─────────────────────────────────────────┐
│ Row 0:   [0.012, -0.034, 0.091, ...]   │  ← 4096 numbers
│ Row 1:   [0.045, 0.023, -0.067, ...]   │
│ ...                                      │
│ Row 450: [0.078, -0.012, 0.045, ...]   │  ← "The"
│ ...                                      │
│ Row 7483:[0.034, 0.089, -0.023, ...]   │  ← "capital"
│ ...                                      │
│ Row 32000: [...]                         │
└─────────────────────────────────────────┘

For token "The" (ID 450):
  → Read row 450 from embedding table
  → Get a vector of 4096 floating-point numbers
  → This vector IS the representation of "The"
```

**Hardware operation:** This is a MEMORY READ. No math, just looking up rows.
- GPU launches threads: each thread reads one element of the embedding vector
- 5 tokens × 4096 elements = 20,480 reads from HBM
- At 2 bytes each (FP16): 40 KB of data read
- This is extremely memory-bound (no compute, just reads)
- Takes: ~0.001 ms (negligible)

**LLM insight:** The embedding table is essentially a dictionary mapping token IDs
to their "meaning" represented as a 4096-dimensional vector. The model LEARNED these
vectors during pre-training — "king" and "queen" end up with similar vectors because
they appeared in similar contexts.

### Step 4: Positional Encoding (GPU, Compute-Light)
```
The embedding tells the model WHAT each token is.
But Transformers process all tokens in parallel, so they don't know
the ORDER of tokens. "cat sat on mat" and "mat on sat cat" would
look the same!

Solution: Add position information to each embedding.

Modern LLMs use RoPE (Rotary Position Embedding):
  → Apply a rotation to the embedding based on position
  → Position 0 gets rotation angle 0
  → Position 1 gets rotation angle θ
  → Position 2 gets rotation angle 2θ
  → etc.

Mathematically: multiply pairs of embedding elements by sin/cos of position

For each token embedding vector (4096 numbers):
  → Apply element-wise sin/cos multiplications
  → 4096 multiplications + 4096 additions per token
  → 5 tokens × 8192 operations = ~40,000 FLOPs
```

**Hardware:** Runs on regular CUDA cores (not Tensor Cores — too small for matrix multiply).
SFU (Special Function Units) compute the sin/cos values.
This is negligible compute — less than 0.001% of total.

### Step 5: Self-Attention — THE KEY OPERATION (GPU, Compute + Memory Heavy)

This is where the magic happens. This is HOW the model understands context.

```
INPUT: 5 token embeddings, each a vector of 4096 numbers
       Arranged as a matrix X of shape [5 × 4096]

STEP 5a: Create Queries, Keys, and Values
─────────────────────────────────────────
Q = X × W_Q    →  [5 × 4096] × [4096 × 4096] = [5 × 4096]
K = X × W_K    →  [5 × 4096] × [4096 × 4096] = [5 × 4096]
V = X × W_V    →  [5 × 4096] × [4096 × 4096] = [5 × 4096]

Each is a MATRIX MULTIPLICATION.
```

**Hardware for Q = X × W_Q:**
```
Matrix multiply [5 × 4096] × [4096 × 4096]:
  FLOPs = 2 × 5 × 4096 × 4096 = 167 million operations

  On GPU:
  1. W_Q (the weight matrix) is 4096×4096×2 bytes = 32 MB in HBM
  2. GPU reads W_Q from HBM → streams through L2 cache → SMs
  3. Tensor Cores on each SM compute 16×16 tile multiply-accumulates
  4. Thousands of threads work on different tiles simultaneously
  5. Result (Q matrix) is written back to HBM

  Time on H100 at 990 TFLOPS: 167M / 990T = 0.0002 ms (compute)
  Time to read W_Q from HBM: 32MB / 3350 GB/s = 0.01 ms (memory)

  MEMORY IS 50x SLOWER THAN COMPUTE! This is memory-bound.
```

**Why this matters for LLM optimization:**
- Quantizing W_Q from FP16 to INT4 makes it 4x smaller (8 MB instead of 32 MB)
- Reading 8 MB instead of 32 MB → 4x faster inference for this operation
- The compute barely changes because we dequantize on-the-fly
- THIS IS WHY QUANTIZATION WORKS. The compute was never the bottleneck.

We do this 3 times (Q, K, V), so 3 × 32 MB = 96 MB of weight reads per attention layer.

```
STEP 5b: Compute Attention Scores
───────────────────────────────────
scores = Q × K^T  →  [5 × 4096] × [4096 × 5] = [5 × 5]

This gives a 5×5 matrix where entry (i,j) means:
"How much should token i pay attention to token j?"

For our input "The capital of France is":
                    The  capital  of   France  is
          The     [ 0.1   0.2    0.0   0.1    0.0 ]
          capital [ 0.1   0.3    0.1   0.4    0.0 ]
scores =  of      [ 0.0   0.1    0.1   0.3    0.0 ]
          France  [ 0.1   0.2    0.1   0.3    0.1 ]
          is      [ 0.1   0.3    0.1   0.8    0.2 ]
                                       ↑
                                 "is" pays HEAVY attention
                                 to "France" — it's figuring
                                 out "is ___" needs info
                                 about France!
```

**Hardware:** This is a SMALL matrix multiply (5×5 output). Very fast.
In practice with batching (64 sequences), it's [64 × 32_heads × 5 × 5] — still small.

**LLM insight:** This attention matrix IS the model "thinking". When the model
generates "Paris", it's because the "is" token's attention vector has a huge weight
on "France" and "capital". The attention mechanism lets each token look at ALL
previous tokens and decide which are relevant.

```
STEP 5c: Apply Causal Mask
────────────────────────────
For autoregressive generation, each token can ONLY attend to
tokens BEFORE it (and itself). Future tokens are masked with -infinity.

                    The  capital  of   France  is
          The     [ 0.1   -∞      -∞    -∞     -∞  ]
          capital [ 0.1   0.3     -∞    -∞     -∞  ]
masked =  of      [ 0.0   0.1    0.1    -∞     -∞  ]
          France  [ 0.1   0.2    0.1   0.3     -∞  ]
          is      [ 0.1   0.3    0.1   0.8    0.2  ]

After softmax, -∞ becomes 0 (zero attention to future tokens).
```

**Hardware:** Element-wise comparison and assignment. Runs on CUDA cores.
Negligible compute.

**LLM insight:** This causal mask is WHY GPT-style models can generate text
left-to-right. Each position only sees the past, so at generation time,
adding one new token doesn't affect the computations for previous tokens.
This is what enables the KV-cache optimization (you'll learn in Month 3).

```
STEP 5d: Softmax
────────────────
Turn raw scores into probabilities (each row sums to 1.0):

                    The  capital  of   France  is
          The     [ 1.0   0.0    0.0   0.0    0.0 ]
          capital [ 0.35  0.65   0.0   0.0    0.0 ]
weights = of      [ 0.25  0.30   0.45  0.0    0.0 ]
          France  [ 0.18  0.22   0.18  0.42   0.0 ]
          is      [ 0.08  0.12   0.08  0.60   0.12 ]
                                       ↑
                                 "is" gives 60% of its
                                 attention to "France"!
```

**Hardware:** Softmax requires: subtract max (for numerical stability), exponentiate
each element (uses SFU for exp()), sum all elements, divide each by sum.
This is a REDUCTION operation (summing across elements) — you'll implement
parallel reduction in Week 3!

```
STEP 5e: Weighted Sum of Values
────────────────────────────────
output = weights × V  →  [5 × 5] × [5 × 4096] = [5 × 4096]

For the "is" token, its output is:
  0.08 × V("The") + 0.12 × V("capital") + 0.08 × V("of")
  + 0.60 × V("France") + 0.12 × V("is")

The output is DOMINATED by V("France") because the attention weight is 0.60.
The model has effectively "retrieved" the information about France
and packed it into the representation of the "is" token.
```

**Hardware:** Another matrix multiply → Tensor Cores.

```
STEP 5f: Output Projection
───────────────────────────
result = output × W_O  →  [5 × 4096] × [4096 × 4096] = [5 × 4096]

Same size matrix multiply as Q, K, V projections.
32 MB weight read from HBM, Tensor Cores compute.
```

**Total attention layer hardware cost:**
```
Weight reads from HBM:
  W_Q (32MB) + W_K (32MB) + W_V (32MB) + W_O (32MB) = 128 MB

Compute:
  4 matrix multiplies of [5 × 4096] × [4096 × 4096]
  + attention score computation
  + softmax
  = ~670 million FLOPs

Time bottleneck: Reading 128 MB at 3350 GB/s = 0.04 ms
                 Computing 670M FLOPs at 990 TFLOPS = 0.0007 ms

  → 57x more time spent on MEMORY than COMPUTE (for 1 sequence)
  → With batch of 64: compute goes up 64x, memory stays same
  → At batch=64: memory 0.04ms, compute 0.045ms → BALANCED!
  → This is why BATCHING is critical for GPU utilization
```

### Step 6: Feed-Forward Network (GPU, Same Pattern)
```
After attention, each token passes through a feed-forward network.
In LLaMA this is "SwiGLU" (fancy name, just matrix multiplies + activation):

up    = input × W_up     →  [5 × 4096] × [4096 × 11008] = [5 × 11008]
gate  = input × W_gate   →  [5 × 4096] × [4096 × 11008] = [5 × 11008]
hidden = up * SiLU(gate)  →  element-wise multiply
output = hidden × W_down →  [5 × 11008] × [11008 × 4096] = [5 × 4096]
```

**Hardware:**
```
Weight reads: W_up (86MB) + W_gate (86MB) + W_down (86MB) = 258 MB
This is 2x MORE memory than attention!
The FFN is actually the DOMINANT cost in each Transformer layer.

This is why:
- LLaMA-7B has hidden_dim=4096 but FFN inner dim=11008 (2.7x larger)
- Most model parameters are in the FFN weights
- Quantizing FFN weights gives the biggest speedup
```

### Step 7: Repeat 32 Times (32 Transformer Layers)
```
Total for all 32 layers:
  Weight reads: 32 × (128 MB attention + 258 MB FFN) = 12,352 MB = ~12 GB
  
  Wait — the ENTIRE model is ~13.4 GB. So yes, generating ONE token
  requires reading essentially THE ENTIRE MODEL from memory once.
  
  At 3350 GB/s: 13.4 GB / 3350 = 4 ms per token
  
  That's ~250 tokens/second for a single sequence.
  ChatGPT generates at ~50-80 tokens/second — makes sense because
  it uses larger models and has overhead.
```

### Step 8: Predict Next Token (GPU → CPU)
```
After 32 layers, we have a vector of 4096 numbers for the "is" position.
This vector now encodes: "I've seen 'The capital of France is' and
I know what should come next."

Final linear layer:
  logits = output × W_vocab  →  [4096] × [4096 × 32000] = [32000]
  
  This gives a score for EVERY word in the vocabulary.
  
  logits = [..., -2.1, ..., 8.7, ..., -1.3, ...]
                              ↑
                         Token 3681 = "Paris" has the highest score!
  
  Apply softmax → probabilities:
  P("Paris") = 0.82
  P("Lyon")  = 0.03
  P("Berlin")= 0.01
  ...
  
  Sample or take argmax → generate "Paris"
```

**Hardware:** One more matrix multiply (Tensor Cores), then softmax (CUDA cores + SFU).

### Step 9: The Token Generation Loop
```
Now we have "The capital of France is Paris"
To generate the NEXT token, we repeat everything BUT:
- We DON'T recompute attention for "The capital of France is"
  (already computed, stored in KV-cache in GPU HBM)
- We ONLY compute the new token "Paris" through all 32 layers
- We attend to ALL previous tokens' K,V (read from KV-cache)

KV-cache size for LLaMA-7B at sequence length 2048:
  2 (K and V) × 32 (layers) × 2048 (tokens) × 4096 (dim) × 2 (FP16 bytes)
  = 1,073,741,824 bytes = ~1 GB per sequence!
  
  For 64 concurrent users: 64 GB of KV-cache alone!
  This is why GPU memory (80GB) is so critical for serving.
  This is why PagedAttention (vLLM) was invented — to manage
  this memory efficiently.
```

## 8.3 Training vs Inference — What's Different on GPU?

```
                        INFERENCE               TRAINING
                        (generating text)       (learning weights)
────────────────────────────────────────────────────────────────────
Batch size              1-64 sequences          256-4096 sequences
Sequence processing     One token at a time     All tokens at once
Matrix multiply shape   [B × 4096] × [4096²]   [B×L × 4096] × [4096²]
                        (B=batch, thin)         (B×L=huge, tall)
Arithmetic intensity    LOW (memory-bound)      HIGH (compute-bound)
GPU bottleneck          Memory bandwidth        Compute (TFLOPS)
Key optimization        Quantize, batch more    Use Tensor Cores, AMP
Tensor Core utilization ~5-30%                  ~60-80%
Memory concern          KV-cache size           Activation memory
                                                + gradient storage
                                                + optimizer states

Why training uses SO much more memory:
  Model weights:    13.4 GB (same)
  Gradients:        13.4 GB (same size as weights)
  Optimizer states: 26.8 GB (Adam keeps 2 copies: momentum + variance)
  Activations:      Variable, can be 10-50+ GB depending on batch/seq
  
  Total for LLaMA-7B training: ~60-100 GB minimum!
  This is why you need A100 80GB or H100 80GB for training.
  This is why ZeRO/FSDP shard across multiple GPUs.
```

## 8.4 Real-World LLM Serving: What Happens at Scale

When you use ChatGPT, here's the actual infrastructure:

```
User: "The capital of France is"
       │
       ↓
┌──────────────────┐
│   Load Balancer  │  Routes request to available GPU server
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Tokenizer (CPU) │  Convert text → token IDs
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Inference Server │  vLLM / TensorRT-LLM / NIM
│                  │
│  ┌────────────┐  │  CONTINUOUS BATCHING:
│  │ Request #1 │  │  Don't wait for all requests to finish.
│  │ Request #2 │  │  As soon as one request finishes,
│  │ Request #3 │  │  add a new one to the batch.
│  │ ...        │  │  This keeps GPU utilization high.
│  └────────────┘  │
│                  │
│  ┌────────────┐  │  PAGED ATTENTION (vLLM):
│  │ KV-Cache   │  │  KV-cache is stored in PAGES (blocks)
│  │ Block Pool │  │  just like virtual memory in an OS.
│  │            │  │  Avoids memory fragmentation.
│  └────────────┘  │  One request can have non-contiguous blocks.
│                  │
│  ┌────────────┐  │  SPECULATIVE DECODING:
│  │ Small model│  │  A small fast model guesses next 4-5 tokens.
│  │ (draft)    │  │  The big model verifies all at once (parallel).
│  └────────────┘  │  If guesses are right → 4-5x faster!
│                  │
│  GPU: H100 or   │
│  multiple GPUs  │
│  with NVLink    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Detokenizer(CPU) │  Token IDs → text
└────────┬─────────┘
         ↓
User sees: "Paris"   (streamed token by token)
```

**Why you need to know all this:**
Every component in this diagram maps to something in your roadmap:
- Load balancer → Month 10 (production systems)
- Tokenizer → Month 1 Week 4 (you'll build one from scratch)
- Continuous batching → Month 4 Week 15
- PagedAttention → Month 4 Week 15
- Speculative decoding → Month 4 Week 15
- KV-cache → Month 3 Week 13
- NVLink multi-GPU → Month 3 Week 9
- TensorRT-LLM/NIM → Month 4-5

## 8.5 The Optimization Stack — Everything Connects

Here's the complete picture of WHY each optimization exists, mapped to hardware:

```
PROBLEM                          SOLUTION                    HARDWARE ROOT CAUSE
─────────────────────────────────────────────────────────────────────────────────
Model too large for 1 GPU        Model parallelism           GPU has limited HBM (80GB)
                                 (split across GPUs)         NVLink enables fast comm

Inference too slow               Quantization (INT4/FP8)     Memory bandwidth bottleneck
(reading weights from HBM)                                   Smaller weights = less reading

GPU underutilized during         Batching (process many      Low arithmetic intensity
single-sequence inference        sequences at once)          for single sequence

Attention memory grows O(N²)     Flash Attention             HBM read/write is slow
with sequence length                                         Keep data in SRAM (shared mem)

KV-cache wastes memory           PagedAttention (vLLM)       GPU memory is precious
(fragmentation)                                              and limited

Autoregressive generation        Speculative decoding        GPU sits idle during
is serial (1 token at a time)                                single-token generation

Training needs more memory       Gradient checkpointing      Recompute vs store tradeoff
than available                   Mixed precision (FP16)      FP16 = half the memory
                                 ZeRO/FSDP sharding          Distribute across GPUs

Training too slow                Tensor Cores (FP16/BF16)    16x faster than FP32 cores
                                 Data parallelism             Use more GPUs

Want better model but can't      LoRA/QLoRA                  Full fine-tuning needs
afford full training                                         3-4x model size in memory

Model hallucinates / wrong       RAG (retrieve documents)    Model weights are fixed
knowledge cutoff                                             at training time
```

Every single row in this table will become a skill you master over the next 18 months.
Day 1 is about understanding the ROOT CAUSE column — the hardware constraints that
make all these optimizations necessary.

---

# PART 9: SELF-TEST (15 Questions)

Close the document, then try to answer from memory. Check your answers after.

**Hardware fundamentals:**

1. **What is the main design difference between CPU and GPU?**
   → CPU: few complex cores optimized for single-thread latency.
   GPU: thousands of simple cores optimized for total throughput.

2. **What % of CPU die is ALUs vs GPU die?**
   → CPU: ~5% ALUs. GPU: ~80% ALUs. The rest on CPU is cache and control logic.

3. **What is SIMT? What is a warp?**
   → SIMT: Single Instruction, Multiple Threads. All 32 threads in a warp execute
   the same instruction on different data. Warp = 32 threads, the fundamental
   execution unit on NVIDIA GPUs.

4. **How does a GPU hide memory latency?**
   → By keeping hundreds of warps ready. When one warp waits for memory, the GPU
   instantly switches to another warp (zero-cost context switch because each warp
   has its own registers). This is why you need massive parallelism.

5. **What are the 5 levels of GPU memory and their approximate latencies?**
   → Registers (~1 cycle) → Shared Memory (~20-30 cycles) → L1 Cache (~30 cycles)
   → L2 Cache (~200 cycles) → Global/HBM (~400-600 cycles).

6. **What do Tensor Cores do that regular CUDA cores cannot?**
   → Matrix multiply-accumulate on 16×16 tiles in a single cycle = 8,192 FLOPs,
   vs regular CUDA core = 2 FLOPs per cycle. 10-16x more throughput. Only works
   with reduced precision (FP16, BF16, FP8, INT8).

**LLM-specific:**

7. **How large is LLaMA-7B in GPU memory at FP16?**
   → 6.7 billion parameters × 2 bytes = ~13.4 GB.

8. **Why is LLM inference memory-bandwidth-bound, not compute-bound?**
   → For each token, you read the entire model (~13.4 GB) from HBM but only do
   ~14 billion FLOPs. At 3350 GB/s bandwidth, reading takes 4ms. At 990 TFLOPS,
   compute takes 0.014ms. Memory is ~300x slower. GPU is idle most of the time.

9. **Why does quantization (FP16→INT4) make inference ~4x faster?**
   → Inference bottleneck is reading weights from HBM. INT4 is 4x smaller than FP16.
   4x less data to read = 4x faster. Compute cost barely changes.

10. **Why does batching improve GPU utilization?**
    → With batch=1, you read weights once and do little compute (low arithmetic
    intensity = memory-bound). With batch=64, you read weights once and do 64x
    more compute. This increases arithmetic intensity toward compute-bound,
    where the GPU's TFLOPS actually get used.

11. **What is KV-cache and why does it matter?**
    → During generation, we store the Key and Value matrices from attention for all
    previous tokens. Without it, we'd recompute attention for ALL past tokens at
    every step (quadratic cost). With it, each new token only computes its own Q
    and attends to cached K,V. Cost: ~1 GB per sequence for 7B model at length 2048.

12. **Why is the FFN (feed-forward network) more expensive than attention?**
    → FFN has weight matrices of size [4096 × 11008] (3 of them = 258 MB).
    Attention has [4096 × 4096] (4 of them = 128 MB). FFN reads 2x more weight data.

13. **Why does training need so much more memory than inference?**
    → Inference: just model weights (~13.4 GB). Training: weights (13.4 GB) + gradients
    (13.4 GB) + optimizer states (26.8 GB for Adam) + activations (10-50+ GB).
    Total: 60-100+ GB vs 13.4 GB.

14. **What is Flash Attention solving at the hardware level?**
    → Standard attention writes the N×N attention matrix to HBM (slow). Flash Attention
    computes attention in tiles that fit in shared memory (fast SRAM), never
    materializing the full N×N matrix. Saves HBM bandwidth → faster.

15. **In one sentence, why does NVIDIA dominate AI?**
    → Tensor Cores give 10-16x more compute for matrix operations (the core LLM
    operation), HBM gives 10x more memory bandwidth than CPUs, and CUDA gives
    programmability — all in one chip with a mature software ecosystem.

---

# PART 10: CODE EXERCISE (Do after landing, with a computer)

Since you can't code on the plane, READ this code and understand it.
After landing, TYPE it yourself and run it.

## Exercise 1: CPU Matrix Multiply in Python (understand the bottleneck)

```python
import numpy as np
import time

def cpu_matmul(A, B):
    """Naive CPU matrix multiply — O(N³) operations"""
    N = A.shape[0]
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Small test
N = 512
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

# Naive Python (VERY slow)
start = time.time()
C_naive = cpu_matmul(A, B)
naive_time = time.time() - start
naive_flops = 2 * N**3 / naive_time / 1e9  # GFLOPS
print(f"Naive Python: {naive_time:.2f}s, {naive_flops:.2f} GFLOPS")

# NumPy (uses optimized BLAS library, multi-threaded)
start = time.time()
C_numpy = A @ B
numpy_time = time.time() - start
numpy_flops = 2 * N**3 / numpy_time / 1e9
print(f"NumPy (BLAS): {numpy_time:.4f}s, {numpy_flops:.2f} GFLOPS")

# Compare
print(f"NumPy is {naive_time/numpy_time:.0f}x faster than naive Python")
print(f"For reference: H100 FP32 = 67,000 GFLOPS")
print(f"For reference: H100 FP16 Tensor Core = 990,000 GFLOPS")
```

## Exercise 2: PyTorch GPU vs CPU (do this on Google Colab)

```python
import torch
import time

N = 4096  # Same size as LLaMA hidden dimension!

# Create random matrices
A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

# CPU timing
start = time.time()
for _ in range(10):
    C_cpu = A_cpu @ B_cpu
cpu_time = (time.time() - start) / 10

# GPU timing (on Colab, select GPU runtime)
A_gpu = A_cpu.cuda()
B_gpu = B_cpu.cuda()

# Warmup
for _ in range(3):
    _ = A_gpu @ B_gpu
torch.cuda.synchronize()

start = time.time()
for _ in range(10):
    C_gpu = A_gpu @ B_gpu
torch.cuda.synchronize()
gpu_time = (time.time() - start) / 10

flops = 2 * N**3
print(f"CPU: {cpu_time*1000:.2f} ms, {flops/cpu_time/1e12:.2f} TFLOPS")
print(f"GPU: {gpu_time*1000:.2f} ms, {flops/gpu_time/1e12:.2f} TFLOPS")
print(f"GPU is {cpu_time/gpu_time:.1f}x faster")
print(f"\nThis {N}x{N} matmul is EXACTLY what happens")
print(f"inside every Transformer layer for Q, K, V projections!")
```

---

# WHAT TO DO NEXT

After this Day 1 study:
- [x] You understand CPU vs GPU architecture at the hardware level
- [x] You can explain SIMT, warps, and the thread hierarchy
- [x] You know why GPUs are faster for LLMs (parallel matmul, Tensor Cores)
- [x] You understand the memory bandwidth bottleneck
- [x] You know the GPU memory hierarchy and why it matters
- [ ] You can trace what happens in hardware when an LLM generates a token *(will click fully in Week 5)*

**Completed: March 13, 2026 (Pune → Kolkata flight)**

**Day 2** will go DEEPER into the SM: exactly how Tensor Cores work,
what happens during a warp schedule cycle, register allocation, and
occupancy. Day 1 gives you the big picture; Day 2 gives you the internals.

---

*Status: ✅ COMPLETED*
*Code exercises: pending (do on Google Colab after landing)*
