# Week 2, Day 1: CUDA Global Memory Deep Dive
## Why Your Kernel Is Fast or Slow — It's Almost Always Memory

Last week you wrote CUDA kernels and saw they hit ~300 GB/s on a T4 (peak ~320).
Today you learn WHY. You'll understand exactly what happens when a warp reads from
GPU memory, why coalesced access is 10-32x faster than random access, and you'll
measure it yourself.

**Time: ~2.5-3 hours**
**Setup: Google Colab with GPU runtime**

---

# PART 0: NEW TERMS FOR TODAY

```
GLOBAL MEMORY   = The main GPU memory (HBM). Largest, slowest. 16-80 GB.
                  Where your weights, KV-cache, activations live.

CACHE LINE      = Minimum unit of data the GPU reads from HBM.
                  Not 1 byte or 4 bytes — a whole CHUNK of 32 or 128 bytes.
                  Like how trucks deliver in crates, not individual items.

MEMORY          = A single read/write request to HBM.
TRANSACTION       Limited to multiples of 32 bytes.
                  Uncoalesced = many small transactions = slow.
                  Coalesced = one big transaction = fast.

COALESCED       = When threads in a warp access consecutive memory addresses,
ACCESS            the GPU combines them into ONE memory transaction.
                  32 threads × 4 bytes = 128 bytes = 1 transaction. ✓

UNCOALESCED     = Threads access scattered addresses, GPU must issue
ACCESS            MANY transactions. Can be 2x to 32x slower.

ALIGNED ACCESS  = Memory address is a multiple of the access size.
                  float access: address divisible by 4.
                  float4 access: address divisible by 16.
                  Misaligned = GPU issues extra transactions.

LATENCY         = Time from requesting data to receiving it.
                  HBM latency: ~400-600 CLOCK CYCLES (~200-300 nanoseconds).
                  Compare: Register access = 1 cycle.

BANDWIDTH       = Total bytes per second the GPU can move from HBM.
                  T4: ~320 GB/s theoretical.
                  A100: ~1,555 GB/s.
                  H100: ~3,350 GB/s.

HBM             = High Bandwidth Memory. The physical memory chips stacked
                  next to the GPU die. Connected via hundreds of wires
                  (not like CPU DDR5 which has ~128 wires).

DDR5            = CPU's main memory type. ~50-100 GB/s bandwidth.
                  30-50x slower than HBM.

STRIDE          = Distance between consecutive memory accesses.
                  stride=1: A[0], A[1], A[2] (consecutive, fast)
                  stride=32: A[0], A[32], A[64] (scattered, slow)
```

---

# PART 1: THE GPU MEMORY HIERARCHY (RECAP + EXPANSION)

You saw this in Day 2 of Week 1. Today we dive into the global memory layer:

```
                     ACCESS TIME      SIZE           WHO CAN SEE IT
─────────────────────────────────────────────────────────────────────
Register            1 cycle          256 KB/SM      Single thread
                                     (65K × 4 bytes)
  ↓
Shared Memory       ~20 cycles       228 KB/SM      All threads in block
(programmer-         (fast SRAM)
 controlled)
  ↓
L1 Cache            ~20 cycles       shared w/ SMEM Single SM
(hardware cache)
  ↓
L2 Cache            ~200 cycles      50 MB          All SMs
                                     (H100: 50MB)
                                     (T4: 4MB)
  ↓
GLOBAL MEMORY       ~400-600 cycles  16-80 GB       Everyone
(HBM)               ← TODAY'S FOCUS
```

**Why global memory dominates everything:**
```
Your LLaMA-7B model (13.4 GB of weights) CAN'T fit in:
  - Registers (256 KB) ❌
  - Shared memory (228 KB) ❌
  - L2 cache (50 MB on H100) ❌ — still 260x too small!

It MUST live in global memory (HBM).
Every token generated requires reading ALL 13.4 GB.
That's why inference = memory-bandwidth-bound.

Make HBM access efficient = faster models.
Make HBM access wasteful = slower models.
Today you learn the difference.
```

---

# PART 2: HOW GLOBAL MEMORY PHYSICALLY WORKS

## 2.1 HBM — Stacked Memory Chips

```
    ┌─────────────────────────────────────────┐
    │           GPU Package (top view)         │
    │                                          │
    │  ┌────┐  ┌────┐  ┌────────┐  ┌────┐     │
    │  │HBM │  │HBM │  │  GPU   │  │HBM │     │
    │  │ #1 │  │ #2 │  │  Die   │  │ #3 │     │
    │  │5GB │  │5GB │  │        │  │5GB │     │
    │  └────┘  └────┘  └────────┘  └────┘     │
    │     │       │       ↑          │        │
    │     └───────┼───────┼──────────┘        │
    │             │       │                   │
    │        Memory Controllers on GPU die    │
    │        (thousands of parallel wires)    │
    └─────────────────────────────────────────┘

Each HBM stack = 4-12 DRAM chips stacked VERTICALLY
Connected via Through-Silicon Vias (TSVs) — thousands of tiny wires.

This is why HBM is so fast:
  DDR5 (CPU): 128 wires, narrow bus
  HBM3 (GPU): 1024 wires per stack × 5 stacks = 5120 wires
  
  More wires = more parallel data transfer = more bandwidth.
```

## 2.2 Memory Transactions: The Unit of Work

**Critical insight:** The GPU CAN'T read 1 byte from HBM. The smallest unit is a transaction.

```
TRANSACTION SIZE:
  L1 cache line: 128 bytes
  L2 cache line: 32 bytes

When a warp (32 threads) wants to read from HBM:
  The GPU checks WHICH ADDRESSES the 32 threads need.
  Groups them into 32-byte or 128-byte segments.
  Issues ONE transaction per unique segment.

Best case (COALESCED):
  32 threads read addresses 0, 4, 8, ..., 124 (all within one 128-byte line)
  → 1 transaction of 128 bytes serves all 32 threads
  → Each thread gets its 4 bytes at full speed

Worst case (UNCOALESCED):
  32 threads read addresses 0, 1024, 2048, ..., all scattered
  → 32 separate 32-byte transactions (1024 bytes total)
  → 8x more data fetched than needed!
  → Bandwidth wasted.
```

## 2.3 Cache Lines — The Key Unit

```
ALL HBM reads go through L2, then to L1 (or shared memory).
L2 operates in 32-byte chunks. L1 in 128-byte chunks.

When you access memory[100]:
  GPU doesn't read 4 bytes from HBM.
  It fetches the 128-byte cache line containing address 100.
  That's memory[96] through memory[127] (if 4-byte aligned floats).

Implication: if your warp's NEXT access is memory[104],
  it's ALREADY IN CACHE. Free.

This is why consecutive access is fast — you pay for the cache line once,
then get 31 more reads "for free" until you cross to the next cache line.
```

---

# PART 3: COALESCED vs UNCOALESCED ACCESS

This is the #1 performance concept in CUDA. Master it and you understand 80% of GPU optimization.

## 3.1 Coalesced Access — The Fast Case

```
Thread assignments within a warp (32 threads):
  Thread 0  reads A[0]
  Thread 1  reads A[1]
  Thread 2  reads A[2]
  ...
  Thread 31 reads A[31]

Addresses (assuming float = 4 bytes, A starts at address 0):
  Thread 0:  addr 0
  Thread 1:  addr 4
  Thread 2:  addr 8
  ...
  Thread 31: addr 124

Total span: 0 to 127 = exactly ONE 128-byte cache line.

GPU issues: 1 memory transaction.
Bytes fetched: 128.
Bytes used:   128.
Efficiency:   100% ✓

This is the ideal access pattern. This is what vec_add does.
```

Visually:
```
Memory:    [0][1][2][3][4][5][6][7] ... [29][30][31]
Threads:    T0 T1 T2 T3 T4 T5 T6 T7     T29 T30 T31
                              ↓
           ONE transaction fetches 128 bytes.
           All 32 threads served.
```

## 3.2 Uncoalesced Access — The Slow Case

### Case A: Strided access (every 32nd element)
```
Thread 0  reads A[0]    (addr 0)
Thread 1  reads A[32]   (addr 128)
Thread 2  reads A[64]   (addr 256)
...
Thread 31 reads A[992]  (addr 3968)

Each thread's address is in a DIFFERENT cache line!
GPU issues: 32 separate 128-byte transactions.
Bytes fetched: 32 × 128 = 4096 bytes.
Bytes used:   32 × 4 = 128 bytes.
Efficiency:   128 / 4096 = 3.125% ← wasted 97% of bandwidth!

Kernel runs ~32x slower due to this alone.
```

Visually:
```
Cache line 0:   [T0][ ][ ][ ] ... [ ][ ][ ][ ]   ← only 1 byte needed from here
Cache line 1:   [T1][ ][ ][ ] ... [ ][ ][ ][ ]   ← only 1 byte needed here
Cache line 2:   [T2][ ][ ][ ] ... [ ][ ][ ][ ]   ← etc...
...
Cache line 31:  [T31][ ][ ][ ] ...               ← 32 separate transactions!
```

### Case B: Random access
```
Threads access random memory addresses:
  Thread 0  reads A[47291]
  Thread 1  reads A[128]
  Thread 2  reads A[9872341]
  ...

Similar result: many cache lines touched, many transactions.
Efficiency can drop below 3%.
```

## 3.3 The Rule for Coalesced Access

```
COALESCED = threads T0, T1, T2, ..., T31 access
            CONSECUTIVE memory addresses.

Good: A[threadIdx.x]           ← consecutive ✓
Good: A[blockIdx.x * N + threadIdx.x]  ← consecutive within a block ✓

Bad:  A[threadIdx.x * 32]      ← strided ✗
Bad:  A[hash(threadIdx.x)]     ← random ✗

The formula `i = blockIdx.x * blockDim.x + threadIdx.x`
GUARANTEES coalesced access when you do A[i].

This is why it's the standard CUDA pattern.
```

## 3.4 Why This Matters For LLMs

```
Matrix stored row-major (standard):
  W = [[w00, w01, w02, w03],
       [w10, w11, w12, w13],
       [w20, w21, w22, w23]]

In memory (1D):
  W[0], W[1], W[2], W[3], W[4], W[5], ...
  w00   w01   w02   w03   w10   w11   ...

If threads access the SAME ROW at consecutive columns → COALESCED ✓
If threads access the SAME COLUMN at consecutive rows → UNCOALESCED ✗
  (Because column elements are far apart in memory: w00, w10, w20 — 
   each separated by 4 × 4 = 16 bytes for a 4-column matrix)

This is why:
  Matrix layout (row-major vs column-major) MATTERS.
  Whether you read across rows or down columns MATTERS.
  cuBLAS is column-major but takes care of all this for you.
```

---

# PART 4: MEMORY ALIGNMENT

## 4.1 What Alignment Means

```
A memory address is "N-byte aligned" if it's divisible by N.

float (4 bytes):   needs 4-byte alignment → address must be multiple of 4
double (8 bytes):  needs 8-byte alignment → address must be multiple of 8
float4 (16 bytes): needs 16-byte alignment → address must be multiple of 16

cudaMalloc() always returns 256-byte-aligned addresses. Good.
But custom allocators or struct layouts can create misaligned access.
```

## 4.2 Why Misaligned Access Is Slow

```
Aligned float4 load (16 bytes):
  Address = 128 (divisible by 16)
  Fetches bytes 128-143 in one transaction.
  
Misaligned float4 load (16 bytes):
  Address = 130 (NOT divisible by 16)
  Spans cache line boundary: needs bytes 128-131 AND 132-143
  GPU issues 2 transactions instead of 1.
  2x slower!
```

## 4.3 Practical Rules

```
✓ Use cudaMalloc() — always returns aligned memory
✓ Keep structs simple — avoid mixing sizes that misalign
✓ Use float4 for vectorized access (read 4 floats at once)

Example of float4 vectorized access (advanced trick):

// Instead of:
for (int i = 0; i < N; i++) C[i] = A[i] + B[i];  // 1 float per thread

// Use float4 (4 floats per thread):
float4 *A4 = reinterpret_cast<float4*>(A);
float4 *B4 = reinterpret_cast<float4*>(B);
float4 *C4 = reinterpret_cast<float4*>(C);
for (int i = 0; i < N/4; i++) {
    float4 a = A4[i];  // loads 16 bytes in ONE instruction
    float4 b = B4[i];
    C4[i] = make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}
// Same total work, but 4x fewer memory instructions → faster.
// Libraries like cuBLAS use this trick extensively.
```

---

# PART 5: THE cudaMemcpy FAMILY

## 5.1 All cudaMemcpy Directions

```c
cudaMemcpy(dst, src, size, kind);

kind values:
  cudaMemcpyHostToDevice     CPU RAM → GPU HBM    (over PCIe, slow: ~25 GB/s)
  cudaMemcpyDeviceToHost     GPU HBM → CPU RAM    (over PCIe, slow: ~25 GB/s)
  cudaMemcpyDeviceToDevice   GPU HBM → GPU HBM    (fast: full HBM speed ~300 GB/s)
  cudaMemcpyHostToHost       CPU RAM → CPU RAM    (avoid, use memcpy())
  cudaMemcpyDefault          GPU infers direction (convenient, uses same bandwidth)
```

## 5.2 Variants for Advanced Use

```c
// Async — don't block the CPU, comes back immediately
cudaMemcpyAsync(dst, src, size, kind, stream);

// 2D — for images or 2D arrays with pitch (row padding)
cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);

// 3D — for 3D data
cudaMemcpy3D(&params);

// Managed memory (unified) — one pointer works on both CPU and GPU
cudaMallocManaged(&ptr, size);
// No need for cudaMemcpy! GPU driver migrates pages as needed.
// Convenient but has hidden costs.
```

## 5.3 PCIe Bottleneck

```
CPU-GPU data transfer goes over PCIe:
  PCIe 3.0 x16: ~12 GB/s
  PCIe 4.0 x16: ~25 GB/s
  PCIe 5.0 x16: ~50 GB/s

Compare:
  GPU HBM bandwidth: 300-3350 GB/s (12-100x faster than PCIe!)

Implication: CPU-GPU transfers are THE bottleneck.
  Moving a 14 GB LLM from CPU → GPU: 14 / 25 = 0.56 seconds (PCIe 4.0)
  Running the same model once generation: ~4 ms on H100

Golden rule: minimize CPU↔GPU transfers.
  - Load weights once, keep them on GPU
  - Process in batches to amortize transfer cost
  - Use async transfers overlapped with compute (next week)
```

---

# PART 6: MEASURING BANDWIDTH

## 6.1 The Formula

```
bandwidth (GB/s) = bytes_transferred / time_taken_in_seconds / 1e9

For a vector add C = A + B with N floats:
  Reads:  2 × N × 4 bytes  (A and B)
  Writes: 1 × N × 4 bytes  (C)
  Total:  3 × N × 4 bytes = 12N bytes

If kernel takes 0.04 ms on N = 1M:
  bandwidth = 12,000,000 bytes / 0.00004 seconds / 1e9
            = 300 GB/s

Compare to T4 peak (~320 GB/s) → 94% efficiency. Excellent!
```

## 6.2 What Peak Bandwidth You Should Expect

```
GPU              Peak HBM bw   Realistic max
─────────────────────────────────────────────
T4 (Colab free)   320 GB/s     ~280 GB/s (88%)
V100              900 GB/s     ~800 GB/s
A100              1555 GB/s    ~1400 GB/s
H100              3350 GB/s    ~3000 GB/s

Below 60% = something is wrong (bad access pattern or small data)
60-85%     = decent, can be improved
85-95%     = excellent, near-optimal kernel
>95%       = you're hitting the hardware wall
```

---

# PART 7: CODING EXERCISES

## Exercise 1: Coalesced vs Strided Access

Setup: Run these in Colab. They show the SAME amount of data read,
but with different access patterns. The performance difference is dramatic.

```c
%%writefile coalesce_test.cu
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

// GOOD: Coalesced access — thread i reads A[i]
__global__ void coalesced_copy(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}

// BAD: Strided access — thread i reads A[i * STRIDE]
__global__ void strided_copy(float *in, float *out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < N) out[i] = in[idx];
}

int main() {
    int N = 16 * 1024 * 1024;   // 16M floats = 64 MB
    size_t bytes = N * sizeof(float);
    
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    
    int block_size = 256;
    int blocks = (N + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    coalesced_copy<<<blocks, block_size>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║          Coalesced vs Strided Access Benchmark            ║\n");
    printf("║          %d elements (%.0f MB)                      ║\n", N, bytes/1e6);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Access Pattern │ Time (ms) │ Bandwidth │ Slowdown        ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    
    // Benchmark coalesced
    cudaEventRecord(start);
    for (int i = 0; i < 50; i++)
        coalesced_copy<<<blocks, block_size>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_coalesced;
    cudaEventElapsedTime(&ms_coalesced, start, stop);
    ms_coalesced /= 50;
    float bw_coalesced = 2.0f * bytes / (ms_coalesced/1000.0f) / 1e9;
    printf("║ Coalesced      │ %9.3f │ %6.0f GB/s │ 1.0x (baseline) ║\n",
           ms_coalesced, bw_coalesced);
    
    // Benchmark different strides
    int strides[] = {2, 4, 8, 16, 32};
    for (int s = 0; s < 5; s++) {
        int stride = strides[s];
        int N_effective = N / stride;
        
        // Warmup
        strided_copy<<<blocks, block_size>>>(d_in, d_out, N, stride);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++)
            strided_copy<<<blocks, block_size>>>(d_in, d_out, N, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_strided;
        cudaEventElapsedTime(&ms_strided, start, stop);
        ms_strided /= 50;
        
        // Effective bytes = only the bytes actually needed
        float bytes_useful = 2.0f * N_effective * sizeof(float);
        float bw_strided = bytes_useful / (ms_strided/1000.0f) / 1e9;
        
        printf("║ Stride %-7d │ %9.3f │ %6.0f GB/s │ %4.1fx slower     ║\n",
               stride, ms_strided, bw_strided, ms_strided/ms_coalesced);
    }
    
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\nLesson: Coalesced access is critical. Stride patterns waste bandwidth.\n");
    
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

```python
!nvcc coalesce_test.cu -o coalesce_test && ./coalesce_test
```

**What you'll see on a T4:**
```
║ Coalesced      │     3.5   │   280 GB/s │ 1.0x (baseline) ║
║ Stride 2       │     3.5   │   140 GB/s │ 1.0x slower     ║
║ Stride 4       │     3.5   │    70 GB/s │ 1.0x slower     ║
║ Stride 8       │     3.5   │    35 GB/s │ 1.0x slower     ║
║ Stride 16      │     3.5   │    17 GB/s │ 1.0x slower     ║
║ Stride 32      │     3.5   │    8 GB/s  │ 1.0x slower     ║
```

**Why same time but lower bandwidth?**
Each strided access still loads FULL cache lines, but you only USE a fraction.
Time is similar because the GPU still does the same amount of work.
But "effective bandwidth" (bytes you USE) drops proportionally to stride.

### Understanding the Code

```c
__global__ void coalesced_copy(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}
```
```
This is the canonical pattern. Each thread reads its own element.
Thread 0: in[0] → out[0]
Thread 1: in[1] → out[1]
...
All threads in a warp read consecutive addresses → COALESCED.
```

```c
__global__ void strided_copy(float *in, float *out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < N) out[i] = in[idx];
}
```
```
With stride=4:
  Thread 0: in[0]  → out[0]
  Thread 1: in[4]  → out[1]
  Thread 2: in[8]  → out[2]
  ...
  Thread 31: in[124] → out[31]
  
Addresses within the warp span: 0 to 124 × 4 bytes = 124 × 4 = 496 bytes.
That's 4 cache lines (128 bytes each).
GPU loads 4 × 128 = 512 bytes, but only uses 32 × 4 = 128 bytes.
Efficiency: 25%.

With stride=32: addresses span 32 × 128 = 4096 bytes = 32 cache lines.
GPU loads 32 × 128 = 4096 bytes, uses 128. Efficiency: 3%.
```

## Exercise 2: Memory Bandwidth Benchmark

```c
%%writefile bw_test.cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        printf("CUDA error: %s\n", cudaGetErrorString(err));      \
        exit(1);                                                 \
    }                                                            \
} while(0)

// Just copy memory: read and write one array
__global__ void copy_kernel(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}

int main() {
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    float peak_bw_gb_s = 2.0f * prop.memoryClockRate * 1000.0f * 
                          (prop.memoryBusWidth / 8) / 1e9;
    
    printf("GPU: %s\n", prop.name);
    printf("Memory clock: %d MHz\n", prop.memoryClockRate / 1000);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("Theoretical peak bandwidth: %.1f GB/s\n\n", peak_bw_gb_s);
    
    printf("Size (MB) | Time (ms) | Bandwidth (GB/s) | Efficiency\n");
    printf("──────────┼───────────┼──────────────────┼───────────\n");
    
    // Test with various sizes
    int sizes_mb[] = {1, 4, 16, 64, 256};
    
    for (int s = 0; s < 5; s++) {
        int size_mb = sizes_mb[s];
        int N = size_mb * 1024 * 1024 / sizeof(float);
        size_t bytes = N * sizeof(float);
        
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        
        int block_size = 256;
        int blocks = (N + block_size - 1) / block_size;
        
        // Warmup
        copy_kernel<<<blocks, block_size>>>(d_in, d_out, N);
        cudaDeviceSynchronize();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++)
            copy_kernel<<<blocks, block_size>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= 50;
        
        // Bytes transferred: read once + write once = 2 × bytes
        float bw = 2.0f * bytes / (ms/1000.0f) / 1e9;
        float efficiency = 100.0f * bw / peak_bw_gb_s;
        
        printf("%8d  │ %9.3f │ %16.1f │ %8.1f%%\n",
               size_mb, ms, bw, efficiency);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_in);
        cudaFree(d_out);
    }
    
    return 0;
}
```

```python
!nvcc bw_test.cu -o bw_test && ./bw_test
```

**What to observe:**
- Small arrays (1 MB) → lower bandwidth (overhead dominates)
- Large arrays (64+ MB) → close to peak bandwidth
- Understanding peak bandwidth is how you know if your kernel is optimized

### Understanding the Device Query Code

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

float peak_bw_gb_s = 2.0f * prop.memoryClockRate * 1000.0f * 
                      (prop.memoryBusWidth / 8) / 1e9;
```
```
Formula for peak theoretical bandwidth:
  peak_bw = 2 × memory_clock × (bus_width / 8)

Where:
  2 × (DDR = Double Data Rate: transfers on both clock edges)
  memory_clock (kHz from CUDA, multiply by 1000 for Hz)
  bus_width (bits) / 8 = bytes per transfer

For T4:
  memory_clock = 5,001,000 kHz = 5 GHz
  bus_width = 256 bits
  peak_bw = 2 × 5e9 × 256/8 / 1e9 = 320 GB/s ✓

For H100:
  memory_clock = 2,619,000 kHz
  bus_width = 5120 bits
  peak_bw = 2 × 2.619e9 × 5120/8 / 1e9 = 3,350 GB/s ✓

This is the HARDWARE LIMIT. No kernel can exceed this.
If your kernel achieves 280 GB/s on T4, that's 88% of peak — excellent.
```

---

# PART 8: TODAY'S MINI-PROJECT 🔨

## Project: "Bandwidth Dashboard — Know Your GPU's Limits"

A tool that comprehensively benchmarks memory bandwidth with different access patterns, data sizes, and data types. Output a clean report.

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

// Copy kernel (baseline: 2 arrays, read + write)
__global__ void copy_kernel(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}

// Scale kernel (1 input + 1 output)
__global__ void scale_kernel(float *in, float *out, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * s;
}

// Triad kernel (2 inputs + 1 output, STREAM benchmark style)
__global__ void triad_kernel(float *A, float *B, float *C, float s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + s * B[i];
}

// Vectorized float4 copy (4 floats per thread, fewer memory instructions)
__global__ void copy_vec4(float4 *in, float4 *out, int N4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N4) out[i] = in[i];
}

// Strided copy (shows bad access pattern impact)
__global__ void strided_copy(float *in, float *out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * stride < N) out[i] = in[i * stride];
}

float run_bench(void (*launcher)(float*, float*, int),
                float *d_in, float *d_out, int N, int block_size) {
    int blocks = (N + block_size - 1) / block_size;
    
    // Warmup
    launcher(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) launcher(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100;
}

void launch_copy(float *in, float *out, int N) {
    int bs = 256;
    copy_kernel<<<(N+bs-1)/bs, bs>>>(in, out, N);
}

void launch_copy_vec4(float *in, float *out, int N) {
    int bs = 256;
    int N4 = N / 4;
    copy_vec4<<<(N4+bs-1)/bs, bs>>>((float4*)in, (float4*)out, N4);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peak_bw = 2.0f * prop.memoryClockRate * 1000.0f *
                     (prop.memoryBusWidth / 8) / 1e9;
    
    int N = 64 * 1024 * 1024;   // 64M floats = 256 MB
    size_t bytes = N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                  BANDWIDTH DASHBOARD                        ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║ GPU:              %-40s ║\n", prop.name);
    printf("║ Memory bus:       %d-bit %s                        ║\n",
           prop.memoryBusWidth, prop.memoryBusWidth >= 1024 ? "HBM" : "GDDR");
    printf("║ Theoretical peak: %.1f GB/s                              ║\n", peak_bw);
    printf("║ Data size:        %d MB                                   ║\n", (int)(bytes/1024/1024));
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║ Test           │ Time (ms) │ Bandwidth │ Efficiency       ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    
    // Test 1: Simple copy (baseline)
    float ms = run_bench(launch_copy, d_A, d_B, N, 256);
    float bw = 2.0f * bytes / (ms/1000.0f) / 1e9;
    printf("║ Copy           │ %9.3f │ %6.0f GB/s │ %5.1f%% of peak  ║\n",
           ms, bw, 100.0f*bw/peak_bw);
    
    // Test 2: Vectorized copy (float4)
    ms = run_bench(launch_copy_vec4, d_A, d_B, N, 256);
    bw = 2.0f * bytes / (ms/1000.0f) / 1e9;
    printf("║ Copy (float4)  │ %9.3f │ %6.0f GB/s │ %5.1f%% of peak  ║\n",
           ms, bw, 100.0f*bw/peak_bw);
    
    // Test 3: Strided access (demonstrate uncoalesced)
    int strides[] = {2, 4, 8, 16};
    for (int s = 0; s < 4; s++) {
        int stride = strides[s];
        int bs = 256;
        int N_strided = N;
        
        // Warmup
        strided_copy<<<(N_strided+bs-1)/bs, bs>>>(d_A, d_B, N_strided, stride);
        cudaDeviceSynchronize();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++)
            strided_copy<<<(N_strided+bs-1)/bs, bs>>>(d_A, d_B, N_strided, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        ms /= 100;
        
        // Effective bandwidth = only what's actually used
        float bw_eff = 2.0f * (N/stride) * sizeof(float) / (ms/1000.0f) / 1e9;
        printf("║ Stride=%-2d      │ %9.3f │ %6.0f GB/s │ %5.1f%% of peak  ║\n",
               stride, ms, bw_eff, 100.0f*bw_eff/peak_bw);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║ LLM Implications:                                          ║\n");
    printf("║   Your GPU can serve LLaMA-7B at ~%.0f tokens/sec          ║\n",
           peak_bw / 13.4f);
    printf("║   With INT4 quantization: ~%.0f tokens/sec                 ║\n",
           peak_bw / 3.4f);
    printf("║   (assuming memory-bound decode phase)                     ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

```python
!nvcc bw_dashboard.cu -o bw_dashboard && ./bw_dashboard
```

**Expected output on T4:**
```
╔════════════════════════════════════════════════════════════╗
║                  BANDWIDTH DASHBOARD                        ║
╠════════════════════════════════════════════════════════════╣
║ GPU:              Tesla T4                                  ║
║ Memory bus:       256-bit GDDR                              ║
║ Theoretical peak: 320.0 GB/s                                ║
║ Data size:        256 MB                                    ║
╠════════════════════════════════════════════════════════════╣
║ Test           │ Time (ms) │ Bandwidth │ Efficiency       ║
╠════════════════════════════════════════════════════════════╣
║ Copy           │    1.800  │   280 GB/s│  87.5% of peak   ║
║ Copy (float4)  │    1.650  │   310 GB/s│  96.9% of peak   ║ ← vectorized wins
║ Stride=2       │    1.800  │   140 GB/s│  43.8% of peak   ║
║ Stride=4       │    1.800  │    70 GB/s│  21.9% of peak   ║
║ Stride=8       │    1.800  │    35 GB/s│  10.9% of peak   ║
║ Stride=16      │    1.800  │    17 GB/s│   5.3% of peak   ║
╠════════════════════════════════════════════════════════════╣
║ LLM Implications:                                          ║
║   Your GPU can serve LLaMA-7B at ~24 tokens/sec            ║
║   With INT4 quantization: ~94 tokens/sec                   ║
║   (assuming memory-bound decode phase)                     ║
╚════════════════════════════════════════════════════════════╝
```

**Why this project is valuable:**
1. You now KNOW your GPU's actual bandwidth ceiling
2. You can quickly check if a new kernel you write is optimized
3. You understand why quantization helps inference speed (bw ÷ model_size = tokens/sec)
4. You've used `float4` vectorization — a key optimization technique

---

# PART 9: CONNECTING TO LLMs

## Why Today Matters For Every LLM Kernel

```
Every LLM operation is memory-bound somewhere:

1. Embedding lookup
   → Thread i reads row[token_id_i] of embedding table
   → If token_ids are random, access is UNCOALESCED
   → Solution: sort by token_id before lookup (token clustering)

2. Attention QK^T
   → Different heads access different memory regions
   → Row-major vs column-major layout matters
   → Flash Attention organizes accesses for coalescing

3. FFN up/down projections
   → Standard matrix multiply
   → Well-optimized in cuBLAS (coalesced internally)

4. KV-cache read during decode
   → Each new token reads all previous K, V
   → Layout: [num_heads, seq_len, head_dim] vs [seq_len, num_heads, head_dim]
   → PagedAttention (vLLM) optimizes this layout

5. Weight loading
   → cuBLAS/cuDNN are coalesced
   → Custom kernels need manual care
```

## Why Inference Is Memory-Bound

```
LLaMA-7B at FP16: 13.4 GB of weights.
Each token = reads ALL weights once.

On T4 (~280 GB/s measured):
  13.4 GB / 280 GB/s = 48 ms per token = ~21 tokens/sec

On H100 (~3000 GB/s measured):
  13.4 GB / 3000 GB/s = 4.5 ms per token = ~224 tokens/sec

QUANTIZATION to INT4: model becomes 3.4 GB.
  H100: 3.4 / 3000 = 1.1 ms = 909 tokens/sec (4x speedup!)

This is why EVERYONE quantizes for inference.
The speedup comes from LESS MEMORY READ, not from faster compute.
```

---

# CHECKLIST

After Day 1 of Week 2:
- [ ] Understand global memory (HBM) and why it dominates performance
- [ ] Know what a cache line is and why it matters (32 bytes L2, 128 bytes L1)
- [ ] Can explain coalesced vs uncoalesced access with concrete examples
- [ ] Understand the `i = blockIdx.x * blockDim.x + threadIdx.x` pattern guarantees coalescing
- [ ] Know memory alignment requirements (4-byte for float, 16-byte for float4)
- [ ] Can use all `cudaMemcpy` directions (H2D, D2H, D2D)
- [ ] Can calculate and measure memory bandwidth
- [ ] Know theoretical peak bandwidth for your GPU
- [ ] Understand why PCIe is the bottleneck between CPU and GPU
- [ ] Built the Bandwidth Dashboard mini-project
- [ ] Can predict LLM inference speed from memory bandwidth

**Tomorrow (Day 2): Shared Memory** — the on-chip SRAM that's 20-100x faster
than HBM. You'll learn to manually cache data for maximum performance.
This is the foundation of Flash Attention.

---

*Status: ⬜ NOT YET COMPLETED*
*Date completed: ___________*
