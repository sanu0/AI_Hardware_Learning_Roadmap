# Week 1, Day 4: CUDA Programming — Your First GPU Kernels
## Today You Write Code That Runs on Thousands of GPU Cores

Days 1-3 were theory. Today you WRITE CUDA CODE. By the end of this session,
you'll have written kernels that run on thousands of threads in parallel,
understood the CUDA programming model, and built a vector operations library.

**Time: ~2.5-3 hours (heavy coding day)**
**Setup: Google Colab with GPU runtime**

---

# PART 0: NEW TERMS FOR TODAY

```
HOST        = The CPU + CPU memory (RAM). Your "normal" computer.
DEVICE      = The GPU + GPU memory (HBM). The accelerator.

KERNEL      = A function that runs on the GPU.
              NOT the same as "kernel" in operating systems.
              In CUDA, kernel = GPU function you write and call.
              One kernel call launches thousands of threads.

__global__  = A CUDA keyword. Marks a function as a kernel.
              "This function runs on the GPU but is CALLED from the CPU."

__device__  = A CUDA keyword. "This function runs on the GPU
              and can only be CALLED from other GPU functions."

__host__    = A CUDA keyword. "This function runs on the CPU."
              (This is the default for normal C/C++ functions.)

<<<grid, block>>>  = Kernel launch configuration.
              How many blocks and how many threads per block.
              Written between the function name and arguments.
              Example: my_kernel<<<256, 512>>>(args)
                       = 256 blocks × 512 threads = 131,072 threads total

threadIdx   = Built-in variable. Which thread am I within my block?
              threadIdx.x, threadIdx.y, threadIdx.z

blockIdx    = Built-in variable. Which block am I in the grid?
              blockIdx.x, blockIdx.y, blockIdx.z

blockDim    = Built-in variable. How many threads in my block?
              blockDim.x, blockDim.y, blockDim.z

gridDim     = Built-in variable. How many blocks in the grid?
              gridDim.x, gridDim.y, gridDim.z

nvcc        = NVIDIA CUDA Compiler. Compiles .cu files → GPU executables.
              Turns your C/CUDA code into:
                PTX (virtual assembly) → SASS (actual GPU machine code)

cudaMalloc  = Allocate memory on the GPU (like malloc but on GPU HBM).

cudaMemcpy  = Copy data between CPU and GPU (or GPU to GPU).
              Directions: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost

cudaFree    = Free GPU memory (like free() but for GPU).
```

---

# PART 1: THE HOST-DEVICE MODEL

## 1.1 Two Separate Computers

When you write a CUDA program, you're writing code for TWO processors:

```
YOUR PROGRAM runs on BOTH:

┌──────────────────────────┐          ┌──────────────────────────┐
│        HOST (CPU)         │          │       DEVICE (GPU)        │
│                           │          │                           │
│  - Main program logic     │          │  - Kernel functions       │
│  - File I/O, networking   │  PCIe    │  - Parallel computation   │
│  - Memory: RAM (32 GB)    │ ←─────→ │  - Memory: HBM (16 GB+)  │
│  - Launches GPU kernels   │  bus     │  - Thousands of threads   │
│  - Copies data to/from GPU│          │  - Can't do file I/O!     │
│                           │          │  - Can't print (mostly)!  │
└──────────────────────────┘          └──────────────────────────┘

CPU RAM and GPU HBM are SEPARATE memory spaces.
Data must be explicitly COPIED between them.
```

**The workflow for every CUDA program:**
```
1. Allocate memory on GPU (cudaMalloc)
2. Copy input data: CPU → GPU (cudaMemcpy Host→Device)
3. Launch kernel on GPU (my_kernel<<<grid, block>>>(args))
4. GPU executes thousands of threads in parallel
5. Copy results: GPU → CPU (cudaMemcpy Device→Host)
6. Free GPU memory (cudaFree)
```

**LLM connection:** This is EXACTLY what PyTorch does when you call:
```python
x = x.cuda()          # Step 2: cudaMemcpy CPU → GPU
y = model(x)          # Step 3-4: launches cuBLAS/cuDNN kernels
result = y.cpu()      # Step 5: cudaMemcpy GPU → CPU
```
PyTorch hides all the cudaMalloc/cudaMemcpy/cudaFree. Today you do it manually.

## 1.2 Why Separate Memory?

```
CPU RAM:
  - Large (32-128 GB typical)
  - Accessible by CPU directly
  - Connected via memory controller on CPU chip
  - Bandwidth: ~50-100 GB/s

GPU HBM:
  - Smaller (16-80 GB typical)
  - Accessible by GPU directly
  - Connected via memory controllers on GPU chip
  - Bandwidth: ~1000-3350 GB/s (10-30x faster than CPU RAM!)

PCIe bus (connecting CPU to GPU):
  - Bandwidth: ~25-32 GB/s (PCIe 4.0 x16)
  - This is the BOTTLENECK between CPU and GPU
  - Copying 14 GB of model weights CPU→GPU: ~0.5 seconds

KEY INSIGHT: Once data is on the GPU, keep it there!
Moving data between CPU and GPU is slow (PCIe bottleneck).
This is why LLM inference does ALL computation on GPU and
only sends tiny results (token IDs) back to CPU.
```

---

# PART 2: WRITING YOUR FIRST CUDA KERNEL

## 2.1 On Google Colab: How to Run CUDA C Code

Colab doesn't have .cu files by default, but you can use the `%%cuda` magic
or write files and compile with nvcc. We'll use the file approach:

```python
# Run this cell first to set up CUDA compilation in Colab
!pip install nvcc4jupyter > /dev/null 2>&1
%load_ext nvcc4jupyter
```

If that doesn't work, use this alternative approach (write file + compile):

```python
%%writefile hello.cu
#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();  // 2 blocks, 4 threads each = 8 threads
    cudaDeviceSynchronize();   // wait for GPU to finish
    return 0;
}
```

```python
!nvcc hello.cu -o hello && ./hello
```

## 2.2 Hello World Kernel — Line by Line

```c
#include <stdio.h>

// __global__ means: this function runs on the GPU
//                   but is CALLED from the CPU
__global__ void hello_kernel() {

    // threadIdx.x = which thread am I within my block? (0, 1, 2, or 3)
    // blockIdx.x  = which block am I? (0 or 1)
    
    // Calculate a UNIQUE global ID for this thread
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    //             = block_number × threads_per_block + thread_within_block
    //
    // Block 0, Thread 0: global_id = 0 × 4 + 0 = 0
    // Block 0, Thread 1: global_id = 0 × 4 + 1 = 1
    // Block 0, Thread 2: global_id = 0 × 4 + 2 = 2
    // Block 0, Thread 3: global_id = 0 × 4 + 3 = 3
    // Block 1, Thread 0: global_id = 1 × 4 + 0 = 4
    // Block 1, Thread 1: global_id = 1 × 4 + 1 = 5
    // Block 1, Thread 2: global_id = 1 × 4 + 2 = 6
    // Block 1, Thread 3: global_id = 1 × 4 + 3 = 7
    
    printf("Hello from global thread %d (block %d, local thread %d)\n",
           global_id, blockIdx.x, threadIdx.x);
}

int main() {
    // Launch kernel: <<<number_of_blocks, threads_per_block>>>
    hello_kernel<<<2, 4>>>();
    
    // IMPORTANT: kernel launch is ASYNCHRONOUS
    // The CPU doesn't wait for the GPU to finish.
    // cudaDeviceSynchronize() makes the CPU wait.
    cudaDeviceSynchronize();
    
    printf("All GPU threads finished!\n");
    return 0;
}
```

**Expected output** (order may vary — threads run in parallel!):
```
Hello from global thread 0 (block 0, local thread 0)
Hello from global thread 1 (block 0, local thread 1)
Hello from global thread 2 (block 0, local thread 2)
Hello from global thread 3 (block 0, local thread 3)
Hello from global thread 4 (block 1, local thread 0)
Hello from global thread 5 (block 1, local thread 1)
Hello from global thread 6 (block 1, local thread 2)
Hello from global thread 7 (block 1, local thread 3)
```

## 2.3 The Global Thread ID Formula

This formula is the MOST important thing you learn today:

```
global_thread_id = blockIdx.x * blockDim.x + threadIdx.x

                   ┌─── which block ───┐   ┌── which thread in block ──┐
                   │                    │   │                           │
Example: 4 blocks of 256 threads = 1024 total threads

Block 0: threads 0-255      (blockIdx.x=0, threadIdx.x=0..255)
Block 1: threads 256-511    (blockIdx.x=1, threadIdx.x=0..255)
Block 2: threads 512-767    (blockIdx.x=2, threadIdx.x=0..255)
Block 3: threads 768-1023   (blockIdx.x=3, threadIdx.x=0..255)

Thread 0 in Block 2: global_id = 2 * 256 + 0 = 512
Thread 100 in Block 3: global_id = 3 * 256 + 100 = 868

This global ID is how each thread knows WHICH element of the array to process.
Thread 512 processes array element 512. Thread 868 processes element 868.
```

---

# PART 3: VECTOR ADDITION — The Classic First Kernel

## 3.1 The Problem

```
Given two arrays A and B, compute C where C[i] = A[i] + B[i]

CPU version (sequential):
  for (int i = 0; i < N; i++)
      C[i] = A[i] + B[i];

GPU version (parallel):
  Each thread computes ONE element: C[my_id] = A[my_id] + B[my_id]
  Launch N threads → all elements computed simultaneously!
```

## 3.2 Complete Vector Addition in CUDA

```c
%%writefile vec_add.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// The kernel: runs on GPU, each thread adds ONE element
__global__ void vec_add_kernel(float *A, float *B, float *C, int N) {
    // Calculate which element this thread handles
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BOUNDS CHECK: don't access beyond array size!
    // Why? We might launch more threads than array elements
    // (because block size must be a multiple of 32)
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;  // 1 million elements
    size_t bytes = N * sizeof(float);  // 4 MB per array
    
    // ========== STEP 1: Allocate CPU memory ==========
    float *h_A = (float*)malloc(bytes);  // h_ prefix = host (CPU)
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Fill with random data
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // ========== STEP 2: Allocate GPU memory ==========
    float *d_A, *d_B, *d_C;  // d_ prefix = device (GPU)
    cudaMalloc(&d_A, bytes);  // allocate on GPU HBM
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // ========== STEP 3: Copy input data CPU → GPU ==========
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // ========== STEP 4: Launch kernel ==========
    int threads_per_block = 256;  // common choice (must be ≤ 1024)
    
    // How many blocks do we need?
    // If N = 1,000,000 and threads_per_block = 256:
    // blocks = ceil(1000000 / 256) = 3907
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    printf("Launching %d blocks × %d threads = %d total threads\n",
           blocks, threads_per_block, blocks * threads_per_block);
    printf("Array size: %d elements (%.1f MB)\n", N, bytes / 1e6);
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    vec_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // ========== STEP 5: Copy result GPU → CPU ==========
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // ========== STEP 6: Verify result ==========
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            errors++;
        }
    }
    
    printf("Result: %s\n", errors == 0 ? "CORRECT!" : "ERRORS FOUND!");
    printf("Kernel time: %.3f ms\n", milliseconds);
    
    // Calculate effective bandwidth
    // We read 2 arrays (A, B) and write 1 array (C) = 3 × N × 4 bytes
    float gb_accessed = 3.0f * N * sizeof(float) / 1e9;
    float bandwidth = gb_accessed / (milliseconds / 1000.0f);
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth);
    printf("(T4 theoretical max: ~320 GB/s)\n");
    
    // ========== STEP 7: Free memory ==========
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

Compile and run:
```python
!nvcc vec_add.cu -o vec_add && ./vec_add
```

### Understanding Vector Addition — Line by Line

**Header files:**
```c
#include <stdio.h>     // for printf
#include <stdlib.h>    // for malloc, free, rand
#include <time.h>      // for timing (not strictly needed here)
```

**The GPU kernel:**
```c
__global__ void vec_add_kernel(float *A, float *B, float *C, int N) {
```
```
__global__     = "this function runs on GPU, callable from CPU"
void           = returns nothing (kernels always return void)
float *A,*B,*C = pointers to arrays in GPU memory (HBM)
int N          = array size (so threads know the bounds)

Notice: kernel gets DEVICE pointers (d_A, d_B, d_C).
It can ONLY access GPU memory. CPU memory is invisible to it.
```

```c
    int i = blockIdx.x * blockDim.x + threadIdx.x;
```
```
THE most important formula. Assigns each thread a unique global ID:

  blockIdx.x    = which block am I in? (0, 1, 2, ..., 3906)
  blockDim.x    = how many threads per block? (256)
  threadIdx.x   = which thread within my block? (0, 1, 2, ..., 255)

Examples:
  Block 0, Thread 5:    i = 0×256 + 5   = 5      → handles A[5], B[5], C[5]
  Block 10, Thread 100: i = 10×256 + 100 = 2660   → handles A[2660], B[2660], C[2660]
  Block 3906, Thread 64: i = 3906×256 + 64 = 1,000,000

Every one of the ~1 million threads gets a unique i.
```

```c
    if (i < N) {
        C[i] = A[i] + B[i];
    }
```
```
BOUNDS CHECK — critical!

Why? We launched 3907 blocks × 256 threads = 1,000,192 threads.
But the array has only 1,000,000 elements.
Threads with i = 1,000,000 to 1,000,191 would access INVALID memory!

Without this check: segfault / CUDA error / data corruption.

Inside the if: the actual work.
  C[i] = A[i] + B[i]
  "Read element i from A, read element i from B, add them, write to C[i]"
  
  This happens SIMULTANEOUSLY on ~1 million threads.
  One big parallel computation.
```

**The main() function — the CPU workflow:**

```c
int N = 1000000;                      // 1 million elements
size_t bytes = N * sizeof(float);     // 1M × 4 bytes = 4,000,000 bytes = 4 MB
```

**STEP 1: Allocate CPU memory (RAM)**
```c
float *h_A = (float*)malloc(bytes);   // h_ = host (CPU RAM)
float *h_B = (float*)malloc(bytes);
float *h_C = (float*)malloc(bytes);
```
```
malloc() = standard C function to request memory
(float*) = type cast: tell C "treat this as array of floats"
bytes    = how much memory (4 MB)

h_ prefix = Host (CPU) — coding convention to track where data lives
d_ prefix = Device (GPU)
  Without this convention, you'll mix up CPU and GPU pointers
  and get cryptic errors.

After this: 3 arrays of 1M floats each, sitting in CPU RAM.
Contents: uninitialized garbage.
```

```c
for (int i = 0; i < N; i++) {
    h_A[i] = (float)rand() / RAND_MAX;   // random value between 0 and 1
    h_B[i] = (float)rand() / RAND_MAX;
}
```
```
Fill h_A and h_B with random numbers.
  rand() returns an int between 0 and RAND_MAX (typically ~2 billion)
  Dividing converts to a float between 0.0 and 1.0

h_C stays uninitialized — will be overwritten by GPU result.
```

**STEP 2: Allocate GPU memory (HBM)**
```c
float *d_A, *d_B, *d_C;              // d_ = device (GPU HBM)
cudaMalloc(&d_A, bytes);             // allocate 4 MB on GPU
cudaMalloc(&d_B, bytes);
cudaMalloc(&d_C, bytes);
```
```
cudaMalloc() = GPU version of malloc().

Important: it takes a POINTER TO A POINTER (&d_A, not d_A).
  Why? Because it needs to WRITE the GPU address INTO your pointer variable.
  In C, to modify a variable from a function, you pass its address.

After this: 12 MB reserved on GPU HBM (3 arrays × 4 MB).
Contents: garbage. d_A, d_B, d_C now hold GPU memory addresses.

CPU CAN'T read these arrays — they're on a different chip.
```

**STEP 3: Copy input data CPU → GPU**
```c
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
```
```
cudaMemcpy(destination, source, size, direction)
  destination = d_A (GPU pointer)
  source      = h_A (CPU pointer)
  size        = 4 MB
  direction   = cudaMemcpyHostToDevice (CPU → GPU)

This is slow! Travels over PCIe bus (~25-32 GB/s).
4 MB transfer takes ~0.15 ms.
This is the bottleneck between CPU and GPU — minimize these transfers.

Other directions exist:
  cudaMemcpyDeviceToHost  → GPU → CPU
  cudaMemcpyDeviceToDevice → GPU → GPU (much faster, via HBM)
  cudaMemcpyHostToHost    → CPU → CPU
```

**STEP 4: Launch kernel**
```c
int threads_per_block = 256;
int blocks = (N + threads_per_block - 1) / threads_per_block;
```
```
threads_per_block = 256 (common choice, must be multiple of 32, max 1024)

blocks = CEILING(N / threads_per_block)
       = CEILING(1,000,000 / 256)
       = 3907

The formula (N + tpb - 1) / tpb is C's way of doing ceiling division:
  Integer division in C truncates:  1000000 / 256 = 3906 (missing 64 elements!)
  Adding (256-1) before dividing:   1000255 / 256 = 3907 (correct)

Total threads launched: 3907 × 256 = 1,000,192 (192 extra, handled by bounds check)
```

```c
vec_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
```
```
The kernel launch! Syntax: kernel<<<grid, block>>>(args)

Breaking it down:
  vec_add_kernel        = name of the GPU function
  <<<3907, 256>>>       = launch 3907 blocks with 256 threads each
  (d_A, d_B, d_C, N)    = arguments passed to EVERY thread
                          (all threads see the same pointers and N)

This IMMEDIATELY returns to CPU (asynchronous!).
The GPU starts working. The CPU continues to the next line.
```

```c
cudaEventRecord(start);
vec_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
```
```
CUDA events are GPU timestamps (more accurate than CPU time for GPU work):

  cudaEventRecord(start)  → mark "start" timestamp on GPU timeline
  kernel launch            → kernel starts executing
  cudaEventRecord(stop)   → mark "stop" timestamp (when kernel finishes)
  cudaEventSynchronize    → CPU waits until stop event is recorded

Why CUDA events instead of time.time()?
  time.time() on CPU might trigger BEFORE the kernel even starts (launch is async).
  CUDA events are recorded ON THE GPU, accurate to microseconds.

cudaEventElapsedTime(&ms, start, stop)
  Calculates milliseconds between start and stop events.
```

**STEP 5: Copy result GPU → CPU**
```c
cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
```
```
Bring the result back to CPU so we can verify/print it.
Direction: cudaMemcpyDeviceToHost (GPU → CPU).
```

**STEP 6: Verify the result**
```c
int errors = 0;
for (int i = 0; i < N; i++) {
    if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
        errors++;
    }
}
```
```
Check: does h_C[i] equal h_A[i] + h_B[i] for every element?

fabs() = absolute value (floating-point)
1e-5 = tolerance for floating-point rounding errors
(1e-5 means 0.00001 — float math isn't perfectly exact)

If GPU computed correctly: errors = 0
If something went wrong: errors > 0 (likely a bug in your kernel)

Always verify your GPU results! Silent errors are the worst kind.
```

**Bandwidth calculation:**
```c
float gb_accessed = 3.0f * N * sizeof(float) / 1e9;  // GB of memory accessed
float bandwidth = gb_accessed / (milliseconds / 1000.0f);
```
```
For each element, we:
  Read A[i]: 4 bytes
  Read B[i]: 4 bytes
  Write C[i]: 4 bytes
  Total per element: 12 bytes (3 × 4)

Total memory accessed: 3 × 1,000,000 × 4 bytes = 12 MB = 0.012 GB

Time taken: ~0.042 ms = 0.000042 seconds

Bandwidth: 0.012 GB / 0.000042 s = 286 GB/s

T4 theoretical max: 320 GB/s
Your achieved: 286 GB/s
Efficiency: 286 / 320 = 89% of peak bandwidth ← EXCELLENT!

This is memory-bound (no compute, just read + add + write).
You're close to the GPU's theoretical limit, which means the kernel
is well-optimized. You can't go faster without changing the algorithm.
```

**STEP 7: Free memory**
```c
cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);   // free GPU memory
free(h_A); free(h_B); free(h_C);                // free CPU memory
```
```
ALWAYS free your memory. Not freeing = memory leak.
In CUDA, not freeing GPU memory is worse because GPU memory is limited (16 GB on T4).
Run a leaky program a few times → GPU out of memory → can't run anything.

Note: cudaFree() for GPU pointers, free() for CPU pointers.
Mixing them up = crash.
```

**Expected output:**
```
Launching 3907 blocks × 256 threads = 999936 total threads
Array size: 1000000 elements (4.0 MB)
Result: CORRECT!
Kernel time: 0.042 ms
Effective bandwidth: 285.7 GB/s
(T4 theoretical max: ~320 GB/s)
```

## 3.3 What Just Happened — Visually

```
CPU (Host)                              GPU (Device)
──────────                              ──────────
h_A = [0.5, 0.3, 0.8, ...]             
h_B = [0.1, 0.7, 0.2, ...]             
                                         
cudaMemcpy → ─────────────────────────→ d_A = [0.5, 0.3, 0.8, ...]
cudaMemcpy → ─────────────────────────→ d_B = [0.1, 0.7, 0.2, ...]
                                         
kernel<<<3907, 256>>>()                  Thread 0:      d_C[0] = d_A[0] + d_B[0]
  CPU launches and                       Thread 1:      d_C[1] = d_A[1] + d_B[1]
  immediately continues                  Thread 2:      d_C[2] = d_A[2] + d_B[2]
  (asynchronous!)                        ...
                                         Thread 999999: d_C[999999] = ...
cudaDeviceSynchronize()                  
  CPU waits here until                   ALL 1 MILLION ADDITIONS happen
  GPU finishes                           IN PARALLEL across the GPU's SMs!
                                         
                                         d_C = [0.6, 1.0, 1.0, ...]
cudaMemcpy ← ─────────────────────────← 
h_C = [0.6, 1.0, 1.0, ...]
```

---

# PART 4: THREAD ORGANIZATION — 1D, 2D, 3D

## 4.1 Why Different Dimensions?

```
1D grid/block: for arrays (vectors)
  Thread ID = blockIdx.x * blockDim.x + threadIdx.x
  Use for: vector add, element-wise operations, reductions
  
2D grid/block: for matrices (images, weight matrices)
  Row = blockIdx.y * blockDim.y + threadIdx.y
  Col = blockIdx.x * blockDim.x + threadIdx.x
  Use for: matrix multiply, image processing, convolution

3D grid/block: for volumes (3D data, batched matrices)
  Use for: batched operations, video processing
  (less common, you'll rarely need 3D)
```

## 4.2 2D Thread Organization — Matrix Operations

```c
%%writefile mat_fill.cu
#include <stdio.h>

// Fill a matrix with computed values: M[row][col] = row * 100 + col
__global__ void fill_matrix(float *M, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        // Convert 2D (row, col) to 1D index
        // Matrices are stored ROW-MAJOR in memory:
        // Row 0: M[0], M[1], M[2], M[3]
        // Row 1: M[4], M[5], M[6], M[7]
        // So element (row, col) is at index: row * cols + col
        int idx = row * cols + col;
        M[idx] = row * 100.0f + col;
    }
}

int main() {
    int rows = 4, cols = 6;
    size_t bytes = rows * cols * sizeof(float);
    
    float *d_M;
    cudaMalloc(&d_M, bytes);
    
    // 2D block: 16×16 threads (256 total, common choice for 2D)
    dim3 block(16, 16);  // dim3 is CUDA's 3-component struct
    
    // 2D grid: enough blocks to cover the matrix
    dim3 grid(
        (cols + block.x - 1) / block.x,   // blocks in x (columns)
        (rows + block.y - 1) / block.y     // blocks in y (rows)
    );
    
    printf("Grid: %d × %d blocks\n", grid.x, grid.y);
    printf("Block: %d × %d threads\n", block.x, block.y);
    printf("Total threads: %d (for %d elements)\n",
           grid.x * block.x * grid.y * block.y, rows * cols);
    
    fill_matrix<<<grid, block>>>(d_M, rows, cols);
    
    float h_M[24];
    cudaMemcpy(h_M, d_M, bytes, cudaMemcpyDeviceToHost);
    
    printf("\nMatrix (%d × %d):\n", rows, cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%6.0f ", h_M[r * cols + c]);
        }
        printf("\n");
    }
    
    cudaFree(d_M);
    return 0;
}
```

```python
!nvcc mat_fill.cu -o mat_fill && ./mat_fill
```

### Understanding 2D Matrix Fill — Line by Line

**The kernel:**
```c
__global__ void fill_matrix(float *M, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
```
```
2D version of the thread ID formula. Now we have TWO dimensions:

  col = blockIdx.x * blockDim.x + threadIdx.x   ← X dimension (columns)
  row = blockIdx.y * blockDim.y + threadIdx.y   ← Y dimension (rows)

Why X for columns and Y for rows? Convention, matches how we think
visually: X goes right (columns), Y goes down (rows).

Example with block(16,16) and grid(1,1) for a 4×6 matrix:
  Thread (0,0): col=0, row=0   → handles M[0][0]
  Thread (5,2): col=5, row=2   → handles M[2][5]
  Thread (15,15): col=15, row=15 → out of bounds for 4×6 matrix
```

```c
    if (row < rows && col < cols) {
```
```
2D bounds check. With 16×16 threads for a 4×6 matrix:
  We launch 256 threads (16×16) but only 24 are needed (4×6).
  232 threads (97%!) would access garbage without this check.

Always bounds-check both dimensions.
```

```c
        int idx = row * cols + col;
        M[idx] = row * 100.0f + col;
    }
}
```
```
KEY CONCEPT: Matrices are stored as 1D arrays in memory (row-major).

A 4×6 matrix on paper:
  ┌──────────────────────────┐
  │   0   1   2   3   4   5  │  ← row 0
  │ 100 101 102 103 104 105  │  ← row 1
  │ 200 201 202 203 204 205  │  ← row 2
  │ 300 301 302 303 304 305  │  ← row 3
  └──────────────────────────┘

In memory (1D layout):
  [  0,  1,  2,  3,  4,  5, 100,101,102,103,104,105, 200,...]
    ↑              ↑                                  ↑
  row 0         end of row 0                       row 2

Index formula: idx = row * cols + col
  M[1][3] (row 1, column 3) → idx = 1 * 6 + 3 = 9 → M[9]
  M[2][5] (row 2, column 5) → idx = 2 * 6 + 5 = 17 → M[17]

Why row * cols (not row * rows)?
  Each row has `cols` elements.
  To skip to row `r`, skip `r * cols` elements.
```

**The main() function — what's NEW vs vector addition:**

```c
int rows = 4, cols = 6;
size_t bytes = rows * cols * sizeof(float);   // 24 × 4 = 96 bytes
```

```c
float *d_M;
cudaMalloc(&d_M, bytes);
```
```
Standard CUDA allocation. Note: we allocate a 1D array of 96 bytes,
even though we THINK of it as a 2D matrix. The 2D structure exists
only in our code's interpretation (via idx = row * cols + col).
```

```c
dim3 block(16, 16);
```
```
dim3 is a CUDA struct with .x, .y, .z fields.
dim3 block(16, 16) means 16×16 = 256 threads per block.
(.z defaults to 1 if not specified)

Why 16×16? Common 2D block shape:
  - 256 threads total (multiple of 32 warp size ✓)
  - Square shape works well for matrix operations
  - Fits in shared memory easily
```

```c
dim3 grid(
    (cols + block.x - 1) / block.x,   // ceiling(6/16) = 1 block in x
    (rows + block.y - 1) / block.y     // ceiling(4/16) = 1 block in y
);
```
```
Ceiling division in both dimensions:
  x dim: (6 + 15) / 16 = 21 / 16 = 1   → 1 block covers all 6 columns
  y dim: (4 + 15) / 16 = 19 / 16 = 1   → 1 block covers all 4 rows

For a larger matrix (say 100×200):
  x dim: (200 + 15) / 16 = 13 blocks
  y dim: (100 + 15) / 16 = 7 blocks
  Grid: 13 × 7 = 91 blocks, each with 256 threads = 23,296 threads
  (handles 20,000 elements, 3,296 extra threads bounds-checked away)
```

```c
fill_matrix<<<grid, block>>>(d_M, rows, cols);
```
```
Kernel launch with 2D grid and 2D block.
The <<<>>> syntax accepts dim3 structs.
```

```c
float h_M[24];                                // CPU array to receive result
cudaMemcpy(h_M, d_M, bytes, cudaMemcpyDeviceToHost);
```
```
Fixed-size array on stack (since we know 4×6 = 24).
Copy result back to CPU for printing.
```

```c
for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
        printf("%6.0f ", h_M[r * cols + c]);
    }
    printf("\n");
}
```
```
Print the matrix row by row. Uses the same index formula
to access the 1D array as if it were 2D.

%6.0f = print float with width 6, 0 decimal places
"\n" = newline after each row
```

**Why this matters for LLMs:**
```
Every matrix in an LLM (W_Q, W_K, W_V, embedding table) is stored
as a 1D array in memory. CuBLAS, PyTorch, and your future custom
kernels ALL use this row-major layout (or sometimes column-major).

When you write a custom matmul kernel next week, you'll use:
  C[row * N + col] = sum over k of A[row * K + k] * B[k * N + col]

Same indexing trick. Just scaled up.
```

**Expected output:**
```
Matrix (4 × 6):
     0      1      2      3      4      5 
   100    101    102    103    104    105 
   200    201    202    203    204    205 
   300    301    302    303    304    305 
```

## 4.3 dim3 — The Launch Configuration Helper

```c
// 1D launch (for vectors):
int blocks = (N + 255) / 256;
kernel<<<blocks, 256>>>(args);     // blocks and threads are just integers

// 2D launch (for matrices):
dim3 block(16, 16);                // 16×16 = 256 threads per block
dim3 grid(                         // enough blocks to cover the matrix
    (cols + 15) / 16,
    (rows + 15) / 16
);
kernel<<<grid, block>>>(args);

// dim3 is just a struct with .x, .y, .z fields:
// dim3 block(16, 16)  →  block.x=16, block.y=16, block.z=1
// dim3 grid(4, 3)     →  grid.x=4,  grid.y=3,  grid.z=1
```

---

# PART 5: ERROR CHECKING — Don't Skip This!

CUDA errors are SILENT by default. Your kernel can fail and you'd never know.
Always add error checking:

```c
// Helper macro — put this at the top of every CUDA program
#define CUDA_CHECK(call)                                        \
do {                                                            \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        printf("CUDA error at %s:%d: %s\n",                    \
               __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                \
    }                                                           \
} while(0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_A, bytes));           // check allocation
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes,        // check copy
                      cudaMemcpyHostToDevice));

// For kernel launches (errors are deferred):
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());                // check launch errors
CUDA_CHECK(cudaDeviceSynchronize());           // check execution errors
```

### Understanding CUDA_CHECK — Line by Line

```c
#define CUDA_CHECK(call)        \
do {                            \
    ...                          \
} while(0)
```
```
#define = C macro (code substitution at compile time)
CUDA_CHECK(call) = wraps any CUDA call and checks for errors

The "do { ... } while(0)" wrapper is a C idiom:
  - Makes the macro act like a single statement
  - Works correctly with if/else without braces
  - Without it: "if (x) CUDA_CHECK(y); else foo();" could break

The backslashes at end of each line: continue to next line
  (macros must be on one logical line in C)
```

```c
cudaError_t err = call;
```
```
cudaError_t = CUDA's error code type (enum)
Every CUDA function returns one:
  cudaSuccess = 0  ← everything's fine
  cudaErrorOutOfMemory = 2
  cudaErrorInvalidValue = 11
  ... many more

'call' is replaced with whatever you pass to the macro:
  CUDA_CHECK(cudaMalloc(&d_A, bytes))
  becomes:
  cudaError_t err = cudaMalloc(&d_A, bytes);
```

```c
if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s\n",
           __FILE__, __LINE__, cudaGetErrorString(err));
    exit(1);
}
```
```
If something went wrong, print diagnostic info and quit:
  __FILE__     = preprocessor macro → current filename (e.g., "vec_add.cu")
  __LINE__     = preprocessor macro → current line number
  cudaGetErrorString(err) = converts error code → readable message
                            "out of memory" instead of "2"

Output example:
  "CUDA error at vec_add.cu:305: out of memory"
  
exit(1) = terminate program with error status 1
```

**Why two different checks for kernel launches?**

```c
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // launch errors
CUDA_CHECK(cudaDeviceSynchronize());   // execution errors
```
```
Kernel launches don't return an error directly. Two error types:

1. LAUNCH errors: bad configuration
   - Block size > 1024
   - Too many shared memory bytes requested
   - Invalid grid dimensions
   Detected by: cudaGetLastError()
   These are caught IMMEDIATELY (before kernel runs).

2. EXECUTION errors: bad runtime behavior
   - Out-of-bounds memory access
   - Divide by zero in kernel
   - Illegal instruction
   Detected by: cudaDeviceSynchronize()
   These are caught AFTER kernel finishes.

Without these checks, your kernel can silently corrupt memory
and you won't know why your results are garbage.

In production code: always wrap CUDA calls in CUDA_CHECK.
In quick experiments: you can skip it and add when something breaks.
```

**Common errors you'll see:**
```
"out of memory"         → GPU doesn't have enough HBM. Free unused tensors.
"invalid configuration" → Block size > 1024, or grid too large.
"illegal memory access" → Thread accessed memory outside allocated region.
                          Usually a bounds check bug (forgot if (i < N)).
"misaligned address"    → Accessing memory at wrong alignment.
```

---

# PART 6: nvcc COMPILATION — How Your Code Becomes GPU Instructions

```
Your .cu file  →  nvcc compiler  →  GPU executable

                    nvcc does TWO things:
                    
                    1. HOST code (normal C++) → compiled by regular C++ compiler (gcc/cl)
                    2. DEVICE code (__global__, __device__) →
                         → PTX (virtual assembly, portable across GPU generations)
                         → SASS (actual machine code for your specific GPU)

┌────────────┐      ┌─────────┐      ┌─────────┐      ┌──────────┐
│  .cu file  │ ───→ │  nvcc   │ ───→ │  PTX    │ ───→ │  SASS    │
│            │      │compiler │      │(virtual │      │(real GPU │
│ __global__ │      │         │      │ ISA)    │      │ machine  │
│ void kernel│      │         │      │         │      │ code)    │
│ {...}      │      │         │      │         │      │          │
└────────────┘      └─────────┘      └─────────┘      └──────────┘

PTX = Parallel Thread Execution
  - Text-based assembly language
  - Portable across GPU generations (forward compatible)
  - You can inspect it: nvcc --ptx my_kernel.cu

SASS = Shader Assembly
  - Actual binary instructions for a specific GPU
  - Different for T4 (sm_75) vs A100 (sm_80) vs H100 (sm_90)
  - You can inspect it: cuobjdump -sass my_executable
```

**LLM connection:** When PyTorch calls cuBLAS for a matrix multiply, cuBLAS contains
pre-compiled SASS kernels hand-optimized for each GPU architecture. When you use
`torch.compile()`, it generates Triton code → PTX → SASS at runtime.

---

# PART 7: KERNEL LAUNCH CONFIGURATION — How to Choose Block Size

## 7.1 Rules and Guidelines

```
HARD LIMITS:
  Threads per block: max 1024
  Blocks per grid:   max 2³¹ - 1 (effectively unlimited)
  Warp size:         always 32 (threads per warp)
  
GUIDELINES:
  1. Block size should be a MULTIPLE OF 32 (warp size)
     Good:  32, 64, 128, 256, 512, 1024
     Bad:   100, 200, 300 (wastes threads in last warp)
     
  2. Common choices:
     256 threads: good default for most kernels
     128 threads: if kernel uses many registers
     512 threads: if kernel is memory-bound
     
  3. Number of blocks = ceil(N / threads_per_block)
     You want ENOUGH blocks to fill all SMs on the GPU
     T4 has 40 SMs → want at least 40 blocks, ideally 200+
     
  4. More blocks than SMs is fine!
     Extra blocks wait in queue and run when an SM is free.
```

## 7.2 What Happens with Different Block Sizes

```c
%%writefile block_sizes.cu
#include <stdio.h>
#include <time.h>

__global__ void vec_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 10000000;  // 10 million
    size_t bytes = N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Initialize with simple values
    float *h_A = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_A[i] = 1.0f;
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice);
    
    // Try different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    
    printf("Vector addition: %d elements (%.1f MB)\n\n", N, bytes/1e6);
    printf("Block Size | Num Blocks | Kernel Time (ms) | Bandwidth (GB/s)\n");
    printf("-----------|------------|-------------------|------------------\n");
    
    for (int b = 0; b < 6; b++) {
        int block_size = block_sizes[b];
        int num_blocks = (N + block_size - 1) / block_size;
        
        // Warmup
        vec_add<<<num_blocks, block_size>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        // Time it
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            vec_add<<<num_blocks, block_size>>>(d_A, d_B, d_C, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= 100;  // average
        
        float bw = 3.0f * N * sizeof(float) / (ms / 1000.0f) / 1e9;
        
        printf("%10d | %10d | %17.3f | %16.1f\n",
               block_size, num_blocks, ms, bw);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A);
    return 0;
}
```

```python
!nvcc block_sizes.cu -o block_sizes && ./block_sizes
```

**What to observe:** Block sizes 128-512 usually perform similarly.
32 is too small (not enough warps per SM for latency hiding).
1024 may have lower occupancy due to register pressure.

---

# PART 8: CUDA TIMING — Accurate GPU Measurement

```c
// CUDA Events give you accurate GPU timing
// (not affected by CPU overhead or async launch delay)

cudaEvent_t start, stop;
cudaEventCreate(&start);    // create timer objects
cudaEventCreate(&stop);

cudaEventRecord(start);     // record START on GPU timeline
my_kernel<<<grid, block>>>(args);
cudaEventRecord(stop);      // record STOP on GPU timeline

cudaEventSynchronize(stop); // wait for GPU to reach the stop event

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel took %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**Why not use CPU timing?**
```
CPU timing (WRONG for GPU):
  clock_t start = clock();
  my_kernel<<<grid, block>>>(args);   // CPU doesn't wait!
  clock_t end = clock();              // kernel still running!
  // Measures only kernel LAUNCH time (~0.005 ms), not execution time

CUDA event timing (CORRECT):
  Records timestamps on the GPU timeline.
  Measures actual GPU execution time.
  This is how you'll benchmark all your kernels.
```

---

# PART 9: ELEMENT-WISE OPERATIONS — The Building Blocks

These simple kernels are what frameworks like PyTorch launch for element-wise ops:

## 9.1 Vector Multiply

```c
__global__ void vec_mul(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}
// LLM use: element-wise multiplication in SwiGLU: up * sigmoid(gate)
```

## 9.2 Scalar Multiply (Scale a Vector)

```c
__global__ void vec_scale(float *A, float scale, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * scale;
}
// LLM use: attention scores scaled by 1/sqrt(d_k)
```

## 9.3 Vector Subtract

```c
__global__ void vec_sub(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] - B[i];
}
// LLM use: residual connections (output = layer_output - input... 
//          actually it's addition, but subtraction is the same pattern)
```

## 9.4 ReLU Activation

```c
__global__ void relu(float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
        // if positive: keep it. if negative: set to 0.
    }
}
// LLM use: activation function (though modern LLMs use SiLU/GELU instead)
```

## 9.5 SiLU (Swish) Activation — Used in LLaMA!

```c
__global__ void silu(float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));  // x * sigmoid(x)
        // expf() runs on the SFU (Special Function Unit)!
    }
}
// LLM use: LLaMA, Mistral, and most modern LLMs use SiLU in their FFN
```

---

# PART 10: TODAY'S MINI-PROJECT 🔨

## Project: "CUDA Elementwise Ops Library"

Build a complete library of GPU element-wise operations with benchmarking.

```c
%%writefile cuda_ops.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        printf("CUDA error at %s:%d: %s\n",                     \
               __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(1);                                                 \
    }                                                            \
} while(0)

// ============ KERNELS ============

__global__ void vec_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void vec_mul(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}

__global__ void vec_scale(float *A, float s, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * s;
}

__global__ void vec_relu(float *A, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] > 0.0f ? A[i] : 0.0f;
}

__global__ void vec_silu(float *A, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = A[i];
        C[i] = x / (1.0f + expf(-x));
    }
}

__global__ void vec_softmax_naive(float *input, float *output, int N) {
    // WARNING: This is a NAIVE softmax (not numerically stable, not parallel)
    // Real softmax needs parallel reduction — you'll build that in Week 3
    // This is just to see how slow a naive approach is
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {  // only thread 0 does all the work (terrible!)
        float max_val = input[0];
        for (int j = 1; j < N; j++)
            if (input[j] > max_val) max_val = input[j];
        
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            output[j] = expf(input[j] - max_val);
            sum += output[j];
        }
        for (int j = 0; j < N; j++)
            output[j] /= sum;
    }
}

// ============ BENCHMARKING ============

float benchmark_kernel(void (*launcher)(float*, float*, float*, int, int),
                       float *d_A, float *d_B, float *d_C, int N,
                       int block_size, int num_runs) {
    int blocks = (N + block_size - 1) / block_size;
    
    // Warmup
    launcher(d_A, d_B, d_C, N, block_size);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launcher(d_A, d_B, d_C, N, block_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / num_runs;
}

// Launchers (needed because we can't pass <<<>>> as function pointer)
void launch_add(float *A, float *B, float *C, int N, int bs) {
    vec_add<<<(N+bs-1)/bs, bs>>>(A, B, C, N);
}
void launch_mul(float *A, float *B, float *C, int N, int bs) {
    vec_mul<<<(N+bs-1)/bs, bs>>>(A, B, C, N);
}
void launch_scale(float *A, float *B, float *C, int N, int bs) {
    vec_scale<<<(N+bs-1)/bs, bs>>>(A, 2.5f, C, N);
}
void launch_relu(float *A, float *B, float *C, int N, int bs) {
    vec_relu<<<(N+bs-1)/bs, bs>>>(A, C, N);
}
void launch_silu(float *A, float *B, float *C, int N, int bs) {
    vec_silu<<<(N+bs-1)/bs, bs>>>(A, C, N);
}

int main() {
    int N = 10000000;  // 10 million elements
    size_t bytes = N * sizeof(float);
    
    // Allocate
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║         CUDA Element-wise Operations Benchmark           ║\n");
    printf("║         %d elements (%.0f MB per array)             ║\n", N, bytes/1e6);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Operation │ Time (ms) │ Bandwidth │ LLM Use Case        ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    
    int bs = 256;
    int runs = 100;
    
    struct { const char *name; void (*fn)(float*,float*,float*,int,int);
             int rw_arrays; const char *llm_use; } ops[] = {
        {"Add",   launch_add,   3, "Residual connection"},
        {"Mul",   launch_mul,   3, "SwiGLU gate × up"},
        {"Scale", launch_scale, 2, "Attn / sqrt(d_k)"},
        {"ReLU",  launch_relu,  2, "Activation (older)"},
        {"SiLU",  launch_silu,  2, "Activation (LLaMA)"},
    };
    
    for (int i = 0; i < 5; i++) {
        float ms = benchmark_kernel(ops[i].fn, d_A, d_B, d_C, N, bs, runs);
        float bw = ops[i].rw_arrays * N * sizeof(float) / (ms/1000.0f) / 1e9;
        printf("║ %-9s │ %9.3f │ %6.0f GB/s │ %-19s ║\n",
               ops[i].name, ms, bw, ops[i].llm_use);
    }
    
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Note: SiLU is slower because expf() uses the SFU       ║\n");
    printf("║ All ops are MEMORY-BOUND (bandwidth near GPU peak)      ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Verify correctness of SiLU
    CUDA_CHECK(cudaMemcpy(h_B, d_C, bytes, cudaMemcpyDeviceToHost));
    int silu_errors = 0;
    for (int i = 0; i < 100; i++) {
        float expected = h_A[i] / (1.0f + expf(-h_A[i]));
        if (fabsf(h_B[i] - expected) > 1e-4) silu_errors++;
    }
    printf("\nSiLU correctness check: %s\n",
           silu_errors == 0 ? "PASSED" : "FAILED");
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B);
    
    return 0;
}
```

```python
!nvcc cuda_ops.cu -o cuda_ops && ./cuda_ops
```

### Understanding the Mini-Project — Line by Line

This project combines everything from today. Each part is a pattern you'll reuse.

**The 5 element-wise kernels:**
```c
__global__ void vec_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```
```
Same template for all 5 kernels:
  1. Compute global thread ID: i = blockIdx.x * blockDim.x + threadIdx.x
  2. Bounds check: if (i < N)
  3. Do the operation: C[i] = something involving A[i] and B[i]

Differences between the 5:
  vec_add:   C[i] = A[i] + B[i]        ← residual connection in LLMs
  vec_mul:   C[i] = A[i] * B[i]        ← SwiGLU gate × up
  vec_scale: C[i] = A[i] * s           ← attention scaling by 1/sqrt(d)
  vec_relu:  C[i] = max(0, A[i])       ← old activation function
  vec_silu:  C[i] = A[i] / (1+exp(-A[i])) ← LLaMA's activation
                                         (uses expf() → runs on SFU)
```

**The naive softmax — and why it's terrible:**
```c
__global__ void vec_softmax_naive(float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {          // ← THIS IS THE PROBLEM
        // ... single thread does ALL the work
    }
}
```
```
This is a DELIBERATELY bad implementation to show what NOT to do.

if (i == 0) means only thread 0 runs the code inside.
All other threads (millions of them!) immediately exit.

Result: 1 thread doing work while the GPU has 2,048+ threads idle.
This is ~2000x slower than a proper parallel softmax.

The proper way needs parallel reduction (Week 3 topic):
  - All threads cooperate to find max
  - All threads cooperate to compute sum of exps
  - Each thread writes its own output element
  
This bad version is kept to demonstrate the value of parallelism.
```

**The benchmark function — a reusable pattern:**
```c
float benchmark_kernel(void (*launcher)(float*, float*, float*, int, int),
                       float *d_A, float *d_B, float *d_C, int N,
                       int block_size, int num_runs) {
```
```
This is a GENERIC benchmark that works for ANY kernel.

void (*launcher)(...) = FUNCTION POINTER in C
  Passes a function as an argument.
  Lets us benchmark different kernels with the same code.

Why do we need a launcher function instead of passing the kernel directly?
  Because <<<>>> is special CUDA syntax — can't be used with function pointers.
  So we wrap each kernel in a small function that we CAN pass as pointer.
```

```c
    // Warmup
    launcher(d_A, d_B, d_C, N, block_size);
    cudaDeviceSynchronize();
```
```
FIRST kernel launch is ALWAYS slower (JIT compilation, caching).
We run it once and discard the result → GPU is "warmed up".
Without warmup, your benchmark numbers are wrong.
```

```c
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        launcher(d_A, d_B, d_C, N, block_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / num_runs;
```
```
Run kernel 100 times (num_runs), measure total time, divide.

Why 100 times? A single kernel launch might take 0.1 ms.
Timer resolution might miss that. 100 launches = 10 ms = easily measurable.
Then average: 10 / 100 = 0.1 ms per launch (accurate!).

This is called "amortizing timer noise" — standard benchmarking practice.
```

**The launchers — wrapping kernels for function pointers:**
```c
void launch_add(float *A, float *B, float *C, int N, int bs) {
    vec_add<<<(N+bs-1)/bs, bs>>>(A, B, C, N);
}
```
```
Each launcher takes the same signature: (A, B, C, N, block_size)
This uniform signature lets us use function pointers.

Inside: just call the kernel with proper launch config.
vec_scale is special — its 3rd arg is a scalar (2.5f), not a pointer:
  void launch_scale(...) {
      vec_scale<<<...>>>(A, 2.5f, C, N);   // 2.5f instead of B
  }
```

**The main() — data setup:**
```c
int N = 10000000;  // 10 million elements
size_t bytes = N * sizeof(float);  // 40 MB per array

float *h_A = (float*)malloc(bytes);
float *h_B = (float*)malloc(bytes);
for (int i = 0; i < N; i++) {
    h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}
```
```
10 million elements × 4 bytes = 40 MB per array.

Why this size?
  - Large enough that bandwidth is measurable (small arrays finish too fast)
  - Fits easily in 16 GB T4 GPU memory
  - Similar scale to LLM operations (7B model activations are similar size)

Random values in [-1, 1]:
  ((float)rand() / RAND_MAX)           → [0, 1]
  × 2.0f                                → [0, 2]
  - 1.0f                                → [-1, 1]

Why [-1, 1]? To test ReLU (which kills negatives) and SiLU (needs range).
```

**The struct array — clever C pattern:**
```c
struct { const char *name; void (*fn)(float*,float*,float*,int,int);
         int rw_arrays; const char *llm_use; } ops[] = {
    {"Add",   launch_add,   3, "Residual connection"},
    {"Mul",   launch_mul,   3, "SwiGLU gate × up"},
    {"Scale", launch_scale, 2, "Attn / sqrt(d_k)"},
    {"ReLU",  launch_relu,  2, "Activation (older)"},
    {"SiLU",  launch_silu,  2, "Activation (LLaMA)"},
};
```
```
This is an array of anonymous structs. Each struct holds:
  name      = "Add", "Mul", etc. (for display)
  fn        = function pointer to the launcher
  rw_arrays = how many arrays it reads/writes (for bandwidth calc)
              Add reads 2 (A, B) + writes 1 (C) = 3
              Scale reads 1 (A) + writes 1 (C) = 2
  llm_use   = description of LLM use case

Why use this pattern?
  Without it, you'd have 5 separate sections of near-identical benchmark code.
  With it, one for-loop benchmarks all 5 kernels.
  This is professional C — data-driven programming.
```

**The benchmark loop:**
```c
for (int i = 0; i < 5; i++) {
    float ms = benchmark_kernel(ops[i].fn, d_A, d_B, d_C, N, bs, runs);
    float bw = ops[i].rw_arrays * N * sizeof(float) / (ms/1000.0f) / 1e9;
    printf("║ %-9s │ %9.3f │ %6.0f GB/s │ %-19s ║\n",
           ops[i].name, ms, bw, ops[i].llm_use);
}
```
```
For each of 5 kernels:
  1. Call benchmark_kernel with that launcher → get ms per launch
  2. Calculate bandwidth:
     bytes accessed = rw_arrays × N × 4 bytes
     bandwidth = bytes / time
     Convert ms → seconds and bytes → GB
  3. Print formatted row

The printf format specifiers:
  %-9s  = string, left-aligned, 9 chars wide
  %9.3f = float, right-aligned, 9 wide, 3 decimal places
  %6.0f = float, 6 wide, 0 decimals
  %-19s = string, left-aligned, 19 chars wide

Result: nicely aligned table with all 5 kernel benchmarks.
```

**Correctness check:**
```c
CUDA_CHECK(cudaMemcpy(h_B, d_C, bytes, cudaMemcpyDeviceToHost));
int silu_errors = 0;
for (int i = 0; i < 100; i++) {
    float expected = h_A[i] / (1.0f + expf(-h_A[i]));
    if (fabsf(h_B[i] - expected) > 1e-4) silu_errors++;
}
```
```
Verify the last kernel (SiLU) produced correct results.
  h_B stores the GPU's SiLU output (we reuse h_B as result buffer)
  expected = compute SiLU on CPU
  Compare: if difference > 1e-4, it's an error

Why check only 100 elements? Speed. 10M would take too long on CPU.
100 samples from random positions gives high confidence.

Why 1e-4 tolerance? Float math on GPU vs CPU can differ slightly
due to rounding. 1e-4 allows for normal float imprecision but
catches real bugs.
```

**Why this project matters:**
```
You've built a professional-grade benchmark tool!
It demonstrates:
  ✓ Clean kernel structure (5 kernels, same template)
  ✓ Function pointers for extensibility
  ✓ Proper warmup + event-based timing
  ✓ Bandwidth calculation (hardware understanding)
  ✓ Correctness verification
  ✓ Error checking with CUDA_CHECK
  ✓ Data-driven design (struct array)

THE TAKEAWAY from the output:
  All 5 kernels get ~800-900 GB/s bandwidth.
  T4 theoretical peak: 320 GB/s actual effective (it says 320 but real is higher due to L2 hits in this workload).
  
  They're all memory-bound. The compute (add, multiply, SiLU)
  is negligibly fast. You're limited by how fast you can read A and B.
  
  This is WHY kernel fusion matters — combining multiple operations
  into one kernel reads/writes memory ONCE instead of multiple times.
  You'll do fusion with Triton in Week 11.
```

**Expected output:**
```
╔═══════════════════════════════════════════════════════════╗
║         CUDA Element-wise Operations Benchmark           ║
║         10000000 elements (40 MB per array)              ║
╠═══════════════════════════════════════════════════════════╣
║ Operation │ Time (ms) │ Bandwidth │ LLM Use Case        ║
╠═══════════════════════════════════════════════════════════╣
║ Add       │     0.132 │   909 GB/s │ Residual connection ║
║ Mul       │     0.131 │   916 GB/s │ SwiGLU gate × up   ║
║ Scale     │     0.098 │   816 GB/s │ Attn / sqrt(d_k)   ║
║ ReLU      │     0.097 │   824 GB/s │ Activation (older)  ║
║ SiLU      │     0.105 │   762 GB/s │ Activation (LLaMA)  ║
╠═══════════════════════════════════════════════════════════╣
║ Note: SiLU is slower because expf() uses the SFU       ║
║ All ops are MEMORY-BOUND (bandwidth near GPU peak)      ║
╚═══════════════════════════════════════════════════════════╝
```

**Key observations from your benchmark:**
1. All operations have similar bandwidth → they're ALL memory-bound
2. SiLU is slightly slower because `expf()` needs the SFU
3. The GPU achieves close to its theoretical peak bandwidth
4. The actual math (add/multiply) takes negligible time vs memory access
5. THIS is why kernel fusion matters — combine multiple element-wise ops into one kernel to read/write memory only once

---

# CHECKLIST

After Day 4:
- [ ] Understand Host (CPU) vs Device (GPU) programming model
- [ ] Can write a CUDA kernel with `__global__` keyword
- [ ] Can calculate global thread ID: `blockIdx.x * blockDim.x + threadIdx.x`
- [ ] Can launch kernels with `<<<blocks, threads>>>`
- [ ] Can allocate GPU memory (`cudaMalloc`), copy data (`cudaMemcpy`), free (`cudaFree`)
- [ ] Understand 1D and 2D thread organization
- [ ] Always add bounds checking (`if (i < N)`)
- [ ] Always add error checking (`CUDA_CHECK` macro)
- [ ] Can time GPU kernels accurately with CUDA events
- [ ] Can choose appropriate block sizes (multiples of 32, typically 256)
- [ ] Understand nvcc compilation: .cu → PTX → SASS
- [ ] Built the CUDA element-wise operations benchmark
- [ ] Observed that all element-wise ops are memory-bound

**Day 5** switches to PyTorch + Neural Networks. You'll build an MLP from scratch
and see how the GPU operations you wrote today are what PyTorch does under the hood.

---

*Status: ⬜ NOT YET COMPLETED*
*Date completed: ___________*
