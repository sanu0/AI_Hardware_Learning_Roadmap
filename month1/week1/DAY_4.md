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
