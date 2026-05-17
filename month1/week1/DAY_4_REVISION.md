# W1D4 Revision — CUDA Programming, Your First GPU Kernels
## ⏱️ 10-minute speed-revision sheet

> **One-line summary:** A CUDA program is split between **host** (CPU code) and **device** (GPU code marked `__global__`). You launch a kernel with `<<<grid, block>>>`, and inside the kernel each thread computes its own global ID with `blockIdx.x * blockDim.x + threadIdx.x` to figure out which data to touch.

---

## 🧠 30-Second Mental Model

```
CPU (host)                          GPU (device)
─────────                          ─────────
1. Allocate host memory             - Has thousands of cores
2. cudaMalloc on device             - Has its own separate memory
3. cudaMemcpy host → device         - Receives kernels via <<<>>>
4. Launch kernel <<<grid, block>>>  - Each thread runs the same code
5. cudaDeviceSynchronize()             on different data (SIMT)
6. cudaMemcpy device → host
7. cudaFree

The GPU does work; the CPU directs it.
```

If you remember **only one thing**: every CUDA program is `[allocate → copy in → launch → sync → copy out → free]`, and inside the kernel each thread uses `blockIdx.x * blockDim.x + threadIdx.x` to find its data.

---

## 1️⃣ Host-Device Model — The Two Computers (60 sec)

| | Host (CPU) | Device (GPU) |
|---|---|---|
| Code marker | normal C/C++ | `__global__` (kernel) or `__device__` (helper) |
| Memory | RAM (DDR) | HBM (GDDR/HBM2/HBM3) |
| Allocation | `malloc`, `new` | `cudaMalloc` |
| Free | `free`, `delete` | `cudaFree` |
| Cores | few, complex | thousands, simple |

### Memory transfer (THE most important pattern):
```c
cudaMemcpy(dst_ptr, src_ptr, num_bytes, direction);
// directions: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice
```
- This is **slow** (PCIe-bound). Minimize transfers; keep data on the GPU as long as possible.

### Memory access rule:
- Host code **cannot** dereference a device pointer
- Device code **cannot** dereference a host pointer
- They live in separate address spaces (Unified Memory hides this, but understand the basics first)

---

## 2️⃣ Your First Kernel Anatomy (60 sec)

```c
__global__ void hello_kernel() {
    int tid = threadIdx.x;
    printf("Hello from thread %d\n", tid);
}

int main() {
    hello_kernel<<<2, 4>>>();   // launch: 2 blocks × 4 threads/block = 8 threads
    cudaDeviceSynchronize();    // wait for GPU
    return 0;
}
```

### Function qualifiers:
- `__global__` → kernel, called from host, runs on device
- `__device__` → helper called from device only
- `__host__` → normal CPU function (default if nothing specified)

### Launch syntax:
```c
my_kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(args);
```
- `grid_dim`: number of blocks (can be `int`, `dim3`)
- `block_dim`: threads per block (max 1024 total)
- shared_mem_bytes (optional): dynamic shared memory size
- stream (optional): defaults to stream 0

### Built-in variables (only valid inside `__global__`):
| Variable | Meaning |
|---|---|
| `threadIdx.x/y/z` | Thread's position within its block |
| `blockIdx.x/y/z` | Block's position within the grid |
| `blockDim.x/y/z` | Number of threads per block |
| `gridDim.x/y/z` | Number of blocks in the grid |

---

## 3️⃣ The Global Thread ID Formula — MEMORIZE (30 sec)

```c
// 1D
int gid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D (e.g., for matrices)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Always check bounds!
if (gid >= N) return;
```

**Mental picture:** all blocks come first by `blockIdx`, each block contains `blockDim` threads. The global thread ID is just `which block × block size + position in block`.

---

## 4️⃣ Vector Addition — The Classic First Kernel (60 sec)

```c
__global__ void vec_add(float* A, float* B, float* C, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) C[gid] = A[gid] + B[gid];
}

int main() {
    int N = 1 << 20;                                // ~1M elements
    size_t bytes = N * sizeof(float);
    
    float *h_A, *h_B, *h_C;                         // host pointers
    h_A = (float*)malloc(bytes);
    // ... initialize h_A, h_B ...
    
    float *d_A, *d_B, *d_C;                         // device pointers
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;       // ceiling division
    vec_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}
```

### Why the bounds check `if (gid < N)`?
- Block count is `(N + 255)/256` (ceiling division) → may launch slightly more threads than `N`
- Without the check, the extra threads would write past the array → memory corruption

---

## 5️⃣ Thread Organization — 1D, 2D, 3D (45 sec)

### When to use each:
- **1D**: vectors, sequences, embeddings (`vec_add`, normalization)
- **2D**: matrices, images, attention scores (`Q @ K^T` of shape `[seq, seq]`)
- **3D**: 3D tensors, video frames, batched matrices

### `dim3` for clean launch configs:
```c
dim3 block(16, 16);                                  // 16×16 = 256 threads/block
dim3 grid((cols + 15)/16, (rows + 15)/16);          // ceil divides
matrix_kernel<<<grid, block>>>(d_A, rows, cols);
```

### Hardware limits:
- Threads per block: **max 1024** (most use 256 or 512)
- Blocks per grid: huge (`2^31 - 1` per dim) — practically unlimited

---

## 6️⃣ Block Size Choice (45 sec)

### Rules of thumb:
- Use a **multiple of 32** (warp size) — otherwise some warps run with inactive threads
- Common picks: **128, 256, 512**. 256 is the safe default.
- **Never go below 32** (bottom of one warp) — too small, low occupancy
- **Don't always max it out at 1024** — large blocks use more registers/shared memory per block, may *reduce* occupancy

### What changes with block size:
| Block size | Pros | Cons |
|---|---|---|
| 32 (1 warp) | Smallest unit, low resource use per block | Almost no latency hiding within block |
| 256 | Sweet spot, good occupancy | — |
| 1024 | Max threads/block | High register pressure may limit blocks/SM |

---

## 7️⃣ Error Checking + Timing (30 sec)

### CUDA_CHECK macro (always wrap CUDA calls):
```c
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

CUDA_CHECK(cudaMalloc(&d_A, bytes));
```

### After a kernel launch, you ALSO need:
```c
my_kernel<<<...>>>(...);
CUDA_CHECK(cudaGetLastError());          // catch launch errors
CUDA_CHECK(cudaDeviceSynchronize());     // catch runtime errors
```

### Timing kernels properly (CUDA events, not CPU timer):
```c
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
my_kernel<<<...>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);
```

---

## 8️⃣ nvcc Compilation Flow (15 sec)

```
.cu file ──nvcc──→  splits host (.cpp) + device code
                  → compiles device code to PTX (virtual ISA)
                  → JIT-compiles PTX to SASS at runtime for the target GPU
                  → links host code with the device binary
                  → outputs single executable
```

### `nvcc` flags you'll see:
- `-arch=sm_75` (compute capability 7.5 = Turing/T4)
- `-O3` (optimize)
- `-Xcompiler "-Wall"` (pass flags to host compiler)

---

## 9️⃣ Element-wise Kernels You Can Write Today (15 sec)

All follow the same pattern: compute `gid`, bounds-check, do one op:

```c
__global__ void vec_mul(float* A, float* B, float* C, int N) { ... C[i] = A[i]*B[i]; }
__global__ void vec_scale(float* A, float s, float* C, int N) { ... C[i] = s*A[i]; }
__global__ void vec_relu(float* A, float* C, int N) { ... C[i] = fmaxf(0.0f, A[i]); }
__global__ void vec_silu(float* A, float* C, int N) { ... C[i] = A[i] / (1.0f + expf(-A[i])); }
```

**SiLU is the activation in LLaMA's FFN.**

---

## 🔍 Quick Recall (60 sec)

1. What does `__global__` mean? *(kernel — called from host, runs on device)*
2. What's the formula for global thread ID in 1D? *(`blockIdx.x * blockDim.x + threadIdx.x`)*
3. Why do you need `if (gid < N)` in kernels? *(ceiling block count may launch extra threads)*
4. What's the max threads per block? *(1024)*
5. Why must block size be a multiple of 32? *(warp size; otherwise warps run with inactive threads)*
6. What does `cudaMemcpy` do? *(copies between host and device memory)*
7. Why use `cudaEvent` for timing instead of CPU clock? *(GPU runs async; CPU clock doesn't see kernel completion)*
8. What's the difference between `__device__` and `__global__`? *(`__device__` callable from device only; `__global__` callable from host)*
9. What's PTX? *(virtual GPU ISA; nvcc emits it, runtime JIT-compiles it to SASS for the target GPU)*
10. What's `cudaDeviceSynchronize()` for? *(blocks CPU until GPU finishes pending work)*

---

## 🎯 If You Remember Only Three Things

1. **The CUDA workflow is always the same:** allocate device memory → copy in → launch kernel → sync → copy out → free. Every kernel you ever write follows this skeleton.
2. **`gid = blockIdx.x * blockDim.x + threadIdx.x` is THE formula.** Inside a kernel, this is how each of the millions of threads figures out which piece of data belongs to it.
3. **Block size of 256 is the safe default**, must be a multiple of 32 (warp size), max 1024. Tuning this is one of the levers for performance — but for now: 256 just works.

---

*Revision file generated from `DAY_4.md`. For deep dive + full worked examples, see the original DAY_4.md.*
