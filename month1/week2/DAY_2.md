# Week 2, Day 2: Shared Memory, Constant Memory, and Bank Conflicts
## The On-Chip Scratchpad That Makes Fast CUDA Kernels Possible

**Time: ~2.5-3 hours**
**Setup: Google Colab with GPU runtime**

---

## Today's Mental Model

Yesterday you learned the painful truth:

> Global memory is huge and high-bandwidth, but it is far away.

Today you learn the escape hatch:

> Shared memory is tiny, explicit, on-chip storage that lets a block reuse data
> without going back to HBM again and again.

By the end of today, you should be able to look at a CUDA kernel and ask:

1. **Is this data reused by multiple threads in the same block?**
2. **Can I load it once from global memory into shared memory?**
3. **Will shared-memory bank conflicts secretly ruin the speedup?**
4. **Is the data actually read-only and uniform enough for constant memory?**
5. **Did I accidentally spill private thread data into local memory?**

This is the day where CUDA starts feeling less like "launch many threads" and
more like **designing a memory choreography**.

---

# PART 1: THE STORY

## 1.1 The Problem With Reading The Same Data Again

Imagine 256 threads in a CUDA block doing a stencil operation:

```c
out[i] = in[i-1] + in[i] + in[i+1];
```

What this snippet is doing:

- One output value is computed from three input values.
- `in[i]` is the center element.
- `in[i-1]` and `in[i+1]` are the left and right neighbors.
- This is a stencil pattern, and it naturally creates data reuse because neighboring
  threads need overlapping input values.

Thread 10 needs:

```text
in[9], in[10], in[11]
```

Thread 11 needs:

```text
in[10], in[11], in[12]
```

Thread 12 needs:

```text
in[11], in[12], in[13]
```

Notice the overlap. The same input values are read again and again.

```
Thread 10:       in[9]   in[10]  in[11]
Thread 11:               in[10]  in[11]  in[12]
Thread 12:                       in[11]  in[12]  in[13]
                                  ↑
                                  read by multiple threads
```

If every thread reads directly from global memory, HBM sees many repeated reads.
That is waste.

The better idea:

1. Load a chunk of `in[]` once from global memory into shared memory.
2. Let all threads in the block reuse that chunk from fast on-chip SRAM.
3. Write final results back to global memory.

That is shared memory.

## 1.2 The Restaurant Kitchen Analogy

Global memory is the warehouse. Huge, far away, slow to fetch from.

Shared memory is the prep table inside the kitchen. Tiny, close, fast.

```
  BAD KITCHEN:

  Chef 0 needs tomato ───────┐
  Chef 1 needs tomato ───────┼── each chef runs to the warehouse
  Chef 2 needs tomato ───────┘

  GOOD KITCHEN:

  One chef brings tomatoes from warehouse → prep table
  Everyone cooks from the prep table
```

CUDA version:

```
  BAD KERNEL:
  every thread repeatedly reads global memory

  GOOD KERNEL:
  block cooperatively loads tile into shared memory
  threads reuse shared tile
```

Shared memory does not make global memory faster. It makes you **need global
memory less often**.

---

# PART 2: WHERE SHARED MEMORY LIVES

## 2.1 The Memory Hierarchy With Shared Memory Highlighted

```
                         ONE NVIDIA GPU
                         ──────────────

      ┌─────────────────────────────────────────────────────────┐
      │                         HBM / GLOBAL MEMORY              │
      │  Huge: GBs        Slow latency: hundreds of cycles       │
      │  Shared by: all SMs, all blocks, whole kernel            │
      └────────────────────────────┬────────────────────────────┘
                                   │
                                   │ load/store
                                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │                         ONE SM                           │
      │                                                         │
      │   ┌─────────────────────────────────────────────────┐   │
      │   │       SHARED MEMORY / L1 REGION                  │   │
      │   │       On-chip SRAM, block-scoped                 │   │
      │   │       Tiny compared to HBM, very fast            │   │
      │   └──────────────────────┬──────────────────────────┘   │
      │                          │                              │
      │   ┌──────────────────────▼──────────────────────────┐   │
      │   │                   REGISTERS                      │   │
      │   │       private variables for each thread          │   │
      │   └─────────────────────────────────────────────────┘   │
      │                                                         │
      │   CUDA cores, Tensor Cores, warp schedulers live here   │
      └─────────────────────────────────────────────────────────┘
```

Shared memory is:

- **On-chip:** physically inside the SM
- **Block-scoped:** visible to threads in the same block
- **Programmer-managed:** you decide what goes in it
- **Temporary:** lifetime is one block execution
- **Fast:** much lower latency than global memory when accessed without bank conflicts

Rough intuition:

| Memory | Latency feel | What it means |
|--------|--------------|---------------|
| Registers | fastest | private thread values, usually single-digit cycles |
| Shared memory | very fast | on-chip SRAM, often tens of cycles |
| Global memory / HBM | much slower | off-chip, hundreds of cycles |

People often summarize this as "shared memory is ~100x faster than global memory."
The exact ratio depends on GPU generation and access pattern, but the lesson is
correct: shared memory is close to the compute; HBM is far away.

The deeper idea is **data movement distance**. A number in a register is already
inside the thread's execution context. A number in shared memory is still on the
SM, close enough that many threads can coordinate around it. A number in HBM has
to travel across the GPU package through memory controllers and caches. High-end
GPUs are not slow because they cannot multiply; they are slow when the numbers
arrive too late. Shared memory is one of your tools for shortening that trip.

## 2.2 Shared Memory Scope And Lifetime

This is the most important rule:

> Threads in the same block can share shared memory. Threads in different blocks cannot.

```
Grid
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── Shared memory tile A   ← only Block 0 can see this
│
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   └── Shared memory tile A   ← separate copy, only Block 1 can see this
│
└── Block 2
    └── separate shared memory again
```

If you launch 100 blocks, each block gets its own private shared-memory allocation.
They do not communicate through shared memory.

This explains a design pattern you will see constantly: make each block own a
small independent tile of the problem. The block loads its tile, computes as much
as possible using that tile, then writes results back to global memory. CUDA likes
problems that can be chopped into many independent tiles because the scheduler can
spread those blocks across SMs without needing block-to-block coordination.

## 2.3 Why Shared Memory Is Not A Cache

L1/L2 cache is automatic:

```text
you read global memory → hardware may cache it
```

Shared memory is explicit:

```text
you declare shared memory
you load data into it
you synchronize threads
you read from it
```

This is more work, but more control.

```
Automatic cache:
  "GPU, please help me if you can."

Shared memory:
  "I know this tile will be reused. Put it here, then all threads reuse it."
```

That explicitness is both power and danger. If you choose the wrong tile, forget
a halo value, skip a synchronization, or create bank conflicts, shared memory can
make a kernel more complicated without making it faster. The professional habit is
to ask: **how many global memory reads did I remove, and what did shared memory
cost me in synchronization, bank conflicts, and occupancy?**

That control is why high-performance CUDA kernels use shared memory everywhere:

- tiled matrix multiplication
- convolution
- reductions
- prefix sums
- Flash Attention
- some transpose/layout-conversion kernels

---

# PART 3: THE SHARED MEMORY PROGRAMMING MODEL

## 3.1 Static Shared Memory

Static shared memory size is known at compile time:

```c
__global__ void kernel(...) {
    __shared__ float tile[256];
    ...
}
```

What this snippet is doing:

- `__global__` means this function is a CUDA kernel launched from the CPU and run on the GPU.
- `__shared__ float tile[256];` allocates 256 `float` values in shared memory for **each block**.
- Because one `float` is 4 bytes, this block uses `256 * 4 = 1024 bytes`, or 1 KB, of shared memory.
- Every thread inside the same block can read and write this `tile`.
- A different block gets a different private `tile`; blocks do not share this allocation.
- The actual per-block limit depends on the GPU. You can check it with `cudaGetDeviceProperties()` using `prop.sharedMemPerBlock`.

Use static shared memory when the size is fixed:

```c
__shared__ float tile[32][32];
```

What this snippet is doing:

- This creates a 2D shared-memory tile with 32 rows and 32 columns.
- It stores `32 * 32 = 1024` floats.
- At 4 bytes per float, this is `1024 * 4 = 4096 bytes`, or 4 KB, of shared memory per block.
- This shape is common because 32 connects naturally to CUDA's warp size.

This is simple and readable.

Static shared memory is great for teaching and for kernels with a natural fixed
tile shape. For example, a transpose kernel often uses a 32×32 tile because that
maps cleanly to a warp-sized mental model. The compiler can see the size, allocate
it predictably, and sometimes optimize around it.

## 3.2 Dynamic Shared Memory

Dynamic shared memory size is chosen at kernel launch time:

```c
__global__ void kernel(...) {
    extern __shared__ float tile[];
    ...
}
```

What this snippet is doing:

- `extern __shared__` declares dynamic shared memory.
- The array has no fixed size in the source code.
- CUDA allocates the size you provide when launching the kernel.
- Inside the kernel, `tile[]` behaves like a shared array owned by the current block.

Launch syntax:

```c
int shared_bytes = blockDim.x * sizeof(float);
kernel<<<blocks, threads, shared_bytes>>>(...);
```

What this snippet is doing:

- `shared_bytes` computes how many bytes of dynamic shared memory each block needs.
- `blockDim.x * sizeof(float)` means one `float` slot per thread.
- The third launch parameter, `shared_bytes`, is the dynamic shared-memory allocation per block.
- If `threads = 256`, then `shared_bytes = 256 * 4 = 1024 bytes`.

Use dynamic shared memory when tile size depends on runtime choices:

- block size selected by benchmark/autotuner
- variable sequence length
- variable tile shape
- one kernel supports many configurations

Dynamic shared memory is what you reach for when one kernel needs to support many
problem sizes. For example, an attention kernel may want different shared-memory
sizes for short vs long sequences, or an autotuner may try several tile sizes and
pick the fastest. The tradeoff is readability: `extern __shared__ float tile[]`
is just a raw buffer, so you must manually carve it into regions if you store
multiple arrays inside it.

Example layout:

```c
extern __shared__ float smem[];
float *tile_A = smem;
float *tile_B = smem + TILE_SIZE;
```

What this snippet is doing:

- `smem` is one raw shared-memory buffer.
- `tile_A` starts at the beginning of that buffer.
- `tile_B` starts after `TILE_SIZE` floats.
- This is how one dynamic shared-memory allocation can be manually split into multiple logical arrays.
- You must allocate enough bytes at launch for both regions, otherwise the arrays overlap or go out of bounds.

That is powerful, but it is also where indexing mistakes hide.

## 3.3 The Most Important Shared Memory Rule: Synchronize

Shared memory is shared between threads, so you must coordinate.

Bad:

```c
shared[threadIdx.x] = global[i];
float x = shared[threadIdx.x + 1];  // maybe neighbor has not written yet!
```

What this snippet is trying to do:

- Each thread copies one value from `global[i]` into `shared[threadIdx.x]`.
- Then each thread immediately tries to read its neighbor from shared memory.
- The problem is timing: the neighbor thread may not have written its value yet.
- So this code can read old, missing, or undefined data.

Good:

```c
shared[threadIdx.x] = global[i];
__syncthreads();
float x = shared[threadIdx.x + 1];
```

What this snippet is doing correctly:

- First, each thread writes its global value into the block-local shared tile.
- `__syncthreads()` waits until every thread in the block reaches the barrier.
- After the barrier, it is safe for a thread to read shared values written by other threads.
- This is the normal shared-memory rhythm: load together, synchronize, then reuse together.

`__syncthreads()` is a **block-wide barrier**:

```
Before barrier:
  fast threads arrive early and wait

At barrier:
  no thread continues until every thread in the block arrives

After barrier:
  all shared-memory writes before the barrier are visible to the block
```

Visual:

```
Thread 0: load tile ───────┐
Thread 1: load tile ───┐   │
Thread 2: load tile ───────┤
Thread 3: load tile ─┐     │
                     ▼     ▼
                __syncthreads()
                     │
                     ▼
             everyone can safely read tile
```

Do not put `__syncthreads()` inside a branch unless every thread in the block
reaches it. This can deadlock:

```c
if (threadIdx.x < 128) {
    __syncthreads();  // BAD if other threads skip it
}
```

What this snippet is warning you about:

- Only threads with `threadIdx.x < 128` enter the `if`.
- Threads with `threadIdx.x >= 128` skip the barrier.
- The first group waits forever for the second group to arrive.
- That can deadlock the whole block.

The barrier is not just "wait a bit." It is a correctness contract. If 255 threads
arrive and one thread never arrives, the 255 wait forever. This is why CUDA kernels
often load boundary/halo values with conditional logic, then place one unconditional
`__syncthreads()` after all loading is done. Branches before the barrier are fine;
skipping the barrier is not.

Also remember that barriers have a cost. If you synchronize after every tiny
operation, you may destroy performance. Good tiled kernels usually follow a rhythm:

```text
load tile -> synchronize -> compute a lot -> synchronize if tile will be reused
```

---

# PART 4: SHARED MEMORY BANKS

## 4.1 Why Banks Exist

Shared memory is fast because it is split into independent memory banks.

Think of shared memory like 32 checkout lanes:

```
                 SHARED MEMORY BANKS
                 ───────────────────

Bank:      0    1    2    3          30   31
           │    │    │    │          │    │
           ▼    ▼    ▼    ▼          ▼    ▼
         ┌───┬───┬───┬───┬───────┬───┬───┐
Words:   │ 0 │ 1 │ 2 │ 3 │  ...  │30 │31 │
         ├───┼───┼───┼───┼───────┼───┼───┤
         │32 │33 │34 │35 │  ...  │62 │63 │
         ├───┼───┼───┼───┼───────┼───┼───┤
         │64 │65 │66 │67 │  ...  │94 │95 │
         └───┴───┴───┴───┴───────┴───┴───┘
```

For FP32, the simple mental model is:

```text
bank = shared_memory_index % 32
```

So:

```text
shared[0]  → bank 0
shared[1]  → bank 1
shared[2]  → bank 2
...
shared[31] → bank 31
shared[32] → bank 0 again
```

If a warp's 32 threads access 32 different banks, all requests happen in parallel.
Fast.

If multiple threads access different addresses in the same bank, the bank must
serve them one after another. Slow.

That is a **bank conflict**.

The bank system exists for the same reason HBM has many wires: parallelism. Shared
memory is not one magical single-port box. It is many small lanes working together.
When a warp spreads its requests across banks, the hardware serves them together.
When a warp piles different addresses onto one bank, that bank becomes the narrow
door everyone is trying to walk through.

Bank conflicts are easy to miss because the code still gives the correct answer.
This is a performance bug, not a correctness bug. That makes it dangerous: your
kernel works, but it quietly leaves speed on the table.

## 4.2 Perfect Shared Memory Access

```c
int tx = threadIdx.x;     // 0..31 in one warp
float x = shared[tx];
```

What this snippet is doing:

- `tx` is the thread's lane-like index inside the block.
- Each thread reads the shared-memory element with the same index as its thread index.
- Thread 0 reads `shared[0]`, thread 1 reads `shared[1]`, and so on.
- For FP32 values, consecutive indices map to consecutive shared-memory banks.
- This is the clean, no-conflict pattern.

```
Thread:   T0  T1  T2  T3              T30 T31
Index:     0   1   2   3      ...      30  31
Bank:      0   1   2   3      ...      30  31

Result: 32 banks used → no conflict → fast
```

This is the shared-memory version of coalescing. In global memory, you wanted
neighboring threads to touch neighboring addresses so memory transactions were
efficient. In shared memory, you want neighboring threads to touch addresses that
land in different banks so the SRAM lanes work in parallel.

## 4.3 Worst Shared Memory Access

```c
int tx = threadIdx.x;     // 0..31
float x = shared[tx * 32];
```

What this snippet is doing:

- Each thread reads an index separated by 32 floats.
- Thread 0 reads `shared[0]`, thread 1 reads `shared[32]`, thread 2 reads `shared[64]`.
- With the simple FP32 rule, all those indices map back to the same bank.
- The addresses are different, so the hardware cannot use the broadcast exception.
- This creates a bank conflict.

```
Thread:   T0   T1   T2   T3          T31
Index:     0   32   64   96   ...    992
Bank:      0    0    0    0   ...      0

Result: 32 threads fight for bank 0 → 32-way bank conflict → slow
```

A 32-way bank conflict does not mean the value is wrong. It means what could have
happened in one parallel step becomes many serialized steps. You paid for 32 lanes
of shared-memory bandwidth but used only one lane effectively.

Important exception:

> If all threads in a warp read the exact same shared-memory address, the hardware
> can broadcast the value. That is not a harmful bank conflict.

Bad conflict:

```text
T0 reads shared[0]
T1 reads shared[32]
T2 reads shared[64]
...
different addresses, same bank → serialized
```

Broadcast:

```text
T0 reads shared[0]
T1 reads shared[0]
T2 reads shared[0]
...
same address → broadcast → fast
```

## 4.4 Why 2D Tiles Cause Bank Conflicts

The classic example is a 32×32 tile:

```c
__shared__ float tile[32][32];
```

What this snippet is doing:

- It declares a 32-by-32 shared-memory tile per block.
- The tile contains `32 * 32 = 1024` floats.
- In C/C++, this 2D array is stored row-major: row 0 first, then row 1, then row 2.
- That row-major layout is why row-wise and column-wise access behave differently.

C stores this in row-major order:

```text
tile[row][col] address index = row * 32 + col
```

What this formula means:

- `row * 32` jumps to the beginning of the selected row.
- `+ col` moves to the selected column inside that row.
- Because the row width is exactly 32, moving down one row moves by exactly 32 floats.

Row-wise access:

```c
tile[threadIdx.y][threadIdx.x]
```

What this snippet is doing:

- `threadIdx.y` chooses the row.
- `threadIdx.x` chooses the column.
- For one warp in this example, the row is fixed and the column changes from 0 to 31.
- So the warp reads consecutive elements in one row, which maps nicely across banks.

For one warp, `threadIdx.y` is fixed and `threadIdx.x = 0..31`.

```text
index = row * 32 + tx
bank  = (row * 32 + tx) % 32 = tx
```

What this formula means:

- `row * 32` is divisible by 32, so it disappears under `% 32`.
- The bank becomes `tx`.
- Since each thread has a different `tx`, each thread hits a different bank.

Every thread uses a different bank. Good.

Column-wise access:

```c
tile[threadIdx.x][threadIdx.y]
```

What this snippet is doing:

- Now `threadIdx.x` chooses the row.
- `threadIdx.y` chooses the column.
- For one warp, neighboring lanes move down different rows instead of across one row.
- Since each row is 32 floats wide, each lane jumps by 32 floats.

For one warp, `threadIdx.y` is fixed and `threadIdx.x = 0..31`.

```text
index = tx * 32 + col
bank  = (tx * 32 + col) % 32 = col
```

What this formula means:

- `tx * 32` is always divisible by 32.
- After `% 32`, only `col` remains.
- If `col` is fixed, every thread maps to the same bank.

Every thread uses the same bank. Bad.

Visual:

```
ROW ACCESS tile[y][x]

T0 T1 T2 T3 ... T31
 │  │  │  │       │
 ▼  ▼  ▼  ▼       ▼
B0 B1 B2 B3 ... B31   → parallel


COLUMN ACCESS tile[x][y]

T0 T1 T2 T3 ... T31
 │  │  │  │       │
 ▼  ▼  ▼  ▼       ▼
B7 B7 B7 B7 ... B7    → serialized
```

## 4.5 The Padding Trick: `[32][33]`

The fix is beautifully simple:

```c
__shared__ float tile[32][33];
```

What this snippet is doing:

- It declares 32 rows, but each row has 33 floats instead of 32.
- The extra column is padding.
- We do not add it because the math needs more values.
- We add it to change the shared-memory bank mapping.

Now the row stride is 33, not 32.

```text
tile[row][col] index = row * 33 + col
bank = (row * 33 + col) % 32
```

What this formula means:

- Moving down one row now moves by 33 floats.
- Since `33 % 32 = 1`, every new row starts one bank later than the previous row.
- This tiny shift breaks the "all lanes hit the same bank" pattern.

Column-wise access:

```text
index = tx * 33 + col
bank  = (tx * 33 + col) % 32
      = (tx + col) % 32
```

What this formula means:

- Under modulo 32, multiplying by 33 behaves like multiplying by 1.
- So the bank changes with `tx`.
- Consecutive threads now land on consecutive banks again.

Now thread 0, 1, 2, 3... land in different banks again.

```
WITHOUT PADDING [32][32]

T0  index 0*32+7   → bank 7
T1  index 1*32+7   → bank 7
T2  index 2*32+7   → bank 7
...
T31 index 31*32+7  → bank 7

32-way conflict.


WITH PADDING [32][33]

T0  index 0*33+7   → bank 7
T1  index 1*33+7   → bank 8
T2  index 2*33+7   → bank 9
...
T31 index 31*33+7  → bank 6

No conflict.
```

This tiny extra column is one of the most famous CUDA tricks.

Important clarification:

This solution is for **row-major shared-memory storage with column-wise access**.
CUDA/C/C++ 2D arrays are row-major, so `tile[32][32]` is stored row by row. Padding
is not needed simply because CUDA is row-major. Padding becomes useful when a warp
accesses that row-major tile in a column-wise or transposed pattern, such as
`tile[threadIdx.x][threadIdx.y]`.

Padding feels silly until you understand the modulo. You are not adding the extra
column because you need more data. You add it to change the address arithmetic.
By making the row stride 33, every new row starts one bank later than the previous
row. That tiny shift breaks the pattern that made all threads collide.

This is a general GPU lesson:

> Sometimes the fastest data structure is not the smallest one. It is the one whose
> layout matches the hardware.

---

# PART 5: CONSTANT MEMORY

## 5.1 What Constant Memory Is

Constant memory is a small read-only memory space, cached and optimized for a
very specific access pattern:

> All threads in a warp read the same address.

It is commonly described as **64 KB** of device constant memory.

Declare it like this:

```c
__constant__ float c_filter[64];
```

What this snippet is doing:

- `__constant__` places `c_filter` in GPU constant memory.
- `c_filter[64]` stores 64 read-only `float` values on the device.
- Kernels can read this array, but they cannot write to it.
- The CPU/host fills it before launching the kernel.

Copy data into it from the host:

```c
cudaMemcpyToSymbol(c_filter, h_filter, 64 * sizeof(float));
```

What this snippet is doing:

- `h_filter` is the CPU-side array.
- `c_filter` is the GPU constant-memory symbol.
- `cudaMemcpyToSymbol` copies bytes from the host array into the device constant-memory variable.
- `64 * sizeof(float)` says exactly how many bytes to copy.

Then kernels read it:

```c
float w = c_filter[k];
```

What this snippet is doing:

- A thread reads the `k`th constant value.
- If every thread in a warp reads the same `k`, constant memory can broadcast that value efficiently.
- If every thread reads a different `k`, the special constant-memory advantage becomes weaker.

Constant memory is controlled from the host side. Kernels cannot write to
`__constant__` variables. You update them before launching the kernel, and every
thread sees the same read-only data. That makes constant memory feel like a tiny
GPU-side configuration page.

## 5.2 Why Constant Memory Can Be Fast

If every thread in a warp reads the same constant address:

```c
float scale = c_params[0];
```

What this snippet is doing:

- Every thread reads the same scalar value from constant memory.
- If the whole warp reads `c_params[0]`, the hardware can serve that as a broadcast.
- This is the ideal constant-memory access pattern.

Then the hardware broadcasts it:

```
Warp reads c_params[0]

T0 T1 T2 T3 ... T31
 │  │  │  │       │
 └──┴──┴──┴───────┘
        one cached broadcast
```

This is perfect for:

- convolution filters
- small lookup tables
- scalar model/config parameters
- fixed coefficients
- small read-only metadata

The broadcast behavior is the main reason constant memory exists. If 32 threads
all need the same scalar, constant memory can behave almost like saying, "tell the
whole warp this value once." That is different from 32 independent global loads.
For small read-only parameters, it is elegant and simple.

## 5.3 When Constant Memory Is Bad

If each thread reads a different constant address:

```c
float x = c_table[threadIdx.x];
```

What this snippet is doing:

- Each thread uses its own thread index to choose a different constant-memory address.
- Thread 0 reads `c_table[0]`, thread 1 reads `c_table[1]`, and so on.
- That is no longer one shared value for the whole warp.
- Constant memory may serialize or lose its broadcast benefit here.

Then the warp no longer gets one broadcast. Accesses may serialize.

```
T0 reads c_table[0]
T1 reads c_table[1]
T2 reads c_table[2]
...
T31 reads c_table[31]

Not one broadcast. Less helpful.
```

Constant memory is not "global memory but magically faster." It is fast when the
warp reads the same location.

This is the pattern distinction:

```text
Good:
  all lanes read c_filter[k] for the same k

Less good:
  lane 0 reads c_table[0]
  lane 1 reads c_table[1]
  lane 2 reads c_table[2]
```

If your access pattern is divergent, normal global memory or texture/read-only
caches may be a better fit. The memory space must match the access pattern.

The rule:

```text
constant memory = tiny + read-only + broadcast-friendly
```

---

# PART 6: LOCAL MEMORY AND REGISTER SPILLS

## 6.1 The Name Is Misleading

Local memory sounds fast.

It is not.

In CUDA, **local memory** means memory private to one thread but stored in global
memory. It often appears when a thread has more private data than registers can hold.

This naming has fooled many beginners. "Local" means local to the thread's
address space, not local to the SM. A local-memory value belongs to one thread,
but physically it can live far away in global memory. So it has privacy like a
register, but latency more like global memory.

```
Thread private data:

Fast path:
  scalar variables → registers

Slow path:
  too many variables / large private arrays → local memory → global memory
```

## 6.2 How Spills Happen

Registers are the fastest storage on the GPU, but each SM has a limited register
file. If a kernel uses too many registers per thread, the compiler may spill some
values into local memory.

Example:

```c
__global__ void too_many_private_values(float *out) {
    float tmp[128];  // private array per thread
    ...
}
```

What this snippet is doing:

- Every thread creates its own private `tmp[128]` array.
- This is not shared between threads.
- Because the array is large for a per-thread private object, the compiler may place it in local memory instead of registers.
- If that happens, the private array may be backed by global memory, making it much slower than it looks.

This array may live in local memory, not registers.

That means every thread is secretly reading/writing global memory for its private
temporary values.

You can inspect this with:

```bash
nvcc -Xptxas -v my_kernel.cu -o my_kernel
```

What this command is doing:

- `nvcc` compiles the CUDA file.
- `-Xptxas -v` asks the lower-level PTX assembler to print resource usage.
- The output tells you how many registers the kernel used and whether any values spilled to local memory.

Look for output like:

```text
ptxas info    : Used 64 registers, 384 bytes spill stores, 384 bytes spill loads
```

What this output means:

- `Used 64 registers` means each thread uses 64 registers.
- `spill stores` means some register values had to be written out to local memory.
- `spill loads` means those spilled values had to be read back later.
- Spills are a warning sign that private thread state became too large for registers.

If you see spills, your kernel may be slower than expected.

Register pressure creates a three-way tradeoff:

1. More registers per thread can make each thread faster because it keeps values close.
2. Too many registers per thread can reduce occupancy because fewer warps fit on an SM.
3. If the compiler runs out of registers, spills create slow local-memory traffic.

This is why high-performance CUDA is not just "use more registers" or "use more
shared memory." Every on-chip resource is limited, and using too much of one can
reduce the amount of parallel work the SM can keep resident.

## 6.3 Local Memory Mental Model

```
Register:
  private to thread
  on-chip
  fastest

Local memory:
  private to thread
  physically in global memory
  slow
  often caused by register pressure or private arrays

Shared memory:
  shared by block
  on-chip
  fast if no bank conflicts
```

---

# PART 7: HANDS-ON — PROVING THE THEORY

Theory is nice. Timers are better.

## 7.1 Exercise 1: Shared Memory 1D Stencil

This example computes:

```c
out[i] = in[i-1] + in[i] + in[i+1]
```

What this snippet is doing:

- It computes one output element from a center value and its two neighbors.
- This is the exact operation the exercise implements twice: once with global
  memory only, and once with a shared-memory tile.

We compare two versions:

1. **Naive:** each thread reads neighbors directly from global memory.
2. **Shared:** each block loads a tile plus halo values into shared memory.

### Why Halo Values Exist

If a block owns these elements:

```
Block owns:      in[100] in[101] ... in[355]
```

The first thread also needs `in[99]`.
The last thread also needs `in[356]`.

Those are halo values:

```
          left halo                  right halo
             ▼                          ▼
... in[99] in[100] in[101] ... in[355] in[356] ...
          └──────── block tile ────────┘
```

Shared memory tile size:

```text
blockDim.x + 2
```

One extra value on the left, one extra value on the right.

This "tile plus halo" pattern shows up everywhere. In convolution, image filters
need neighboring pixels. In finite-difference physics simulations, each cell needs
neighboring cells. In attention and GEMM, the "halo" idea becomes tile boundaries
and edge cases. The core habit is the same: when a block owns a tile, ask what
extra neighboring data the computation needs to be correct.

```c
%%writefile shared_stencil.cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        printf("CUDA error at %s:%d: %s\n",                       \
               __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__global__ void stencil_global(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N - 1) {
        out[i] = in[i - 1] + in[i] + in[i + 1];
    }
}

__global__ void stencil_shared(const float *in, float *out, int N) {
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int i  = blockIdx.x * blockDim.x + tx;

    // Center value. Shift by +1 because tile[0] is reserved for left halo.
    if (i < N) {
        tile[tx + 1] = in[i];
    }

    // First thread loads left halo.
    if (tx == 0) {
        int left = i - 1;
        tile[0] = (left >= 0) ? in[left] : 0.0f;
    }

    // Last thread loads right halo.
    if (tx == blockDim.x - 1) {
        int right = i + 1;
        tile[tx + 2] = (right < N) ? in[right] : 0.0f;
    }

    __syncthreads();

    if (i > 0 && i < N - 1) {
        out[i] = tile[tx] + tile[tx + 1] + tile[tx + 2];
    }
}

float benchmark_global(const float *d_in, float *d_out, int N, int blocks, int threads) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    stencil_global<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < 200; r++) {
        stencil_global<<<blocks, threads>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 200.0f;
}

float benchmark_shared(const float *d_in, float *d_out, int N, int blocks, int threads) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int shared_bytes = (threads + 2) * sizeof(float);

    stencil_shared<<<blocks, threads, shared_bytes>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < 200; r++) {
        stencil_shared<<<blocks, threads, shared_bytes>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 200.0f;
}

int main() {
    int N = 32 * 1024 * 1024;  // 32M floats = 128 MB
    size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float ms_global = benchmark_global(d_in, d_out, N, blocks, threads);
    float ms_shared = benchmark_shared(d_in, d_out, N, blocks, threads);

    // Useful bytes for stencil:
    // naive reads 3 floats + writes 1 float per output = 16 bytes useful per element.
    // shared version reduces global reads through reuse, but report same useful work for fair comparison.
    float useful_gb = (float)N * 4.0f * sizeof(float) / 1e9f;
    float bw_global = useful_gb / (ms_global / 1000.0f);
    float bw_shared = useful_gb / (ms_shared / 1000.0f);

    printf("╔════════════════════════════════════════════════╗\n");
    printf("║        Shared Memory 1D Stencil Demo           ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ Version        │ Time (ms) │ Useful GB/s       ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ Global only    │ %8.3f │ %8.1f          ║\n", ms_global, bw_global);
    printf("║ Shared tile    │ %8.3f │ %8.1f          ║\n", ms_shared, bw_shared);
    printf("╚════════════════════════════════════════════════╝\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

### Code Walkthrough: Shared Stencil

Read this program in layers.

```text
%%writefile shared_stencil.cu
```

This is a Colab/Jupyter command. It writes the following code into a file named
`shared_stencil.cu` so `nvcc` can compile it.

```text
#include <stdio.h>
#include <stdlib.h>
```

These headers provide `printf()` and `exit()`.

```text
#define CUDA_CHECK(call) ...
```

This macro wraps CUDA API calls. If a CUDA call fails, it prints the file name,
line number, and CUDA error string, then exits. This keeps errors visible instead
of letting the program silently continue.

```text
__global__ void stencil_global(...)
```

This is the naive kernel. Each GPU thread computes one output element.

```text
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This converts block/thread coordinates into the global array index. For example,
if `blockDim.x = 256`, block 0 covers indices 0-255, block 1 covers 256-511.

```text
if (i > 0 && i < N - 1)
```

This avoids the boundary elements because `out[0]` would need `in[-1]`, and
`out[N-1]` would need `in[N]`.

```text
out[i] = in[i - 1] + in[i] + in[i + 1];
```

This is the stencil. The thread reads the left neighbor, center value, and right
neighbor directly from global memory.

```text
__global__ void stencil_shared(...)
```

This is the shared-memory version. It still computes one output per thread, but
the block first stages input values into a shared tile.

```text
extern __shared__ float tile[];
```

This creates a dynamic shared-memory array. Its size is decided at launch by the
third kernel launch parameter.

```text
int tx = threadIdx.x;
int i  = blockIdx.x * blockDim.x + tx;
```

`tx` is the thread's local index inside the block. `i` is the matching global
index in the full input array.

```text
tile[tx + 1] = in[i];
```

Each thread copies its own global value into shared memory. The `+1` shift leaves
`tile[0]` open for the left halo.

```text
if (tx == 0) { ... tile[0] = ...; }
```

Only the first thread in the block loads the left halo value. This is the value
just before the block's first normal element.

```text
if (tx == blockDim.x - 1) { ... tile[tx + 2] = ...; }
```

Only the last thread in the block loads the right halo value. This is the value
just after the block's last normal element.

```text
__syncthreads();
```

This is essential. It guarantees the center values and halo values are all loaded
before any thread reads from `tile`.

```text
out[i] = tile[tx] + tile[tx + 1] + tile[tx + 2];
```

Now the thread computes the stencil from shared memory. `tile[tx]` is left,
`tile[tx + 1]` is center, and `tile[tx + 2]` is right.

```text
benchmark_global(...)
benchmark_shared(...)
```

These functions time the two kernels. They do one warm-up launch first, then time
many repeated launches using CUDA events.

```text
int shared_bytes = (threads + 2) * sizeof(float);
```

The shared kernel needs one slot per thread plus two halo slots. With 256 threads,
that is 258 floats.

```text
stencil_shared<<<blocks, threads, shared_bytes>>>(...)
```

The third launch argument gives each block that many bytes of dynamic shared
memory.

```text
float useful_gb = (float)N * 4.0f * sizeof(float) / 1e9f;
```

For each output element, the stencil conceptually uses 3 reads and 1 write, so
the program reports useful work as 4 floats per element.

```text
printf(...)
```

The print block shows timing and useful bandwidth so you can compare global-only
access versus shared-memory reuse.

Run:

```python
!nvcc shared_stencil.cu -o shared_stencil && ./shared_stencil
```

What this command is doing:

- `nvcc shared_stencil.cu -o shared_stencil` compiles the CUDA program.
- `./shared_stencil` runs the compiled executable.
- In Colab, the `!` means "run this as a shell command."

### How To Read The Result

Do not expect shared memory to always win dramatically here.

Why?

Modern GPUs already have L1/L2 caches, and this simple stencil is very cache-friendly.
The lesson is not "shared memory always wins."

The lesson is:

> Shared memory helps when you can reuse data predictably inside a block and avoid
> repeated global memory traffic.

This becomes much more important in tiled matrix multiplication, convolution, and
Flash Attention, where reuse is massive.

A useful rule of thumb:

```text
If each loaded value is reused 1 time, shared memory may not help.
If each loaded value is reused 10, 100, or 1000 times, shared memory becomes powerful.
```

Shared memory is a reuse amplifier.

## 7.2 Exercise 2: Bank Conflict vs Padding

This one is the real proof.

We compare:

```c
__shared__ float tile[32][32];  // conflict when read column-wise
```

What this snippet is doing:

- It creates a 32-by-32 shared tile with row stride 32.
- Column-wise reads from this layout can make every lane in a warp hit the same bank.

versus:

```c
__shared__ float tile[32][33];  // padded, no conflict
```

What this snippet is doing:

- It creates the same logical 32-row tile, but with one extra padding column.
- The row stride becomes 33, which shifts each row by one bank.
- That breaks the column-wise bank conflict.

```c
%%writefile bank_conflict_demo.cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        printf("CUDA error at %s:%d: %s\n",                       \
               __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define TILE 32

__global__ void conflict_kernel(float *out, int iters) {
    __shared__ volatile float tile[TILE][TILE];  // row stride = 32 → conflicts on column read

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    tile[ty][tx] = (float)(tid);
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iters; i++) {
        // Column-wise read. For a warp with ty fixed and tx = 0..31:
        // tile[tx][ty] maps all threads to the same bank.
        sum += tile[tx][ty];
    }

    out[tid] = sum;
}

__global__ void padded_kernel(float *out, int iters) {
    __shared__ volatile float tile[TILE][TILE + 1];  // row stride = 33 → padding avoids conflicts

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    tile[ty][tx] = (float)(tid);
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iters; i++) {
        sum += tile[tx][ty];
    }

    out[tid] = sum;
}

float time_conflict(float *d_out, int iters, int repeats) {
    dim3 block(TILE, TILE);  // 1024 threads, max legal block size

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    conflict_kernel<<<1, block>>>(d_out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < repeats; r++) {
        conflict_kernel<<<1, block>>>(d_out, iters);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}

float time_padded(float *d_out, int iters, int repeats) {
    dim3 block(TILE, TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    padded_kernel<<<1, block>>>(d_out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < repeats; r++) {
        padded_kernel<<<1, block>>>(d_out, iters);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}

int main() {
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, TILE * TILE * sizeof(float)));

    int iters = 2000;
    int repeats = 200;

    float ms_conflict = time_conflict(d_out, iters, repeats);
    float ms_padded   = time_padded(d_out, iters, repeats);

    printf("╔════════════════════════════════════════════════╗\n");
    printf("║          Shared Memory Bank Conflict Demo      ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ Kernel          │ Time (ms) │ Relative         ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ [32][32]        │ %8.4f │ %6.2fx          ║\n", ms_conflict, ms_conflict / ms_padded);
    printf("║ [32][33] padded │ %8.4f │ %6.2fx          ║\n", ms_padded, 1.0f);
    printf("╚════════════════════════════════════════════════╝\n");

    printf("\nLesson: one padding column can turn serialized bank access into parallel bank access.\n");

    cudaFree(d_out);
    return 0;
}
```

### Code Walkthrough: Bank Conflict Demo

```text
%%writefile bank_conflict_demo.cu
```

This writes the program into `bank_conflict_demo.cu` in a notebook environment.

```text
#define TILE 32
```

The tile width is 32 so it matches the warp size and makes the bank-conflict
pattern easy to see.

```text
__global__ void conflict_kernel(float *out, int iters)
```

This kernel intentionally creates a shared-memory bank conflict.

```text
__shared__ volatile float tile[TILE][TILE];
```

This allocates a 32-by-32 shared tile. The row stride is 32. `volatile` discourages
the compiler from optimizing away the repeated shared-memory reads, because this
demo wants to measure the access pattern.

```text
int tx = threadIdx.x;
int ty = threadIdx.y;
int tid = ty * blockDim.x + tx;
```

`tx` and `ty` are the thread's 2D coordinates inside the block. `tid` flattens
that 2D coordinate into a single output index.

```text
tile[ty][tx] = (float)(tid);
```

Each thread writes one value into the shared tile. This write is row-wise, so it
is bank-friendly.

```text
__syncthreads();
```

The block waits until the full tile has been written.

```text
sum += tile[tx][ty];
```

This reads the tile with row and column swapped. For a 32-by-32 tile, this
column-wise pattern maps many lanes to the same bank. That is the intentional
conflict.

```text
out[tid] = sum;
```

The final sum is written to global memory so the compiler has to keep the loop's
work.

```text
__global__ void padded_kernel(float *out, int iters)
```

This kernel does the same logical work, but with a padded shared-memory layout.

```text
__shared__ volatile float tile[TILE][TILE + 1];
```

The tile has 33 columns. The extra column changes the row stride from 32 to 33,
which spreads column-wise reads across banks.

```text
sum += tile[tx][ty];
```

This line is intentionally the same as in the conflict kernel. The only difference
is the memory layout, which proves the padding is what changes performance.

```text
time_conflict(...)
time_padded(...)
```

These timing functions launch each kernel once as a warm-up, then repeatedly launch
it under CUDA events.

```text
dim3 block(TILE, TILE);
```

This creates a 32-by-32 thread block, which is 1024 threads. That is the maximum
common CUDA block size.

```text
int iters = 2000;
int repeats = 200;
```

`iters` repeats the shared-memory read inside each kernel to amplify the bank
conflict. `repeats` launches the whole kernel many times to get a stable average.

```text
ms_conflict / ms_padded
```

This prints how many times slower the conflicting layout is compared with the
padded layout.

Run:

```python
!nvcc bank_conflict_demo.cu -o bank_conflict_demo && ./bank_conflict_demo
```

What this command is doing:

- `nvcc` compiles the CUDA source file.
- `-o bank_conflict_demo` names the executable.
- `./bank_conflict_demo` runs it and prints the timing comparison.

### What You Should See

Exact numbers depend on GPU architecture and compiler behavior, but the padded
version should usually be faster.

If the speedup is smaller than expected, that is not failure. Modern GPUs have
improved shared-memory behavior, and compilers are clever. The concept still
matters because bank conflicts appear in real tiled kernels.

This demo intentionally repeats the same access many times so the timing signal is
large enough to notice. Real kernels may have bank conflicts mixed with arithmetic,
global-memory traffic, and synchronization, so the conflict may not dominate the
whole runtime. Nsight Compute is how you eventually confirm it.

## 7.3 Exercise 3: Constant Memory Broadcast

This demo compares two constant-memory access patterns:

1. All threads read the same constant address → broadcast-friendly.
2. Threads read different constant addresses → less broadcast-friendly.

```c
%%writefile constant_memory_demo.cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        printf("CUDA error at %s:%d: %s\n",                       \
               __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__constant__ float c_table[256];

__global__ void constant_broadcast(float *out, int N, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < iters; k++) {
        sum += c_table[k & 255];  // all threads read same address each iteration
    }
    out[i] = sum;
}

__global__ void constant_divergent(float *out, int N, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int lane = threadIdx.x & 31;
    float sum = 0.0f;
    for (int k = 0; k < iters; k++) {
        sum += c_table[(lane + k) & 255];  // different lanes read different addresses
    }
    out[i] = sum;
}

float time_broadcast(float *d_out, int N, int iters) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constant_broadcast<<<blocks, threads>>>(d_out, N, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < 50; r++) {
        constant_broadcast<<<blocks, threads>>>(d_out, N, iters);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 50.0f;
}

float time_divergent(float *d_out, int N, int iters) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constant_divergent<<<blocks, threads>>>(d_out, N, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < 50; r++) {
        constant_divergent<<<blocks, threads>>>(d_out, N, iters);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 50.0f;
}

int main() {
    float h_table[256];
    for (int i = 0; i < 256; i++) h_table[i] = 1.0f + i;

    CUDA_CHECK(cudaMemcpyToSymbol(c_table, h_table, sizeof(h_table)));

    int N = 8 * 1024 * 1024;
    int iters = 100;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    float ms_broadcast = time_broadcast(d_out, N, iters);
    float ms_divergent = time_divergent(d_out, N, iters);

    printf("╔════════════════════════════════════════════════╗\n");
    printf("║             Constant Memory Demo               ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ Pattern         │ Time (ms) │ Meaning          ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ Broadcast       │ %8.3f │ same addr/warp   ║\n", ms_broadcast);
    printf("║ Divergent       │ %8.3f │ many addr/warp   ║\n", ms_divergent);
    printf("╚════════════════════════════════════════════════╝\n");

    cudaFree(d_out);
    return 0;
}
```

### Code Walkthrough: Constant Memory Demo

```text
%%writefile constant_memory_demo.cu
```

This writes the full demo program into a CUDA source file.

```text
__constant__ float c_table[256];
```

This creates a 256-float table in device constant memory. Kernels can read it.
The host fills it before the kernels run.

```text
__global__ void constant_broadcast(float *out, int N, int iters)
```

This kernel tests the good constant-memory pattern: warp-uniform reads.

```text
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= N) return;
```

Each thread gets a global output index. Threads outside the valid range exit.

```text
sum += c_table[k & 255];
```

For a fixed loop value `k`, every thread in the warp reads the same constant
address. `k & 255` keeps the index between 0 and 255. This is broadcast-friendly.

```text
out[i] = sum;
```

Each thread writes its final sum so the work is observable.

```text
__global__ void constant_divergent(float *out, int N, int iters)
```

This kernel tests the weaker pattern: different lanes read different constant
addresses.

```text
int lane = threadIdx.x & 31;
```

This computes the thread's lane index inside a warp, from 0 to 31.

```text
sum += c_table[(lane + k) & 255];
```

Now each lane reads a different table entry for the same loop iteration. That
breaks the clean broadcast pattern.

```text
time_broadcast(...)
time_divergent(...)
```

These functions time each kernel using CUDA events. Each one launches a warm-up
kernel before measuring repeated launches.

```text
float h_table[256];
for (int i = 0; i < 256; i++) h_table[i] = 1.0f + i;
```

This creates and initializes the CPU-side version of the constant table.

```text
cudaMemcpyToSymbol(c_table, h_table, sizeof(h_table));
```

This copies the CPU table into the GPU's constant-memory symbol `c_table`.

```text
int N = 8 * 1024 * 1024;
int iters = 100;
```

The program uses many output elements and repeated table reads so the timing
difference is visible.

```text
printf(...)
```

The print section compares the broadcast-friendly access pattern with the divergent
access pattern.

Run:

```python
!nvcc constant_memory_demo.cu -o constant_memory_demo && ./constant_memory_demo
```

What this command is doing:

- It compiles the constant-memory demo with `nvcc`.
- Then it runs the executable and prints the timing table.

### What This Proves

Constant memory is best when a warp reads the same address.

This is why small read-only coefficients are good candidates:

```text
same filter weights used by many threads
same scale factor used by many threads
same model metadata used by many threads
```

But if every lane reads a different address, constant memory loses its special
broadcast advantage.

Think of constant memory as a megaphone, not a library. It is excellent when one
small fact needs to be announced to the whole warp. It is not ideal when every
thread wants to browse a different shelf.

---

# PART 8: MINI-PROJECT — TILED MATRIX TRANSPOSE

## "The Bank Conflict Detective"

Matrix transpose is the perfect shared-memory mini-project because it has both:

- global memory coalescing problem
- shared memory bank conflict problem

You will write three transpose kernels:

1. **Naive transpose:** reads coalesced, writes uncoalesced.
2. **Shared transpose:** uses shared tile to make global reads/writes coalesced.
3. **Padded shared transpose:** uses `[32][33]` to avoid bank conflicts.

Why transpose matters:

- layout conversion in ML frameworks
- attention tensor rearrangement
- image processing
- memory-format optimization
- preparing for tiled GEMM

## The Mental Picture

Input matrix:

```
A rows are contiguous in memory:

row 0: A[0][0] A[0][1] A[0][2] ...
row 1: A[1][0] A[1][1] A[1][2] ...
```

Naive transpose:

```c
B[col][row] = A[row][col];
```

What this snippet is doing:

- It reads one value from row `row`, column `col` in matrix `A`.
- It writes that value to row `col`, column `row` in matrix `B`.
- That swap of row and column is the definition of transpose.
- The difficulty is not the math; the difficulty is making the GPU reads and
  writes memory-friendly.

Reads are row-wise, good.
Writes become column-wise, bad.

Shared-memory transpose:

```text
1. Load A tile row-wise into shared memory      → coalesced global reads
2. Read shared tile transposed                  → fast on-chip, but watch banks
3. Write B tile row-wise                        → coalesced global writes
```

The trick is to turn an uncoalesced global write into a shared-memory rearrangement.

```c
%%writefile transpose_project.cu
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        printf("CUDA error at %s:%d: %s\n",                       \
               __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define TILE 32

__global__ void transpose_naive(const float *in, float *out, int width, int height) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

__global__ void transpose_shared_conflict(const float *in, float *out, int width, int height) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // Transposed block coordinates
    int out_x = blockIdx.y * TILE + threadIdx.x;
    int out_y = blockIdx.x * TILE + threadIdx.y;

    if (out_x < height && out_y < width) {
        out[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_shared_padded(const float *in, float *out, int width, int height) {
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    int out_x = blockIdx.y * TILE + threadIdx.x;
    int out_y = blockIdx.x * TILE + threadIdx.y;

    if (out_x < height && out_y < width) {
        out[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

float time_naive(const float *d_in, float *d_out, int width, int height) {
    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < 100; r++) {
        transpose_naive<<<grid, block>>>(d_in, d_out, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

float time_shared_conflict(const float *d_in, float *d_out, int width, int height) {
    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    transpose_shared_conflict<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < 100; r++) {
        transpose_shared_conflict<<<grid, block>>>(d_in, d_out, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

float time_shared_padded(const float *d_in, float *d_out, int width, int height) {
    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    transpose_shared_padded<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int r = 0; r < 100; r++) {
        transpose_shared_padded<<<grid, block>>>(d_in, d_out, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

int main() {
    int width = 4096;
    int height = 4096;
    size_t bytes = (size_t)width * height * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    float ms_naive = time_naive(d_in, d_out, width, height);
    float ms_conf  = time_shared_conflict(d_in, d_out, width, height);
    float ms_pad   = time_shared_padded(d_in, d_out, width, height);

    // Transpose reads matrix once and writes matrix once.
    float moved_gb = 2.0f * bytes / 1e9f;

    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║              Tiled Transpose Project               ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║ Kernel              │ Time (ms) │ Effective GB/s   ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║ Naive               │ %8.3f │ %8.1f         ║\n", ms_naive, moved_gb / (ms_naive / 1000.0f));
    printf("║ Shared [32][32]     │ %8.3f │ %8.1f         ║\n", ms_conf,  moved_gb / (ms_conf  / 1000.0f));
    printf("║ Shared [32][33]     │ %8.3f │ %8.1f         ║\n", ms_pad,   moved_gb / (ms_pad   / 1000.0f));
    printf("╚════════════════════════════════════════════════════╝\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

### Code Walkthrough: Tiled Transpose

```text
%%writefile transpose_project.cu
```

This writes the project code into `transpose_project.cu`.

```text
#define TILE 32
```

The matrix is processed in 32-by-32 tiles. This shape makes the connection to
warps and bank conflicts easy to see.

```text
__global__ void transpose_naive(...)
```

This is the direct transpose kernel. It does not use shared memory.

```text
int x = blockIdx.x * TILE + threadIdx.x;
int y = blockIdx.y * TILE + threadIdx.y;
```

These lines compute the input matrix coordinate handled by the current thread.
`x` is the column and `y` is the row.

```text
out[x * height + y] = in[y * width + x];
```

This writes the transposed value. The input index is row-major for the original
matrix. The output index swaps row and column. The problem is that either reads or
writes become poorly coalesced.

```text
__global__ void transpose_shared_conflict(...)
```

This version uses shared memory to make global memory access nicer, but it still
has shared-memory bank conflicts.

```text
__shared__ float tile[TILE][TILE];
```

This allocates a 32-by-32 shared tile. The row stride is exactly 32, so transposed
shared-memory reads can conflict.

```text
tile[threadIdx.y][threadIdx.x] = in[y * width + x];
```

Each thread loads one value from global memory into shared memory. This load is
row-wise and coalesced when neighboring threads read neighboring `x` values.

```text
__syncthreads();
```

The block waits until the full tile is loaded before reading it transposed.

```text
int out_x = blockIdx.y * TILE + threadIdx.x;
int out_y = blockIdx.x * TILE + threadIdx.y;
```

The output block coordinates are swapped. This is how the tile moves to its
transposed position in the output matrix.

```text
out[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
```

The thread reads the shared tile transposed and writes to global memory. The
global write is now coalesced, but the shared-memory read can have bank conflicts
because the tile has a 32-float row stride.

```text
__global__ void transpose_shared_padded(...)
```

This is the improved shared-memory transpose.

```text
__shared__ float tile[TILE][TILE + 1];
```

The tile has one extra column. The logical tile is still 32-by-32, but the memory
stride is 33. This avoids the column-wise shared-memory bank conflict.

```text
tile[threadIdx.y][threadIdx.x] = in[y * width + x];
out[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
```

These are the same logical load and transposed read as before. The key difference
is the padded layout, not the transpose math.

```text
time_naive(...)
time_shared_conflict(...)
time_shared_padded(...)
```

These functions measure the three versions with CUDA events. Each function warms
up once, then averages 100 launches.

```text
dim3 block(TILE, TILE);
dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
```

`block` creates one 32-by-32 thread block per tile. `grid` computes how many tiles
are needed to cover the full matrix, including edge cases where dimensions are not
perfect multiples of 32.

```text
int width = 4096;
int height = 4096;
```

The test matrix is large enough that memory access patterns matter.

```text
float moved_gb = 2.0f * bytes / 1e9f;
```

A transpose reads the matrix once and writes the matrix once, so total data moved
is approximately `2 * bytes`.

```text
moved_gb / (ms / 1000.0f)
```

This converts the measured time into effective GB/s.

Run:

```python
!nvcc transpose_project.cu -o transpose_project && ./transpose_project
```

What this command is doing:

- It compiles the transpose project.
- Then it runs the executable and prints timing plus effective bandwidth for all
  three kernels.

## What To Look For

You should usually see:

```text
Naive transpose          slowest
Shared [32][32]          faster, but bank conflicts remain
Shared [32][33] padded   fastest or near-fastest
```

This project connects everything:

- global coalescing from Day 1
- shared memory tiling from today
- bank conflicts from today
- padding trick `[32][33]`
- tiled thinking needed for GEMM

This is not a toy. Matrix transpose is a real layout operation in high-performance
ML systems.

The important conceptual move is this:

```text
Use shared memory as a safe place to perform an awkward rearrangement.
```

Global memory wants contiguous reads and contiguous writes. A transpose naturally
breaks one of those. Shared memory lets you read global memory in the order it
likes, rearrange the data on-chip, then write global memory in the order it likes.
That same trick appears in many optimized kernels: make global memory happy, do
the messy layout work close to the SM.

---

# PART 9: HOW THIS CONNECTS TO LLMs

## 9.1 Matrix Multiplication

Every Transformer layer is full of matrix multiplication:

```text
Q = X Wq
K = X Wk
V = X Wv
FFN = down(silu(up(X)) * gate(X))
```

Fast GEMM does not read one element from global memory, use it once, and throw it
away. That would waste bandwidth.

Fast GEMM:

1. Loads tiles of A and B into shared memory.
2. Reuses those tiles for many multiply-add operations.
3. Accumulates results in registers.
4. Writes final tile to global memory.

```
Global A tile ─┐
               ├──> shared memory tiles ──> registers ──> C tile
Global B tile ─┘
```

The whole game is reuse.

In a naive GEMM, an element of A or B may be fetched from global memory again and
again by different threads. In a tiled GEMM, a block pays the global-memory cost
once for a tile, then reuses the tile for many multiply-adds. That raises
**arithmetic intensity**: more math per byte loaded. You will see this phrase
constantly in GPU performance work.

## 9.2 Flash Attention

Attention wants to compute:

```text
softmax(QK^T)V
```

Naively, this creates a huge attention matrix in global memory.

Flash Attention avoids that by:

1. Loading Q/K/V blocks into on-chip memory.
2. Computing partial attention.
3. Keeping running softmax statistics.
4. Avoiding the giant HBM read/write of the full attention matrix.

That is the same idea you learned today:

> Move small tiles close to the compute, reuse them, and avoid unnecessary HBM traffic.

Flash Attention is not magic because it changed the math of attention. It is magic
because it changed where intermediate data lives. The attention matrix is too large
and too temporary to deserve a round trip through HBM. Shared memory/registers let
the kernel compute what it needs and throw away what it does not.

## 9.3 Constant Memory In LLM Kernels

Constant memory is useful for small read-only values:

- scale factors
- small lookup tables
- fixed kernel coefficients
- metadata that many threads read uniformly

It is not where model weights live. LLM weights are far too large.

So when you hear "constant memory" in an LLM context, think **small constants**,
not parameters. Model weights live in global memory/HBM, maybe quantized and
cached/tiled by kernels. Constant memory is for tiny read-only values that many
threads use uniformly.

## 9.4 Local Memory In LLM Kernels

Local memory shows up when kernels get too ambitious:

- too many temporary variables
- large private arrays
- high register pressure
- compiler spills

In fused LLM kernels, register pressure is a serious design constraint. Sometimes
a kernel that does "more fusion" gets slower because it spills too much.

That is a professional-level CUDA lesson:

> More work per kernel is good only until registers and shared memory become the bottleneck.

This is why kernel fusion is an art. Fusing two operations can save HBM traffic,
but it also increases the number of live temporary values. If those temporaries
spill to local memory, the fused kernel can lose the benefit it was supposed to
gain. The best CUDA engineers think in budgets: registers, shared memory, occupancy,
memory traffic, and arithmetic intensity.

---

# PART 10: WHAT YOU NOW UNDERSTAND

Close this document and answer these without looking:

1. **Why does shared memory help?**
2. **What is the scope and lifetime of shared memory?**
3. **Why do we need `__syncthreads()` after loading a shared tile?**
4. **What are shared memory banks?**
5. **What causes a bank conflict?**
6. **Why does `[32][33]` fix column-wise bank conflicts?**
7. **When is constant memory fast?**
8. **Why is local memory slow despite its name?**
9. **How do register spills show up in compiler output?**
10. **How does shared memory prepare you for tiled GEMM and Flash Attention?**

If you can answer these, you now understand one of the biggest differences between
"CUDA code that runs" and "CUDA code that is designed."

---

# CHECKLIST

- [ ] Understand shared memory as on-chip, block-scoped SRAM
- [ ] Can explain why shared memory reduces global memory traffic through reuse
- [ ] Know static `__shared__` vs dynamic `extern __shared__`
- [ ] Know why `__syncthreads()` is required after cooperative tile loading
- [ ] Understand 32 shared-memory banks and the FP32 bank mapping mental model
- [ ] Can explain bank conflicts and the broadcast exception
- [ ] Can explain why `[32][33]` padding fixes column-wise access
- [ ] Understand constant memory: 64 KB, read-only, warp broadcast-friendly
- [ ] Understand local memory: private per-thread data stored in global memory when spilled
- [ ] Ran the shared-memory stencil demo
- [ ] Ran the bank-conflict padding demo
- [ ] Ran the constant-memory broadcast demo
- [ ] Built the tiled transpose mini-project
- [ ] Can connect shared memory to tiled GEMM and Flash Attention

---

# DETAILED ANSWERS

## 1. Why does shared memory help?

Shared memory helps because it lets threads in the same block reuse data from
on-chip SRAM instead of repeatedly fetching the same values from global memory.

Global memory/HBM is large but far away. Shared memory is tiny but close to the
SM's execution units. The point is not that shared memory makes one global load
faster. The point is that shared memory reduces how many global loads you need.

Example:

```text
Naive:
  thread 0 reads A[k] from global
  thread 1 reads A[k] from global
  thread 2 reads A[k] from global

Shared:
  one group of threads loads A[k] into shared memory
  many threads reuse A[k] from shared memory
```

This is why shared memory is central to tiled matrix multiplication. A tile of A
and a tile of B are loaded once, then reused for many multiply-add operations.
That increases arithmetic intensity:

```text
more math per byte loaded from HBM
```

## 2. What is the scope and lifetime of shared memory?

Shared memory is **block-scoped** and **temporary**.

Block-scoped means:

```text
threads inside the same block can read/write the same shared-memory allocation
threads in different blocks cannot see each other's shared memory
```

Temporary means:

```text
shared memory exists only while that block is running
when the block finishes, its shared memory disappears
```

If a grid has 100 blocks, each block gets its own private shared-memory region.
There is no global shared-memory pool where all blocks communicate. For block-to-block
communication, you usually write to global memory and launch another kernel, or use
more advanced cooperative-group patterns later.

This is why CUDA algorithms are designed as independent tiles:

```text
one block owns one tile
the block uses shared memory for that tile
the block writes final results to global memory
```

## 3. Why do we need `__syncthreads()` after loading a shared tile?

Because threads run independently, even inside the same block. Some threads may
finish loading their part of the tile earlier than others.

Without synchronization:

```c
tile[threadIdx.x] = global[i];
float neighbor = tile[threadIdx.x + 1];  // neighbor may not be loaded yet
```

Code meaning:

- The first line copies one global-memory value into shared memory.
- The second line immediately reads a neighboring shared-memory slot.
- It is unsafe because that neighboring slot may be written by another thread that
  has not reached its store yet.

That read can happen before the neighboring thread writes its value. The result is
a race condition.

`__syncthreads()` is a block-wide barrier:

```text
all threads must arrive
all shared-memory writes before the barrier become visible
then all threads continue
```

Correct pattern:

```c
tile[threadIdx.x] = global[i];
__syncthreads();
// now it is safe to read tile values written by other threads
```

Code meaning:

- First, each thread writes its own shared-memory slot.
- Then the whole block waits at `__syncthreads()`.
- After the barrier, all earlier shared-memory writes are visible to the block.

The barrier must be reached by every thread in the block. If only some threads
execute it, the block can deadlock.

## 4. What are shared memory banks?

Shared memory is divided into independent banks so many threads can access it in
parallel. The standard beginner mental model is:

```text
32 banks
FP32 word index maps to bank = index % 32
```

So:

```text
shared[0]  → bank 0
shared[1]  → bank 1
...
shared[31] → bank 31
shared[32] → bank 0 again
```

If a warp's 32 threads access 32 different banks, the access can be served in
parallel. If many threads access different addresses in the same bank, those
accesses serialize.

This is the shared-memory version of the coalescing idea:

```text
global memory: make transactions efficient
shared memory: spread accesses across banks
```

## 5. What causes a bank conflict?

A bank conflict happens when multiple threads in the same warp access **different
addresses** that map to the same shared-memory bank.

Conflict example:

```c
float x = shared[threadIdx.x * 32];
```

Code meaning:

- Thread 0 reads index 0.
- Thread 1 reads index 32.
- Thread 2 reads index 64.
- Those are different addresses, but they map to the same bank in the simple FP32
  model.

For a warp:

```text
thread 0 → shared[0]    → bank 0
thread 1 → shared[32]   → bank 0
thread 2 → shared[64]   → bank 0
...
thread 31 → shared[992] → bank 0
```

All 32 threads hit bank 0. The bank must serve them in multiple serialized steps,
so the access is much slower.

Important exception: if all threads read the **exact same address**, the hardware
can broadcast that value. That is not a harmful bank conflict.

```text
same address, same bank       → broadcast, fast
different addresses, same bank → conflict, serialized
```

## 6. Why does `[32][33]` fix column-wise bank conflicts?

A 2D shared tile:

```c
__shared__ float tile[32][32];
```

Code meaning:

- This allocates a 32-by-32 shared-memory tile per block.
- The row stride is 32 floats.
- That stride is exactly what creates the classic column-wise bank conflict.

is stored row-major. The address index is:

```text
index = row * 32 + col
```

For column-wise access:

```c
tile[threadIdx.x][constant_col]
```

Code meaning:

- `threadIdx.x` chooses the row.
- `constant_col` keeps the column fixed.
- Neighboring lanes therefore walk down rows in the same column.

the bank is:

```text
bank = (threadIdx.x * 32 + constant_col) % 32
     = constant_col
```

Every thread maps to the same bank. That creates a 32-way bank conflict.

With padding:

```c
__shared__ float tile[32][33];
```

Code meaning:

- This allocates a padded shared-memory tile.
- The logical tile still behaves like 32 useful columns.
- The extra column changes the row stride to 33, improving the bank mapping.

the address index becomes:

```text
index = row * 33 + col
```

For column-wise access:

```text
bank = (threadIdx.x * 33 + constant_col) % 32
     = (threadIdx.x + constant_col) % 32
```

Now consecutive threads land in consecutive banks. The extra column changes the
modulo arithmetic. You add padding not because you need more data, but because you
need a better memory layout for the hardware.

## 7. When is constant memory fast?

Constant memory is fast when all threads in a warp read the same address. In that
case, the hardware can broadcast one value to the whole warp.

Good pattern:

```c
float scale = c_params[0];  // every lane reads c_params[0]
```

Code meaning:

- Every lane reads the same constant-memory address.
- That is the pattern constant memory is built to broadcast efficiently.

This is useful for:

- small filter coefficients
- scalar configuration values
- small read-only lookup tables when access is uniform
- fixed constants used by many threads

Bad or less useful pattern:

```c
float x = c_table[threadIdx.x];  // each lane reads a different address
```

Code meaning:

- Each lane reads a different constant-memory address.
- This breaks the ideal broadcast pattern.
- It may still work correctly, but it is usually less efficient.

Now the warp does not get one clean broadcast. Accesses may serialize or lose the
constant-cache advantage.

The rule:

```text
constant memory = tiny + read-only + warp-uniform access
```

It is not where LLM weights live. It is far too small for that.

## 8. Why is local memory slow despite its name?

In CUDA, "local" means local to a thread's private address space. It does **not**
mean physically close to the SM.

Local memory often lives in global memory/HBM. It appears when a thread has private
data that cannot fit in registers, such as:

- too many live variables
- large private arrays
- register spills
- compiler decisions caused by register pressure

Example:

```c
float tmp[128];  // private per-thread array
```

Code meaning:

- Every thread gets its own private 128-float temporary array.
- Because it is private, it is not shared memory.
- Because it is large, the compiler may store it in local memory instead of registers.

This may become local memory. Every thread has its own `tmp`, but those values may
be stored in global memory. That makes accesses much slower than registers.

This is why register pressure matters. A kernel can become slower after adding more
fusion or more temporary variables because the compiler starts spilling to local
memory.

## 9. How do register spills show up in compiler output?

Compile with:

```bash
nvcc -Xptxas -v my_kernel.cu -o my_kernel
```

Command meaning:

- Compile the CUDA program.
- Ask `ptxas` to print kernel resource usage.
- Use the output to check register count and spills.

Look for lines like:

```text
ptxas info    : Used 64 registers
ptxas info    : 384 bytes spill stores
ptxas info    : 384 bytes spill loads
```

The important words are:

```text
spill stores
spill loads
```

They mean the compiler could not keep all private thread values in registers. It
stored some values in local memory and later loaded them back.

Why this matters:

```text
register access → very fast, on-chip
spill/local access → much slower, often global memory path
```

Spills also indicate a resource tradeoff. Maybe the kernel has too many temporary
variables. Maybe it fused too much work. Maybe reducing tile size or simplifying
the kernel would improve occupancy and remove spills.

## 10. How does shared memory prepare you for tiled GEMM and Flash Attention?

Shared memory teaches the central GPU optimization pattern:

```text
load a tile from global memory
reuse it many times on-chip
avoid unnecessary HBM traffic
write final result back to global memory
```

In tiled GEMM:

```text
C = A × B
```

each block loads a tile of A and a tile of B into shared memory. Threads reuse
those tiles for many multiply-adds before loading the next tile. This raises
arithmetic intensity: more math per byte read from HBM.

In Flash Attention:

```text
softmax(QK^T)V
```

the kernel avoids materializing the huge attention matrix in global memory. It
loads Q/K/V blocks into on-chip memory, computes partial attention, maintains
running softmax statistics, and writes only the needed output.

Both are the same idea:

> Keep temporary/reused data close to the SM, and avoid round trips to HBM.

**Tomorrow (Day 3): Backpropagation & Gradient Descent** — how neural networks
learn by pushing errors backward through the computation graph.

---

*Status: ⬜ NOT YET COMPLETED*
*Date completed: ___________*
