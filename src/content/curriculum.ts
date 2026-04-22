import type { Module } from '../types';

export const curriculum: Module[] = [
  // ===================================================================
  // MODULE 1: Hello, Triton
  // ===================================================================
  {
    id: 'mod-1',
    title: 'Hello, Triton',
    description:
      'Understand what Triton is, the GPU programming model, and write your first kernel.',
    icon: 'cpu',
    difficulty: 'beginner',
    lessons: [
      {
        id: 'mod-1-lesson-1',
        title: 'What is Triton?',
        content: `## What is Triton?

**Triton** is a programming language and compiler by OpenAI that makes writing GPU kernels dramatically simpler than CUDA.

### Why Triton?

Writing fast GPU code with CUDA requires managing:
- Thread blocks and warps
- Shared memory and synchronization
- Memory coalescing patterns
- Register pressure

Triton abstracts these away. You think in **blocks of data** rather than individual threads.

### The Key Idea

In Triton, you write a kernel that operates on a **block** of elements at a time. The Triton compiler handles mapping that to GPU threads, managing shared memory, and optimizing memory access.

\`\`\`python
# CUDA: you think about individual threads
# Triton: you think about blocks of data
\`\`\`

### How This Simulator Works

In this guide, we use a **NumPy-based Triton simulator**. Your code uses the same Triton API, but executes on CPU via NumPy. This means:

- Same syntax as real Triton
- No GPU required
- Results are mathematically identical
- You can learn all concepts before touching real hardware

> The simulator is automatically loaded. Just \`import triton\` and \`import triton.language as tl\`.

Try running the example code to see the simulator in action!`,
        code: `import triton
import triton.language as tl
import numpy as np

# The simulator is ready!
print("Triton simulator version: NumPy backend")
print(f"NumPy version: {np.__version__}")

# Quick demo: triton utility functions
print(f"\\nCeiling division: triton.cdiv(10, 3) = {triton.cdiv(10, 3)}")
print(f"Next power of 2: triton.next_power_of_2(13) = {triton.next_power_of_2(13)}")
`,
      },
      {
        id: 'mod-1-lesson-gpu-primer',
        title: 'A Short Note on GPUs',
        content: `## A Short Note on GPUs

Before we write any Triton, a minute on what a GPU actually is. If you're comfortable with this, skip ahead.

### Why GPUs are fast

A CPU has a handful of powerful cores (say, 16) optimized for branchy, sequential code. A GPU has *thousands* of simpler cores optimized to do the **same instruction** on **different data** at the same time. This is the SIMD/SPMD model: one program, many data points.

For operations that look like *"add these million numbers pairwise"*, a GPU is roughly two orders of magnitude faster than a CPU. For operations that look like *"parse this JSON"*, it's not.

Machine learning lives in the first camp. A forward pass through a large language model is billions of floating-point operations arranged in a way that maps almost perfectly to the GPU's strengths: matmuls, element-wise ops, softmaxes, reductions. That's why every modern LLM runs on one.

### The CPU-GPU dance

Your Python code runs on the CPU. Your tensors live in GPU memory. When you call a kernel, you're not calling a Python function; you're:

1. Telling the GPU *"run this code, this many times in parallel"*
2. Passing it pointers to GPU memory
3. Waiting for it to finish
4. Using the result back on the CPU

In Triton, that handoff looks like:

\`\`\`python
add_kernel[(grid_size,)](x, y, output, n, BLOCK_SIZE=256)
\`\`\`

The \`[grid]\` part is the launch configuration. \`(grid_size,)\` says *"run this kernel \`grid_size\` independent times in parallel"*. Each instance is called a **program**, and each program handles a different piece of the data.

### CUDA vs Triton

Here's what vector addition looks like in CUDA (skim it; you won't be writing this):

\`\`\`c
__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}
\`\`\`

Each **thread** computes *one* element. You have to think about thread indices, block sizes, warp scheduling, shared memory banks, coalesced loads. CUDA gives you power, but makes you manage every detail.

In Triton, you write:

\`\`\`python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)
\`\`\`

Each **program** handles a *block* of elements. The Triton compiler figures out the rest: how to map your block to threads, where to stage data, how to coalesce loads. You trade some fine-grained control for a lot less complexity.

This guide teaches you to think in blocks, not threads. Next tutorial, we'll make that concrete.`,
        code: `# A quick intuition for what a kernel launch computes

# Say we want to add two vectors of length n, in blocks of BLOCK_SIZE.
# The number of programs we need to launch is ceil(n / BLOCK_SIZE).

sizes = [100, 1000, 10_000, 100_000, 1_000_000]
block_size = 256

print(f"{'n':>12} {'block':>8} {'programs':>12}")
print("-" * 36)
for n in sizes:
    programs = -(-n // block_size)  # ceiling division in Python
    print(f"{n:>12} {block_size:>8} {programs:>12}")

print()
print("On a real GPU, those programs run in parallel across thousands of cores.")
print("In this guide's simulator, we run them sequentially for learning purposes.")
`,
      },
      {
        id: 'mod-1-lesson-2',
        title: 'The GPU Programming Model',
        content: `## The GPU Programming Model

GPUs execute thousands of operations in parallel. To harness this, we need to understand how work is organized.

### Grids and Programs

When you launch a Triton kernel, you specify a **grid**: the number of independent program instances to run.

\`\`\`python
# Launch 4 programs
kernel[(4,)](...)
\`\`\`

Each program gets a unique **program ID** via \`tl.program_id(axis)\`. With a 1D grid, each program processes a different chunk of data.

### Blocks

Each program operates on a **block** of elements. The block size is typically a power of 2 (64, 128, 256, 1024).

\`\`\`python
BLOCK_SIZE = 256
pid = tl.program_id(0)       # Which program am I?
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# offsets = [pid*256, pid*256+1, ..., pid*256+255]
\`\`\`

### The Pattern

Almost every Triton kernel follows this pattern:

1. **Get program ID**: \`pid = tl.program_id(0)\`
2. **Compute offsets**: which elements does this program handle?
3. **Load data**: read from input arrays
4. **Compute**: do the actual work
5. **Store results**: write to output array

### Masks

When data size isn't a perfect multiple of BLOCK_SIZE, some programs will read past the end. We use **masks** to handle boundaries safely.

\`\`\`python
mask = offsets < n_elements  # True for valid positions
\`\`\`

Try the example to see program IDs and offsets in action!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def demo_kernel(n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program gets a unique ID
    pid = tl.program_id(axis=0)

    # Compute which elements this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask for boundary handling
    mask = offsets < n_elements

    # Print info (using Python print in simulator)
    print(f"  Program {pid}: offsets [{block_start}..{block_start + BLOCK_SIZE - 1}], "
          f"valid elements: {int(tl.sum(mask.astype(np.int32)))}")

n = 10
BLOCK_SIZE = 4
grid_size = triton.cdiv(n, BLOCK_SIZE)  # = 3

print(f"Array size: {n}, Block size: {BLOCK_SIZE}")
print(f"Grid size: {grid_size} programs\\n")

demo_kernel[(grid_size,)](n, BLOCK_SIZE=BLOCK_SIZE)

print(f"\\nNotice: Program 2 only has {n - 2*BLOCK_SIZE} valid elements (mask handles the rest)")
`,
      },
      {
        id: 'mod-1-lesson-3',
        title: 'Your First Kernel: Vector Add',
        content: `## Your First Kernel: Vector Addition

Let's write a complete Triton kernel that adds two vectors element-wise:

\`\`\`
output[i] = x[i] + y[i]
\`\`\`

### The Kernel

\`\`\`python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
\`\`\`

### Breaking It Down

1. **\`@triton.jit\`**: Marks this as a Triton kernel
2. **\`x_ptr, y_ptr, output_ptr\`**: Pointers to arrays in GPU memory
3. **\`BLOCK_SIZE: tl.constexpr\`**: Compile-time constant (not a runtime variable)
4. **\`tl.load(ptr + offsets, mask)\`**: Load a block of elements
5. **\`tl.store(ptr + offsets, value, mask)\`**: Write a block of elements

### Pointer Arithmetic

In Triton, when you write \`x_ptr + offsets\`, it creates a block of pointers. Each pointer addresses one element. \`tl.load\` reads all of them in one operation.

### Launching the Kernel

\`\`\`python
grid = (triton.cdiv(n, BLOCK_SIZE),)
add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)
\`\`\`

The \`[grid]\` syntax specifies how many program instances to launch.

Run the example to see it work!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Adds two vectors element-wise."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input blocks
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute and store
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# --- Driver code ---
n = 20
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
output = np.zeros(n, dtype=np.float32)

BLOCK_SIZE = 8
grid = (triton.cdiv(n, BLOCK_SIZE),)

print(f"Adding two vectors of size {n}")
print(f"Grid: {grid[0]} programs, Block size: {BLOCK_SIZE}\\n")

add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)

# Verify
expected = x + y
max_err = np.max(np.abs(output - expected))
print(f"Results:  {output}")
print(f"Expected: {expected}")
print(f"Max error: {max_err:.2e}")
print(f"Correct: {np.allclose(output, expected)}")
`,
      },
    ],
    puzzles: [
      {
        id: 'mod-1-puzzle-1',
        title: 'Puzzle: Vector Multiply',
        description: `## Vector Multiply

Write a Triton kernel that multiplies two vectors element-wise:

\`\`\`
output[i] = x[i] * y[i]
\`\`\`

This is very similar to vector addition. You need to:

1. Get the program ID
2. Compute block offsets
3. Create a mask for boundary handling
4. Load both input vectors
5. **Multiply** them (instead of adding)
6. Store the result

The test will verify your kernel produces the correct output.`,
        difficulty: 'easy',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO: Implement element-wise multiplication
    # 1. Get program ID
    # 2. Compute offsets
    # 3. Create mask
    # 4. Load x and y
    # 5. Multiply and store
    pass

# --- Driver code (do not modify) ---
n = 20
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
output = np.zeros(n, dtype=np.float32)

grid = (triton.cdiv(n, 8),)
mul_kernel[grid](x, y, output, n, BLOCK_SIZE=8)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * y, mask=mask)`,
        testCode: `# Verification
expected = x * y
assert np.allclose(output, expected, atol=1e-5), f"Wrong result! Max error: {np.max(np.abs(output - expected)):.6f}"
print(f"All {n} elements correct!")
print(f"Sample: output[:5] = {output[:5]}")
print(f"Expected:           {expected[:5]}")
print("PASSED")`,
        hints: [
          'Start with tl.program_id(axis=0) to get the current program ID.',
          'Compute offsets as: pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)',
          'The only difference from vector add is using * instead of +',
        ],
      },
      {
        id: 'mod-1-puzzle-2',
        title: 'Puzzle: Vector Subtract',
        description: `## Vector Subtract

Write a Triton kernel that subtracts one vector from another, element-wise:

\`\`\`
output[i] = x[i] - y[i]
\`\`\`

The skeleton is almost the vector-add kernel from Tutorial 1. You need to:

1. Get the program ID
2. Compute block offsets
3. Create a mask for safe boundary handling
4. Load both input vectors
5. **Subtract** \`y\` from \`x\`
6. Store the result

The test will compare your output against \`x - y\` computed in NumPy.`,
        difficulty: 'easy',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def sub_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO: Implement element-wise subtraction (x - y)
    # 1. Get program ID
    # 2. Compute offsets
    # 3. Create mask
    # 4. Load x and y
    # 5. Subtract and store
    pass

# --- Driver code (do not modify) ---
n = 20
np.random.seed(1)
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
output = np.zeros(n, dtype=np.float32)

grid = (triton.cdiv(n, 8),)
sub_kernel[grid](x, y, output, n, BLOCK_SIZE=8)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def sub_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x - y, mask=mask)`,
        testCode: `expected = x - y
max_err = float(np.max(np.abs(output - expected)))
assert np.allclose(output, expected, atol=1e-5), f"Wrong result! Max error: {max_err:.6f}"
print(f"All {n} elements correct!")
print(f"Sample: output[:5] = {output[:5]}")
print(f"Expected:           {expected[:5]}")
print("PASSED")`,
        hints: [
          'Copy the vector-multiply structure: pid, offsets, mask, load, store.',
          'offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE); mask = offsets < n_elements.',
          'The only change from add is the operator: tl.store(output_ptr + offsets, x - y, mask=mask).',
        ],
      },
    ],
    quiz: {
      id: 'mod-1-quiz',
      questions: [
        {
          id: 'mod-1-q1',
          question:
            'What does tl.program_id(0) return in a Triton kernel?',
          options: [
            'The thread index within a warp',
            'The unique ID of the current program instance',
            'The total number of programs',
            'The block size',
          ],
          correctIndex: 1,
          explanation:
            'tl.program_id(axis) returns the unique identifier of the current program instance along the specified grid axis. Each program handles a different block of data.',
        },
        {
          id: 'mod-1-q2',
          question: 'Why do we use masks in Triton kernels?',
          options: [
            'To improve performance',
            'To enable parallel execution',
            'To handle cases where data size is not a multiple of block size',
            'To select which GPU threads to use',
          ],
          correctIndex: 2,
          explanation:
            'Masks prevent out-of-bounds memory access when the total number of elements is not perfectly divisible by the block size. The last block may have fewer valid elements.',
        },
        {
          id: 'mod-1-q3',
          question: 'What does triton.cdiv(10, 3) return?',
          options: ['3', '4', '3.33', '10'],
          correctIndex: 1,
          explanation:
            'triton.cdiv performs ceiling division: (10 + 3 - 1) // 3 = 4. This ensures we launch enough programs to cover all elements.',
        },
        {
          id: 'mod-1-q4',
          question:
            'What does the tl.constexpr annotation on a kernel parameter mean?',
          options: [
            'The parameter is a floating point number',
            'The parameter can change at runtime',
            'The parameter is a compile-time constant known before kernel execution',
            'The parameter is optional',
          ],
          correctIndex: 2,
          explanation:
            'tl.constexpr marks a parameter as a compile-time constant. Triton uses this for optimizations like unrolling loops and sizing arrays. BLOCK_SIZE is the most common constexpr.',
        },
      ],
    },
  },

  // ===================================================================
  // MODULE 2: Memory & Pointers
  // ===================================================================
  {
    id: 'mod-2',
    title: 'Memory & Pointers',
    description:
      'Master Triton\'s pointer model, memory access patterns, and safe boundary handling.',
    icon: 'hard-drive',
    difficulty: 'beginner',
    lessons: [
      {
        id: 'mod-2-lesson-1',
        title: 'Pointers and Offsets',
        content: `## Pointers and Offsets

In Triton, data lives in GPU memory. Kernels receive **pointers** to this data and compute **offsets** to access specific elements.

### How Pointers Work

When you pass an array to a kernel, Triton gives you a pointer to its first element. You use arithmetic to reach other elements:

\`\`\`python
# ptr points to element 0
# ptr + 1 points to element 1
# ptr + offsets points to a block of elements
\`\`\`

### Block Pointers

\`tl.arange(0, BLOCK_SIZE)\` creates a vector of indices. Adding this to a pointer creates a **block pointer**: a vector of pointers, each addressing a different element:

\`\`\`python
offsets = tl.arange(0, BLOCK_SIZE)  # [0, 1, 2, ..., 255]
block_ptrs = ptr + offsets           # points to elements [0..255]
\`\`\`

### Strided Access

For non-contiguous access (e.g., every other element, or column access in a matrix), you multiply offsets by a stride:

\`\`\`python
# Access every other element
offsets = tl.arange(0, BLOCK_SIZE) * 2  # [0, 2, 4, ...]

# Access a column in a row-major matrix
row_offsets = tl.arange(0, BLOCK_SIZE) * num_cols
\`\`\`

Run the example to see different access patterns!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def strided_read_kernel(input_ptr, output_ptr, stride, n, BLOCK_SIZE: tl.constexpr):
    """Read every 'stride'-th element."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Strided access: multiply offsets by stride
    input_offsets = offsets * stride
    mask = input_offsets < n

    vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, vals, mask=offsets < (n // stride))

# Demo: read every 3rd element
data = np.arange(30, dtype=np.float32)
print(f"Input:  {data}")

output = np.zeros(10, dtype=np.float32)
strided_read_kernel[(1,)](data, output, 3, 30, BLOCK_SIZE=16)

print(f"Every 3rd element: {output[:10]}")
print(f"Expected:          {data[::3]}")
`,
      },
      {
        id: 'mod-2-lesson-2',
        title: 'Masks and Boundary Handling',
        content: `## Masks and Boundary Handling

GPU kernels process data in fixed-size blocks, but real data rarely fits perfectly. Masks are how Triton handles this mismatch safely.

### The Problem

If you have 1000 elements and BLOCK_SIZE=256:
- Programs 0-2 handle elements [0..767], all valid
- Program 3 handles elements [768..1023], only 232 are valid!

Without a mask, program 3 would read/write 24 invalid elements.

### Creating Masks

\`\`\`python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements  # Boolean array
\`\`\`

### Using Masks

**Loading with mask:**
\`\`\`python
x = tl.load(ptr + offsets, mask=mask, other=0.0)
# Invalid positions get 'other' value (default 0.0)
\`\`\`

**Storing with mask:**
\`\`\`python
tl.store(ptr + offsets, result, mask=mask)
# Only valid positions are written
\`\`\`

### The \`other\` Parameter

When loading with a mask, \`other\` specifies what value to use for masked-out positions. This is important for reductions:

- **Sum**: use \`other=0.0\` (zeros don't affect sum)
- **Max**: use \`other=float('-inf')\` (won't be the max)
- **Product**: use \`other=1.0\` (ones don't affect product)

Run the example to see masks in action!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def safe_copy_kernel(src_ptr, dst_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load with mask - out-of-bounds positions get -999
    data = tl.load(src_ptr + offsets, mask=mask, other=-999.0)

    # Only store valid positions
    tl.store(dst_ptr + offsets, data, mask=mask)

# Demo: 7 elements with BLOCK_SIZE=4 (needs 2 programs)
n = 7
src = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.float32)
dst = np.full(8, -1.0, dtype=np.float32)  # extra slot to show masking works

print(f"Source ({n} elements): {src}")
print(f"Dest before: {dst}")

grid = (triton.cdiv(n, 4),)
print(f"Grid size: {grid[0]} programs, BLOCK_SIZE=4\\n")

safe_copy_kernel[grid](src, dst, n, BLOCK_SIZE=4)

print(f"Dest after:  {dst}")
print(f"Element 7 (out of bounds) untouched: {dst[7] == -1.0}")
`,
      },
    ],
    puzzles: [
      {
        id: 'mod-2-puzzle-1',
        title: 'Puzzle: Reverse Array',
        description: `## Reverse Array

Write a kernel that reverses an array:

\`\`\`
output[i] = input[n - 1 - i]
\`\`\`

The trick is computing the correct **read offsets** to load elements in reverse order, while writing them in forward order.

Each program should:
1. Compute its output write offsets normally
2. Compute the corresponding input read offsets (reversed)
3. Load from the reversed positions
4. Store to the forward positions`,
        difficulty: 'easy',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def reverse_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Output offsets (where to write)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < n

    # TODO: Compute input offsets (where to read from, reversed)
    # Hint: if output position is i, input position is (n - 1 - i)
    in_offsets = out_offsets  # FIX THIS

    data = tl.load(input_ptr + in_offsets, mask=mask)
    tl.store(output_ptr + out_offsets, data, mask=mask)

# --- Driver code ---
n = 10
x = np.arange(n, dtype=np.float32)
output = np.zeros(n, dtype=np.float32)

grid = (triton.cdiv(n, 4),)
reverse_kernel[grid](x, output, n, BLOCK_SIZE=4)

print(f"Input:    {x}")
print(f"Reversed: {output}")
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def reverse_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < n

    in_offsets = (n - 1) - out_offsets

    data = tl.load(input_ptr + in_offsets, mask=mask)
    tl.store(output_ptr + out_offsets, data, mask=mask)`,
        testCode: `expected = x[::-1]
assert np.allclose(output, expected), f"Wrong! Got {output}, expected {expected}"
print(f"Input:    {x}")
print(f"Reversed: {output}")
print(f"Expected: {expected}")
print("PASSED")`,
        hints: [
          'If the output position is i, where should you read from in the input?',
          'The reversed position for index i is: (n - 1 - i)',
          'Replace in_offsets = out_offsets with: in_offsets = (n - 1) - out_offsets',
        ],
      },
      {
        id: 'mod-2-puzzle-2',
        title: 'Puzzle: Strided Copy',
        description: `## Strided Copy

Copy every *k*-th element of an input array into a compact output array:

\`\`\`
output[i] = input[i * stride]   for i in range(n_out)
\`\`\`

This is the pattern behind reading a column of a row-major matrix, or slicing \`x[::stride]\`. The read offsets are *strided* (multiply by \`stride\`), while the write offsets are contiguous.

Steps:

1. Compute contiguous **write** offsets: \`pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\`
2. Compute **read** offsets: the same values, multiplied by \`stride\`
3. Mask against \`n_out\` so you don't read past the source or write past the destination
4. Load from the strided positions, store to the contiguous positions`,
        difficulty: 'easy',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def strided_copy_kernel(src_ptr, dst_ptr, stride, n_out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < n_out

    # TODO: Compute the read offsets (every 'stride'-th element)
    in_offsets = out_offsets  # FIX THIS

    data = tl.load(src_ptr + in_offsets, mask=mask)
    tl.store(dst_ptr + out_offsets, data, mask=mask)

# --- Driver code ---
src = np.arange(32, dtype=np.float32)
stride = 3
n_out = len(src) // stride   # 10
output = np.zeros(n_out, dtype=np.float32)

grid = (triton.cdiv(n_out, 8),)
strided_copy_kernel[grid](src, output, stride, n_out, BLOCK_SIZE=8)

print(f"Source:   {src}")
print(f"Strided:  {output}")
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def strided_copy_kernel(src_ptr, dst_ptr, stride, n_out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < n_out

    in_offsets = out_offsets * stride

    data = tl.load(src_ptr + in_offsets, mask=mask)
    tl.store(dst_ptr + out_offsets, data, mask=mask)`,
        testCode: `expected = src[::stride][:n_out]
assert np.allclose(output, expected), f"Wrong! Got {output}, expected {expected}"
print(f"Source:   {src}")
print(f"Strided:  {output}")
print(f"Expected: {expected}")
print("PASSED")`,
        hints: [
          'The write offsets are contiguous; only the *read* offsets need to change.',
          'Tutorial 1 showed this pattern: multiply offsets by the stride.',
          'Replace in_offsets = out_offsets with in_offsets = out_offsets * stride.',
        ],
      },
    ],
    quiz: {
      id: 'mod-2-quiz',
      questions: [
        {
          id: 'mod-2-q1',
          question:
            'What happens if you call tl.load without a mask on data smaller than BLOCK_SIZE?',
          options: [
            'It automatically handles the boundary',
            'It reads garbage or crashes due to out-of-bounds access',
            'It returns zeros for invalid positions',
            'It raises a compile error',
          ],
          correctIndex: 1,
          explanation:
            'Without a mask, Triton will attempt to read from all positions including out-of-bounds ones, which can read garbage data or cause a memory access violation.',
        },
        {
          id: 'mod-2-q2',
          question:
            'When computing a maximum reduction, what should the "other" value be for masked loads?',
          options: [
            '0.0',
            '1.0',
            'float("-inf") (negative infinity)',
            'float("inf") (positive infinity)',
          ],
          correctIndex: 2,
          explanation:
            'Negative infinity ensures masked-out positions won\'t affect the maximum. Any real value in the data will be larger than -inf.',
        },
        {
          id: 'mod-2-q3',
          question:
            'To access every 4th element of an array, how should you compute offsets?',
          options: [
            'tl.arange(0, BLOCK_SIZE)',
            'tl.arange(0, BLOCK_SIZE) + 4',
            'tl.arange(0, BLOCK_SIZE) * 4',
            'tl.arange(0, BLOCK_SIZE * 4)',
          ],
          correctIndex: 2,
          explanation:
            'Multiplying by the stride (4) gives offsets [0, 4, 8, 12, ...], accessing every 4th element.',
        },
        {
          id: 'mod-2-q4',
          question:
            'What is the purpose of pointer arithmetic like "x_ptr + offsets" in Triton?',
          options: [
            'It modifies the original array',
            'It creates a block of pointers to specific memory locations',
            'It copies the data to a new location',
            'It allocates new memory',
          ],
          correctIndex: 1,
          explanation:
            'Adding an offset array to a pointer creates a block pointer: a collection of addresses that tl.load/tl.store can use to access multiple elements at once.',
        },
      ],
    },
  },

  // ===================================================================
  // MODULE 3: Element-wise Operations
  // ===================================================================
  {
    id: 'mod-3',
    title: 'Element-wise Operations',
    description:
      'Build activation functions, learn operation fusion, and understand why custom kernels matter.',
    icon: 'zap',
    difficulty: 'intermediate',
    lessons: [
      {
        id: 'mod-3-lesson-1',
        title: 'Activation Functions',
        content: `## Activation Functions in Triton

Neural networks rely on activation functions like ReLU, Sigmoid, and GELU. Writing these as Triton kernels lets you fuse them with other operations for better performance.

### ReLU

The simplest activation: \`relu(x) = max(0, x)\`

\`\`\`python
result = tl.maximum(x, 0.0)
\`\`\`

### Sigmoid

\`sigmoid(x) = 1 / (1 + exp(-x))\`

\`\`\`python
result = 1.0 / (1.0 + tl.exp(-x))
# or simply: tl.sigmoid(x)
\`\`\`

### GELU (Gaussian Error Linear Unit)

Used in transformers (BERT, GPT). The approximate version:

\`\`\`python
# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
\`\`\`

### Why Custom Kernels?

In PyTorch, \`F.gelu(x)\` works fine. But if you need \`F.gelu(x * weight + bias)\`, that's 3 separate kernel launches (multiply, add, gelu). A Triton kernel does it in **one launch**, saving memory bandwidth.

This is called **kernel fusion**, the topic of our next tutorial.

Run the example to see all three activations!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def activations_kernel(x_ptr, relu_ptr, sigmoid_ptr, gelu_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)

    # ReLU
    relu_out = tl.maximum(x, 0.0)
    tl.store(relu_ptr + offsets, relu_out, mask=mask)

    # Sigmoid
    sigmoid_out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(sigmoid_ptr + offsets, sigmoid_out, mask=mask)

    # GELU (approximate)
    k = 0.7978845608  # sqrt(2/pi)
    gelu_out = 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x * x * x)))
    tl.store(gelu_ptr + offsets, gelu_out, mask=mask)

# ---
n = 8
x = np.linspace(-3, 3, n).astype(np.float32)
relu = np.zeros(n, dtype=np.float32)
sigmoid = np.zeros(n, dtype=np.float32)
gelu = np.zeros(n, dtype=np.float32)

grid = (triton.cdiv(n, 128),)
activations_kernel[grid](x, relu, sigmoid, gelu, n, BLOCK_SIZE=128)

print(f"{'x':>8} {'ReLU':>8} {'Sigmoid':>8} {'GELU':>8}")
print("-" * 36)
for i in range(n):
    print(f"{x[i]:8.3f} {relu[i]:8.3f} {sigmoid[i]:8.3f} {gelu[i]:8.3f}")
`,
      },
      {
        id: 'mod-3-lesson-2',
        title: 'Fusing Operations',
        content: `## Fusing Operations

**Fusion** means combining multiple operations into a single kernel. This is one of the most powerful benefits of writing custom Triton kernels.

### The Memory Bandwidth Problem

Modern GPUs compute much faster than they can move data. For simple operations like ReLU:

\`\`\`
Without fusion (3 kernel launches):
1. Load x, compute x*w, store intermediate  (read + write)
2. Load intermediate, add bias, store        (read + write)
3. Load result, apply relu, store            (read + write)
= 6 memory operations
\`\`\`

\`\`\`
With fusion (1 kernel launch):
1. Load x, compute relu(x*w + bias), store  (1 read + 1 write)
= 2 memory operations
\`\`\`

That's a **3x reduction** in memory traffic!

### Fusion in Triton

Fusing is natural in Triton; you just write all operations in the same kernel:

\`\`\`python
@triton.jit
def fused_kernel(x_ptr, w_ptr, bias_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    result = tl.maximum(x * w + b, 0.0)  # fused: mul + add + relu
    tl.store(out_ptr + offs, result, mask=mask)
\`\`\`

Try the example comparing fused vs. separate operations!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def fused_linear_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr
):
    """Computes ReLU(x * weight + bias) in a single kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(weight_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)

    # All three operations fused into one kernel
    result = tl.maximum(x * w + b, 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)

# ---
n = 24
x = np.random.randn(n).astype(np.float32)
weight = np.random.randn(n).astype(np.float32) * 0.1
bias = np.random.randn(n).astype(np.float32) * 0.01
output = np.zeros(n, dtype=np.float32)

grid = (triton.cdiv(n, 8),)
fused_linear_relu_kernel[grid](x, weight, bias, output, n, BLOCK_SIZE=8)

# Verify against numpy
expected = np.maximum(x * weight + bias, 0.0)
print(f"Fused ReLU(x*w+b) kernel: 4 arrays, 3 programs")
print(f"Max error: {np.max(np.abs(output - expected)):.2e}")
print(f"Correct: {np.allclose(output, expected)}")
print(f"% of activations > 0: {np.mean(output > 0)*100:.1f}%")
`,
      },
    ],
    puzzles: [
      {
        id: 'mod-3-puzzle-1',
        title: 'Puzzle: Fused GELU + Scale',
        description: `## Fused GELU + Scale

Write a kernel that computes:

\`\`\`
output[i] = scale * GELU(x[i])
\`\`\`

Where GELU is the approximate version:
\`\`\`
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
\`\`\`

And \`tanh\` can be computed as:
\`\`\`
tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
\`\`\`

Or use \`np.tanh\` in the simulator.

**Key constant:** \`sqrt(2/pi) ≈ 0.7978845608\``,
        difficulty: 'medium',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def fused_gelu_scale_kernel(x_ptr, output_ptr, scale, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)

    # TODO: Compute GELU(x) * scale
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # sqrt(2/pi) ≈ 0.7978845608
    result = x  # FIX THIS

    tl.store(output_ptr + offsets, result, mask=mask)

# --- Driver code ---
n = 24
x = np.random.randn(n).astype(np.float32)
output = np.zeros(n, dtype=np.float32)
scale = 2.0

grid = (triton.cdiv(n, 8),)
fused_gelu_scale_kernel[grid](x, output, scale, n, BLOCK_SIZE=8)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def fused_gelu_scale_kernel(x_ptr, output_ptr, scale, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)

    k = 0.7978845608  # sqrt(2/pi)
    gelu = 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x * x * x)))
    result = scale * gelu

    tl.store(output_ptr + offsets, result, mask=mask)`,
        testCode: `# Verify
k = 0.7978845608
expected_gelu = 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x**3)))
expected = scale * expected_gelu
max_err = np.max(np.abs(output - expected))
assert max_err < 1e-4, f"Wrong result! Max error: {max_err:.6f}"
print(f"Max error: {max_err:.2e}")
print(f"Sample output[:5]: {output[:5]}")
print(f"Expected:          {expected[:5]}")
print("PASSED")`,
        hints: [
          'Start by computing the inner part: x + 0.044715 * x * x * x',
          'Multiply by sqrt(2/pi) ≈ 0.7978845608 and apply np.tanh',
          'GELU = 0.5 * x * (1.0 + tanh_result), then multiply by scale',
        ],
      },
      {
        id: 'mod-3-puzzle-2',
        title: 'Puzzle: Fused GELU + Bias Add',
        description: `## Fused GELU + Bias Add

Write a kernel that computes, in one pass:

\`\`\`
output[i] = GELU(x[i] + bias[i])
\`\`\`

This is the fused pattern you'd see after a linear projection in a transformer MLP block. Two memory reads, one write, no intermediate buffer.

Use the approximate GELU from Tutorial 1:

\`\`\`
GELU(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
\`\`\`

Constant: \`sqrt(2/pi) ≈ 0.7978845608\`. In the simulator, \`np.tanh\` works on the loaded block.

Steps:

1. Load \`x\` and \`bias\` with matching offsets and mask
2. Compute \`z = x + bias\`
3. Apply approximate GELU to \`z\`
4. Store the result`,
        difficulty: 'medium',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def fused_gelu_bias_kernel(x_ptr, bias_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # TODO: Compute GELU(x + bias)
    # 1. z = x + bias
    # 2. GELU(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    result = x + bias  # FIX THIS

    tl.store(output_ptr + offsets, result, mask=mask)

# --- Driver code ---
n = 24
np.random.seed(3)
x = np.random.randn(n).astype(np.float32)
bias = np.random.randn(n).astype(np.float32) * 0.1
output = np.zeros(n, dtype=np.float32)

grid = (triton.cdiv(n, 8),)
fused_gelu_bias_kernel[grid](x, bias, output, n, BLOCK_SIZE=8)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def fused_gelu_bias_kernel(x_ptr, bias_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    z = x + bias
    k = 0.7978845608  # sqrt(2/pi)
    result = 0.5 * z * (1.0 + np.tanh(k * (z + 0.044715 * z * z * z)))

    tl.store(output_ptr + offsets, result, mask=mask)`,
        testCode: `k = 0.7978845608
z = x + bias
expected = 0.5 * z * (1.0 + np.tanh(k * (z + 0.044715 * z**3)))
max_err = float(np.max(np.abs(output - expected)))
assert max_err < 1e-4, f"Wrong! Max error: {max_err:.6f}"
print(f"Max error: {max_err:.2e}")
print(f"Sample output[:5]: {output[:5]}")
print(f"Expected:          {expected[:5]}")
print("PASSED")`,
        hints: [
          'First sum the two loads into z = x + bias, then apply GELU to z (not to x).',
          'Copy the GELU formula from the Fused GELU + Scale puzzle, but pass z instead of x.',
          'Remember np.tanh works on the block: tanh_in = 0.7978845608 * (z + 0.044715 * z * z * z); result = 0.5 * z * (1.0 + np.tanh(tanh_in)).',
        ],
      },
    ],
    quiz: {
      id: 'mod-3-quiz',
      questions: [
        {
          id: 'mod-3-q1',
          question: 'What is the main benefit of kernel fusion?',
          options: [
            'It makes the code shorter',
            'It reduces memory bandwidth usage by avoiding intermediate reads/writes',
            'It enables parallel execution',
            'It reduces the number of floating point operations',
          ],
          correctIndex: 1,
          explanation:
            'Fusion combines multiple operations into one kernel, eliminating the need to write intermediate results to memory and read them back. This saves memory bandwidth, which is often the bottleneck.',
        },
        {
          id: 'mod-3-q2',
          question: 'How does tl.maximum(x, 0.0) implement ReLU?',
          options: [
            'It rounds x to the nearest integer',
            'It returns x if x > 0, otherwise returns 0',
            'It clips x between 0 and 1',
            'It computes the absolute value of x',
          ],
          correctIndex: 1,
          explanation:
            'tl.maximum(x, 0.0) returns the element-wise maximum of x and 0, which is exactly the ReLU function: max(0, x).',
        },
        {
          id: 'mod-3-q3',
          question:
            'If you need to compute y = sigmoid(x * w + b), how many kernel launches would a naive PyTorch implementation use?',
          options: [
            '1 launch',
            '2 launches',
            '3 launches (multiply, add, sigmoid)',
            'It depends on the batch size',
          ],
          correctIndex: 2,
          explanation:
            'Without fusion, PyTorch launches a separate kernel for each operation: one for multiplication, one for addition, and one for sigmoid. A Triton kernel can do all three in one launch.',
        },
      ],
    },
  },

  // ===================================================================
  // MODULE 4: Reductions
  // ===================================================================
  {
    id: 'mod-4',
    title: 'Reductions',
    description:
      'Learn block reductions (sum, max, min) and implement the softmax function from scratch.',
    icon: 'git-merge',
    difficulty: 'intermediate',
    lessons: [
      {
        id: 'mod-4-lesson-1',
        title: 'Block Reductions',
        content: `## Block Reductions

A **reduction** combines many values into fewer values (or one). Common reductions: sum, max, min, mean.

### In Triton

Triton provides built-in reduction operations:

\`\`\`python
total = tl.sum(x, axis=0)      # sum of all elements
maximum = tl.max(x, axis=0)    # maximum value
minimum = tl.min(x, axis=0)    # minimum value
\`\`\`

These operate within a single program's block. Each program reduces its own block of data independently.

### Row-wise Reductions

For 2D data (matrices), you often need to reduce along one axis:

\`\`\`python
# For a row of shape [BLOCK_SIZE]:
row_sum = tl.sum(row, axis=0)     # scalar
row_max = tl.max(row, axis=0)     # scalar
\`\`\`

### The Row-per-Program Pattern

A common pattern: each program handles one row of a matrix.

\`\`\`python
pid = tl.program_id(0)  # which row
row_start = pid * stride
offsets = row_start + tl.arange(0, BLOCK_SIZE)
\`\`\`

This maps naturally to operations like softmax, layer norm, and attention, all of which need per-row reductions.

Run the example to see row-wise sum and max!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def row_reduce_kernel(input_ptr, sum_ptr, max_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Each program computes the sum and max of one row."""
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load one row
    row_start = row_idx * n_cols
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Reduce
    row_sum = tl.sum(row, axis=0)
    row_max = tl.max(row, axis=0)

    # Store scalar results
    tl.store(sum_ptr + row_idx, row_sum)
    tl.store(max_ptr + row_idx, row_max)

# ---
n_rows, n_cols = 4, 8
matrix = np.random.randn(n_rows, n_cols).astype(np.float32)
row_sums = np.zeros(n_rows, dtype=np.float32)
row_maxs = np.zeros(n_rows, dtype=np.float32)

print("Matrix:")
print(matrix)
print()

row_reduce_kernel[(n_rows,)](matrix, row_sums, row_maxs, n_cols, BLOCK_SIZE=16)

print(f"Row sums: {row_sums}")
print(f"Expected: {matrix.sum(axis=1)}")
print(f"Row maxs: {row_maxs}")
print(f"Expected: {matrix.max(axis=1)}")
`,
      },
      {
        id: 'mod-4-lesson-2',
        title: 'Softmax',
        content: `## Softmax in Triton

Softmax is one of the most important operations in deep learning. It converts a vector of numbers into a probability distribution:

\`\`\`
softmax(x_i) = exp(x_i) / sum(exp(x_j))
\`\`\`

### The Numerical Stability Trick

Computing \`exp(x)\` directly can overflow. The standard trick:

\`\`\`
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
\`\`\`

Subtracting the max doesn't change the result but prevents overflow.

### Softmax as Three Reductions

1. **Find max**: \`m = max(x)\`
2. **Compute exp and sum**: \`s = sum(exp(x - m))\`
3. **Normalize**: \`output = exp(x - m) / s\`

### In Triton (Row-wise)

Each program handles one row:

\`\`\`python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    x = tl.load(input_ptr + row * n_cols + offsets, mask=mask, other=float('-inf'))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum
    tl.store(output_ptr + row * n_cols + offsets, result, mask=mask)
\`\`\`

This is a real production pattern: Triton's fused softmax is faster than PyTorch's built-in on many GPUs!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row_start = row_idx * n_cols
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))

    # Numerically stable softmax
    x_max = tl.max(x, axis=0)
    x_safe = x - x_max
    x_exp = tl.exp(x_safe)
    x_sum = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum

    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

# ---
n_rows, n_cols = 3, 6
x = np.random.randn(n_rows, n_cols).astype(np.float32)
output = np.zeros_like(x)

softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=8)

# Verify
def np_softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

expected = np_softmax(x)

print("Input:")
print(x)
print("\\nSoftmax output:")
print(output)
print("\\nRow sums (should be 1.0):", output.sum(axis=1))
print(f"Max error vs numpy: {np.max(np.abs(output - expected)):.2e}")
`,
      },
    ],
    puzzles: [
      {
        id: 'mod-4-puzzle-1',
        title: 'Puzzle: Row-wise Normalize',
        description: `## Row-wise Normalize

Write a kernel that normalizes each row to have zero mean and unit variance:

\`\`\`
mean = sum(row) / n_cols
var = sum((row - mean)^2) / n_cols
output = (row - mean) / sqrt(var + epsilon)
\`\`\`

This is the core of **Layer Normalization**.

You'll need:
1. Load a row
2. Compute the mean (sum / n_cols)
3. Compute variance
4. Normalize: (x - mean) / sqrt(var + eps)

Use \`epsilon = 1e-5\` for numerical stability.`,
        difficulty: 'medium',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def normalize_kernel(input_ptr, output_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols

    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # TODO: Compute mean
    mean = 0.0  # FIX THIS

    # TODO: Compute variance
    var = 1.0  # FIX THIS

    # TODO: Normalize
    result = x  # FIX THIS

    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

# --- Driver code ---
n_rows, n_cols = 4, 8
x = np.random.randn(n_rows, n_cols).astype(np.float32) * 5 + 3
output = np.zeros_like(x)
eps = 1e-5

normalize_kernel[(n_rows,)](x, output, n_cols, eps, BLOCK_SIZE=16)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def normalize_kernel(input_ptr, output_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols

    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / n_cols
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
    result = (x - mean) / tl.sqrt(var + eps)

    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)`,
        testCode: `# Verify
mean = x.mean(axis=1, keepdims=True)
var = x.var(axis=1, keepdims=True)
expected = (x - mean) / np.sqrt(var + eps)
max_err = np.max(np.abs(output - expected))
assert max_err < 1e-4, f"Wrong! Max error: {max_err:.6f}"
print(f"Output means (should be ~0): {output.mean(axis=1)}")
print(f"Output stds (should be ~1):  {output.std(axis=1)}")
print(f"Max error: {max_err:.2e}")
print("PASSED")`,
        hints: [
          'Mean: tl.sum(x, axis=0) / n_cols',
          'Variance: tl.sum((x - mean) * (x - mean), axis=0) / n_cols',
          'Normalize: (x - mean) / tl.sqrt(var + eps)',
        ],
      },
      {
        id: 'mod-4-puzzle-2',
        title: 'Puzzle: Row-wise Max',
        description: `## Row-wise Max

For each row of a matrix, find its maximum value and write a single scalar out:

\`\`\`
row_max[r] = max(matrix[r, :])
\`\`\`

This is the first pass of a numerically-stable softmax. One program handles one row, just like the Block Reductions tutorial.

Steps:

1. Use \`tl.program_id(0)\` to pick a row
2. Load the row — remember: for a **max** reduction, masked-out slots must be \`float('-inf')\`, not \`0.0\`
3. Reduce with \`tl.max(row, axis=0)\`
4. Store the single scalar at \`row_max_ptr + row_idx\``,
        difficulty: 'medium',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def row_max_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols

    # TODO: Load the row with the correct 'other' for a max reduction,
    # reduce with tl.max, and store one scalar per row.
    pass

# --- Driver code ---
n_rows, n_cols = 4, 10
np.random.seed(7)
matrix = np.random.randn(n_rows, n_cols).astype(np.float32)
row_max = np.zeros(n_rows, dtype=np.float32)

row_max_kernel[(n_rows,)](matrix, row_max, n_cols, BLOCK_SIZE=16)

print("Matrix:")
print(matrix.round(3))
print(f"\\nRow maxes: {row_max}")
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def row_max_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols

    row = tl.load(input_ptr + row_start + col_offsets,
                  mask=mask, other=float('-inf'))
    m = tl.max(row, axis=0)
    tl.store(output_ptr + row_idx, m)`,
        testCode: `expected = matrix.max(axis=1)
max_err = float(np.max(np.abs(row_max - expected)))
assert np.allclose(row_max, expected, atol=1e-5), f"Wrong! Got {row_max}, expected {expected}"
print(f"Row maxes: {row_max}")
print(f"Expected:  {expected}")
print(f"Max error: {max_err:.2e}")
print("PASSED")`,
        hints: [
          'Each program handles one row: row_idx = tl.program_id(0) and row_start = row_idx * n_cols.',
          'For max, the masked "other" value must be float("-inf") so invalid lanes never win.',
          'Reduce with m = tl.max(row, axis=0), then tl.store(output_ptr + row_idx, m) — a single scalar per row.',
        ],
      },
    ],
    quiz: {
      id: 'mod-4-quiz',
      questions: [
        {
          id: 'mod-4-q1',
          question:
            'Why do we subtract the maximum before computing exp in softmax?',
          options: [
            'To make the computation faster',
            'To ensure the output sums to 1',
            'To prevent numerical overflow in the exponential',
            'To handle negative inputs',
          ],
          correctIndex: 2,
          explanation:
            'exp(x) can overflow for large x. Subtracting max(x) ensures the largest exponent is exp(0)=1, preventing overflow while not changing the mathematical result.',
        },
        {
          id: 'mod-4-q2',
          question:
            'In a row-wise reduction pattern, what does each program ID correspond to?',
          options: [
            'A column of the matrix',
            'A row of the matrix',
            'A block of consecutive elements',
            'A single element',
          ],
          correctIndex: 1,
          explanation:
            'In the row-per-program pattern, each program_id corresponds to one row. The program loads the entire row, performs reductions across columns, and writes the result.',
        },
        {
          id: 'mod-4-q3',
          question:
            'What is the correct "other" value when loading data for a sum reduction with masking?',
          options: [
            'float("inf")',
            'float("-inf")',
            '0.0',
            '1.0',
          ],
          correctIndex: 2,
          explanation:
            'For sum, masked-out elements should be 0 since adding 0 doesn\'t affect the sum. For max, use -inf. For product, use 1.',
        },
        {
          id: 'mod-4-q4',
          question:
            'After applying softmax to a row, what should the sum of that row be?',
          options: [
            '0.0',
            '1.0',
            'The number of elements',
            'It depends on the input',
          ],
          correctIndex: 1,
          explanation:
            'Softmax converts any real-valued vector into a probability distribution. By definition, the elements sum to 1.0.',
        },
      ],
    },
  },

  // ===================================================================
  // MODULE 5: Matrix Multiplication
  // ===================================================================
  {
    id: 'mod-5',
    title: 'Matrix Multiplication',
    description:
      'Master 2D indexing, write a matrix multiply kernel, and learn about tiling.',
    icon: 'grid-3x3',
    difficulty: 'advanced',
    lessons: [
      {
        id: 'mod-5-lesson-1',
        title: '2D Grids and Indexing',
        content: `## 2D Grids and Indexing

So far we've used 1D grids. For matrix operations, we need **2D grids** where each program handles a 2D block of the output.

### 2D Program IDs

\`\`\`python
row_pid = tl.program_id(0)  # which row block
col_pid = tl.program_id(1)  # which column block
\`\`\`

### 2D Offsets

For an output block at position (row_pid, col_pid):

\`\`\`python
row_offsets = row_pid * BLOCK_M + tl.arange(0, BLOCK_M)  # shape [BLOCK_M]
col_offsets = col_pid * BLOCK_N + tl.arange(0, BLOCK_N)  # shape [BLOCK_N]
\`\`\`

### Row-Major Memory Layout

Matrices in memory are stored row-by-row. To access element [i, j] in a matrix with \`stride\` columns:

\`\`\`
address = base_ptr + i * stride + j
\`\`\`

For a block of elements, we expand this to 2D:

\`\`\`python
# Create 2D offset grid
offsets = row_offsets[:, None] * stride + col_offsets[None, :]
# Shape: [BLOCK_M, BLOCK_N]
\`\`\`

This uses broadcasting: \`[:, None]\` adds a column dimension, \`[None, :]\` adds a row dimension.

Try the example to see 2D indexing!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Transpose a matrix: output[j,i] = input[i,j]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Input offsets [BLOCK_M, BLOCK_N] in row-major
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2D offsets for input (M x N, stride=N)
    in_offsets = row_offs[:, None] * N + col_offs[None, :]
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)

    # Load block from input
    block = tl.load(input_ptr + in_offsets, mask=mask)

    # 2D offsets for output (N x M, stride=M) - rows and cols swapped
    out_offsets = col_offs[:, None] * M + row_offs[None, :]
    out_mask = (col_offs[:, None] < N) & (row_offs[None, :] < M)

    # Store transposed block
    tl.store(output_ptr + out_offsets, tl.trans(block), mask=out_mask)

# ---
M, N = 4, 6
matrix = np.arange(M * N, dtype=np.float32).reshape(M, N)
output = np.zeros((N, M), dtype=np.float32)

BLOCK_M, BLOCK_N = 4, 4
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

transpose_kernel[grid](matrix, output, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

print("Input (4x6):")
print(matrix)
print("\\nTransposed (6x4):")
print(output)
print(f"\\nCorrect: {np.allclose(output, matrix.T)}")
`,
      },
      {
        id: 'mod-5-lesson-2',
        title: 'Matrix Multiplication',
        content: `## Matrix Multiplication in Triton

Matrix multiplication (matmul) is **the** fundamental GPU operation. It computes:

\`\`\`
C[i,j] = sum(A[i,k] * B[k,j]) for all k
\`\`\`

### The Tiled Approach

A naive matmul loads entire rows/columns. Tiled matmul processes the K dimension in chunks:

\`\`\`
For each output block C[m_block, n_block]:
    accumulator = zeros
    For k_block in range(0, K, BLOCK_K):
        a_block = A[m_block, k_block]   # shape [BLOCK_M, BLOCK_K]
        b_block = B[k_block, n_block]   # shape [BLOCK_K, BLOCK_N]
        accumulator += dot(a_block, b_block)
    C[m_block, n_block] = accumulator
\`\`\`

### Why Tiling Matters

Without tiling, each program loads O(K) data. With tiling, data loaded into fast memory (registers/cache) is reused across multiple computations, improving the compute-to-memory ratio.

### The \`tl.dot\` Operation

Triton provides \`tl.dot(a, b)\` for block-level matrix multiplication. This maps to hardware tensor cores on modern GPUs.

### Key Parameters

- **BLOCK_M, BLOCK_N**: Output tile size (rows x cols per program)
- **BLOCK_K**: Reduction tile size (chunk of K processed per iteration)

Run the example to see a complete tiled matmul!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this output block
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_offsets = m_offsets[:, None] * stride_am + k_offsets[None, :]
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N]
        b_offsets = k_offsets[:, None] * stride_bk + n_offsets[None, :]
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    c_offsets = m_offsets[:, None] * N + n_offsets[None, :]
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)

# ---
M, N, K = 8, 6, 10
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

BLOCK_M, BLOCK_N, BLOCK_K = 4, 4, 4
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

matmul_kernel[grid](A, B, C, M, N, K, K, N,
                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

expected = A @ B
print(f"Matrix multiply: ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
print(f"Max error: {np.max(np.abs(C - expected)):.2e}")
print(f"Correct: {np.allclose(C, expected, atol=1e-4)}")
`,
      },
    ],
    puzzles: [
      {
        id: 'mod-5-puzzle-1',
        title: 'Puzzle: Vector Dot Product',
        description: `## Vector Dot Product

Write a kernel that computes the dot product of two vectors:

\`\`\`
result = sum(a[i] * b[i]) for all i
\`\`\`

This is simpler than a full matmul: a single program loads both vectors, multiplies element-wise, and reduces with \`tl.sum\`.

Steps:
1. Load both input vectors with masking
2. Multiply them element-wise
3. Sum the products using \`tl.sum\`
4. Store the scalar result

**Assume the vectors fit in a single block** (n <= BLOCK_SIZE).`,
        difficulty: 'medium',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def dot_kernel(a_ptr, b_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # TODO: Load a and b, compute dot product, store result
    # 1. Load vectors
    # 2. Multiply element-wise
    # 3. Sum all products
    # 4. Store the scalar result
    pass

# --- Driver code ---
n = 16
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
result = np.zeros(1, dtype=np.float32)

dot_kernel[(1,)](a, b, result, n, BLOCK_SIZE=16)

print(f"Dot product: {result[0]:.6f}")
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def dot_kernel(a_ptr, b_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    dot = tl.sum(a * b, axis=0)
    tl.store(output_ptr, dot)`,
        testCode: `expected = np.dot(a, b)
err = abs(result[0] - expected)
assert err < 1e-3, f"Wrong! Got {result[0]:.6f}, expected {expected:.6f}, error {err:.6f}"
print(f"Expected: {expected:.6f}")
print(f"Error:    {err:.2e}")
print("PASSED")`,
        hints: [
          'Load both vectors: a = tl.load(a_ptr + offsets, mask=mask, other=0.0)',
          'Element-wise multiply: products = a * b',
          'Reduce: dot = tl.sum(products, axis=0), then tl.store(output_ptr, dot)',
        ],
      },
      {
        id: 'mod-5-puzzle-2',
        title: 'Puzzle: Outer Product',
        description: `## Outer Product

Given two 1D vectors \`a\` of length \`M\` and \`b\` of length \`N\`, write a kernel that produces the 2D outer product \`C\`:

\`\`\`
C[i, j] = a[i] * b[j]
\`\`\`

A single program can handle the whole output (assume \`M, N <= BLOCK_SIZE\`). This is your first real use of **broadcasting** with \`[:, None]\` and \`[None, :]\` — the same trick used for 2D offsets in Tutorial 1.

Steps:

1. Load \`a\` as a 1D block of shape \`[BLOCK_M]\`, \`b\` as \`[BLOCK_N]\`, both with 1D masks
2. Broadcast-multiply: \`a[:, None] * b[None, :]\` produces a \`[BLOCK_M, BLOCK_N]\` block
3. Build 2D offsets \`row_offs[:, None] * N + col_offs[None, :]\` into the \`M*N\` output
4. Build a 2D mask \`(row_offs[:, None] < M) & (col_offs[None, :] < N)\`
5. Store the block`,
        difficulty: 'medium',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def outer_product_kernel(
    a_ptr, b_ptr, c_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    row_mask = row_offs < M
    col_mask = col_offs < N

    a = tl.load(a_ptr + row_offs, mask=row_mask, other=0.0)
    b = tl.load(b_ptr + col_offs, mask=col_mask, other=0.0)

    # TODO:
    # 1. Broadcast: out = a[:, None] * b[None, :]
    # 2. Compute 2D output offsets into a row-major (M x N) buffer
    # 3. Build a 2D mask and store
    pass

# --- Driver code ---
M, N = 6, 5
np.random.seed(11)
a = np.random.randn(M).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

outer_product_kernel[(1,)](a, b, C, M, N, BLOCK_M=8, BLOCK_N=8)

print(f"a = {a.round(3)}")
print(f"b = {b.round(3)}")
print("C =")
print(C.round(3))
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def outer_product_kernel(
    a_ptr, b_ptr, c_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    row_mask = row_offs < M
    col_mask = col_offs < N

    a = tl.load(a_ptr + row_offs, mask=row_mask, other=0.0)
    b = tl.load(b_ptr + col_offs, mask=col_mask, other=0.0)

    out = a[:, None] * b[None, :]

    c_offsets = row_offs[:, None] * N + col_offs[None, :]
    c_mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    tl.store(c_ptr + c_offsets, out, mask=c_mask)`,
        testCode: `expected = np.outer(a, b)
max_err = float(np.max(np.abs(C - expected)))
assert np.allclose(C, expected, atol=1e-5), f"Wrong! Max error {max_err:.6f}"
print("Expected:")
print(expected.round(3))
print(f"\\nMax error: {max_err:.2e}")
print("PASSED")`,
        hints: [
          'Load a and b as 1D blocks first; their broadcast product a[:, None] * b[None, :] has shape [BLOCK_M, BLOCK_N].',
          'For a row-major (M x N) output, the offset for element [i, j] is row_offs[:, None] * N + col_offs[None, :].',
          'Build the 2D mask with & so both row and column bounds are respected: (row_offs[:, None] < M) & (col_offs[None, :] < N).',
        ],
      },
    ],
    quiz: {
      id: 'mod-5-quiz',
      questions: [
        {
          id: 'mod-5-q1',
          question:
            'Why is matrix multiplication tiled in Triton (processing K in chunks)?',
          options: [
            'To make the code simpler',
            'To maximize data reuse from fast memory (registers/cache)',
            'Because GPUs can only multiply small matrices',
            'To reduce the number of floating point operations',
          ],
          correctIndex: 1,
          explanation:
            'Tiling loads small blocks into fast memory (registers/L1 cache) and reuses them for multiple multiply-accumulate operations. This dramatically improves the compute-to-memory ratio.',
        },
        {
          id: 'mod-5-q2',
          question:
            'In a 2D grid, what do tl.program_id(0) and tl.program_id(1) correspond to?',
          options: [
            'Thread X and Y within a block',
            'The row block and column block of the output',
            'The input and output matrices',
            'The M and K dimensions',
          ],
          correctIndex: 1,
          explanation:
            'In a 2D grid, program_id(0) identifies which row block of the output this program computes, and program_id(1) identifies which column block.',
        },
        {
          id: 'mod-5-q3',
          question:
            'What is the purpose of tl.dot(a, b) in a matmul kernel?',
          options: [
            'It computes a vector dot product',
            'It performs block-level matrix multiplication on tiles',
            'It computes the element-wise product',
            'It transposes the matrix',
          ],
          correctIndex: 1,
          explanation:
            'tl.dot(a, b) performs matrix multiplication on 2D block tiles, mapping to hardware tensor cores for maximum performance.',
        },
      ],
    },
  },

  // ===================================================================
  // MODULE 6: Advanced Kernels
  // ===================================================================
  {
    id: 'mod-6',
    title: 'Advanced Kernels',
    description:
      'Build production patterns: layer normalization, attention scores, and autotuning.',
    icon: 'brain',
    difficulty: 'advanced',
    lessons: [
      {
        id: 'mod-6-lesson-1',
        title: 'Layer Normalization',
        content: `## Layer Normalization

Layer normalization is used in every transformer layer. It normalizes each token independently:

\`\`\`
y = gamma * (x - mean) / sqrt(variance + eps) + beta
\`\`\`

Where gamma and beta are learnable parameters.

### Why Write This in Triton?

PyTorch's LayerNorm is multiple separate operations:
1. Compute mean
2. Compute variance
3. Normalize
4. Scale by gamma
5. Add beta

That's 5 kernel launches and 5 round-trips to memory. In Triton, it's **one kernel**, **one memory read**, **one memory write**.

### The Pattern

\`\`\`python
@triton.jit
def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) ** 2, axis=0) / N
    x_norm = (x - mean) / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)

    out = gamma * x_norm + beta
    tl.store(out_ptr + row * N + offs, out, mask=mask)
\`\`\`

Each program handles one row (one token in a sequence).

Run the example to see a complete LayerNorm!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row and parameters
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)

    # Normalize
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    x_norm = diff / tl.sqrt(var + eps)

    # Scale and shift
    out = gamma * x_norm + beta
    tl.store(out_ptr + row * N + offs, out, mask=mask)

# ---
batch, hidden = 4, 16
x = np.random.randn(batch, hidden).astype(np.float32) * 3 + 1
gamma = np.ones(hidden, dtype=np.float32)
beta = np.zeros(hidden, dtype=np.float32)
output = np.zeros_like(x)
eps = 1e-5

layernorm_kernel[(batch,)](x, gamma, beta, output, hidden, eps, BLOCK_SIZE=32)

# Verify
mean = x.mean(axis=1, keepdims=True)
var = x.var(axis=1, keepdims=True)
expected = (x - mean) / np.sqrt(var + eps) * gamma + beta

print(f"LayerNorm on ({batch} x {hidden})")
print(f"Output means (should be ~0): {output.mean(axis=1).round(6)}")
print(f"Output stds  (should be ~1): {output.std(axis=1).round(4)}")
print(f"Max error: {np.max(np.abs(output - expected)):.2e}")
`,
      },
      {
        id: 'mod-6-lesson-2',
        title: 'Online Softmax & Flash Attention',
        content: `## Online Softmax & Flash Attention

The naive approach to attention computes **all** scores first, applies softmax, then multiplies by V. This requires storing the full N\u00d7N score matrix \u2014 O(n\u00b2) memory.

**Online softmax** (the algorithm behind Flash Attention) processes keys one at a time, maintaining a running softmax. This uses O(1) extra memory per query.

### The Problem with Standard Softmax

Standard softmax needs two passes over the data:
1. Find the max (for numerical stability)
2. Compute exp(x - max) / sum(exp(x - max))

You can't start computing output until you've seen every score. That means materializing the full score matrix.

### The Online Softmax Trick

Online softmax does it in **one pass** by correcting previous results when a new maximum is found:

\`\`\`
For each new key k with value v:
    score = dot(q, k) / sqrt(d)
    m_new = max(m_old, score)
    alpha = exp(m_old - m_new)    # correction factor (\u2264 1)
    p     = exp(score - m_new)    # new probability
    l     = l * alpha + p         # rescale running sum
    acc   = acc * alpha + p * v   # rescale running accumulator
\`\`\`

When the max doesn't change: \`alpha = 1\`, nothing gets rescaled.
When the max increases: \`alpha < 1\`, all previous contributions shrink correctly.

### Why This Matters

- **Flash Attention** uses exactly this algorithm (on blocks of keys, not individual keys)
- O(1) extra memory per query instead of O(n)
- Enables attention on sequences with millions of tokens
- This is how **vLLM**, Flash Attention, and all modern inference engines work

The example implements full attention (Q\u00d7K\u00d7V \u2192 output) with online softmax. Compare: our Module 4 softmax only computed weights, this computes the final output in one fused pass.`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def online_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len, d_k, sm_scale,
    BLOCK_D: tl.constexpr,
):
    """Full attention with online softmax - the core of Flash Attention."""
    q_idx = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d_k

    q = tl.load(q_ptr + q_idx * d_k + d_offs, mask=d_mask, other=0.0)

    # Online softmax state
    m_i = -1e10                                     # running max score
    l_i = 0.0                                       # running sum of exp
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)    # output accumulator

    for k_idx in range(seq_len):
        k = tl.load(k_ptr + k_idx * d_k + d_offs, mask=d_mask, other=0.0)
        v = tl.load(v_ptr + k_idx * d_k + d_offs, mask=d_mask, other=0.0)

        # Attention score
        score = tl.sum(q * k, axis=0) * sm_scale

        # Online softmax update
        m_new = tl.maximum(m_i, score)
        alpha = tl.exp(m_i - m_new)    # correction for previous values
        p = tl.exp(score - m_new)       # probability for this key

        l_i = l_i * alpha + p           # rescaled running sum
        acc = acc * alpha + p * v       # rescaled weighted values
        m_i = m_new

    # Final normalization
    out = acc / l_i
    tl.store(out_ptr + q_idx * d_k + d_offs, out, mask=d_mask)

# ---
seq_len, d_k = 6, 8
np.random.seed(42)
Q = np.random.randn(seq_len, d_k).astype(np.float32)
K = np.random.randn(seq_len, d_k).astype(np.float32)
V = np.random.randn(seq_len, d_k).astype(np.float32)
output = np.zeros_like(Q)
scale = 1.0 / np.sqrt(d_k)

online_attention_kernel[(seq_len,)](Q, K, V, output, seq_len, d_k, scale, BLOCK_D=16)

# Verify against standard two-pass attention
scores = Q @ K.T * scale
scores_max = scores.max(axis=1, keepdims=True)
attn = np.exp(scores - scores_max)
attn = attn / attn.sum(axis=1, keepdims=True)
expected = attn @ V

print("Online softmax attention (Flash Attention core)")
print(f"Max error vs standard attention: {np.max(np.abs(output - expected)):.2e}")
print(f"Correct: {np.allclose(output, expected, atol=1e-5)}")
print(f"\\nOutput row 0:   {output[0].round(4)}")
print(f"Expected row 0: {expected[0].round(4)}")
`,
      },
      {
        id: 'mod-6-lesson-3',
        title: 'Autotuning',
        content: `## Autotuning

Different GPUs and data sizes perform best with different block sizes. **Autotuning** tries multiple configurations and picks the fastest one.

### triton.autotune

In real Triton, you decorate a kernel with \`@triton.autotune\`:

\`\`\`python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
\`\`\`

### How It Works

1. On first call, Triton benchmarks each config
2. Results are cached for the given problem size
3. Subsequent calls use the fastest config automatically

### Config Parameters

Common parameters to tune:
- **BLOCK_M, BLOCK_N, BLOCK_K**: Tile sizes
- **num_warps**: How many warps per program (GPU-specific)
- **num_stages**: Pipeline depth for memory loading

### Best Practices

- Start with powers of 2: 32, 64, 128, 256
- Larger blocks = more parallelism but more register pressure
- \`key\` parameter specifies which arguments affect performance (usually matrix dimensions)

In our simulator, autotune just uses the first config. On real GPUs, it's one of Triton's most powerful features.`,
        code: `import triton
import triton.language as tl
import numpy as np

# In real Triton, these configs would be benchmarked
configs = [
    {"BLOCK_SIZE": 64},
    {"BLOCK_SIZE": 128},
    {"BLOCK_SIZE": 256},
]

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# Simulate autotuning by trying different block sizes
n = 32
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)

print("Simulated autotuning results:")
print(f"{'Config':<20} {'Grid Size':<12} {'Status':<10}")
print("-" * 42)

for config in configs:
    bs = config["BLOCK_SIZE"]
    out = np.zeros(n, dtype=np.float32)
    grid = (triton.cdiv(n, bs),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=bs)
    correct = np.allclose(out, x + y)
    print(f"BLOCK_SIZE={bs:<8} {grid[0]:<12} {'PASS' if correct else 'FAIL'}")

print(f"\\nIn real Triton, the fastest config would be selected automatically.")
print(f"Key factors: GPU architecture, data size, memory bandwidth.")
`,
      },
    ],
    puzzles: [
      {
        id: 'mod-6-puzzle-1',
        title: 'Puzzle: Fused Softmax + Scale',
        description: `## Fused Softmax with Temperature Scaling

Write a kernel that computes temperature-scaled softmax on each row:

\`\`\`
output[i,:] = softmax(input[i,:] / temperature)
\`\`\`

Where softmax is:
\`\`\`
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
\`\`\`

Steps:
1. Load the row
2. Divide by temperature
3. Compute numerically stable softmax
4. Store the result

**Temperature < 1** makes the distribution sharper (more confident).
**Temperature > 1** makes it softer (more uniform).`,
        difficulty: 'medium',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def scaled_softmax_kernel(input_ptr, output_ptr, temperature, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols

    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))

    # TODO: Apply temperature scaling and softmax
    # 1. Divide by temperature
    # 2. Find max for numerical stability
    # 3. Compute exp(x - max)
    # 4. Sum and normalize
    result = x  # FIX THIS

    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

# --- Driver code ---
n_rows, n_cols = 4, 8
x = np.random.randn(n_rows, n_cols).astype(np.float32)
output = np.zeros_like(x)
temperature = 0.5  # Sharp distribution

scaled_softmax_kernel[(n_rows,)](x, output, temperature, n_cols, BLOCK_SIZE=16)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def scaled_softmax_kernel(input_ptr, output_ptr, temperature, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols

    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))

    x = x / temperature
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum

    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)`,
        testCode: `# Verify
x_scaled = x / temperature
e = np.exp(x_scaled - x_scaled.max(axis=1, keepdims=True))
expected = e / e.sum(axis=1, keepdims=True)
max_err = np.max(np.abs(output - expected))
assert max_err < 1e-5, f"Wrong! Max error: {max_err:.6f}"
print(f"Temperature: {temperature}")
print(f"Row sums (should be 1.0): {output.sum(axis=1)}")
print(f"Max error: {max_err:.2e}")
print("PASSED")`,
        hints: [
          'First divide x by temperature: x = x / temperature',
          'Then apply standard numerically stable softmax: x_max = tl.max(x, axis=0)',
          'Full sequence: x/temp -> max -> exp(x-max) -> sum -> divide',
        ],
      },
      {
        id: 'mod-6-puzzle-2',
        title: 'Puzzle: Fused Add + LayerNorm',
        description: `## Fused Residual Add + Layer Normalization

In transformers, a common pattern is:

\`\`\`
output = LayerNorm(x + residual)
\`\`\`

Write a kernel that fuses the residual addition with layer normalization:

1. Load both x and residual
2. Add them: \`h = x + residual\`
3. Compute mean and variance of h
4. Normalize: \`(h - mean) / sqrt(var + eps)\`

Assume gamma=1 and beta=0 (no learnable parameters) for simplicity.`,
        difficulty: 'hard',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def fused_add_layernorm_kernel(x_ptr, residual_ptr, output_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row_start = row * n_cols

    # TODO: Implement fused add + layernorm
    # 1. Load x and residual
    # 2. Add them
    # 3. Compute mean
    # 4. Compute variance
    # 5. Normalize
    pass

# --- Driver code ---
n_rows, n_cols = 4, 16
x = np.random.randn(n_rows, n_cols).astype(np.float32)
residual = np.random.randn(n_rows, n_cols).astype(np.float32)
output = np.zeros_like(x)
eps = 1e-5

fused_add_layernorm_kernel[(n_rows,)](x, residual, output, n_cols, eps, BLOCK_SIZE=32)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def fused_add_layernorm_kernel(x_ptr, residual_ptr, output_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row_start = row * n_cols

    x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
    res = tl.load(residual_ptr + row_start + offs, mask=mask, other=0.0)

    h = x + res
    mean = tl.sum(h, axis=0) / n_cols
    diff = h - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    result = diff / tl.sqrt(var + eps)

    tl.store(output_ptr + row_start + offs, result, mask=mask)`,
        testCode: `# Verify
h = x + residual
mean = h.mean(axis=1, keepdims=True)
var = h.var(axis=1, keepdims=True)
expected = (h - mean) / np.sqrt(var + eps)
max_err = np.max(np.abs(output - expected))
assert max_err < 1e-4, f"Wrong! Max error: {max_err}"
print(f"Output means (~0): {output.mean(axis=1).round(6)}")
print(f"Output stds  (~1): {output.std(axis=1).round(4)}")
print(f"Max error: {max_err:.2e}")
print("PASSED")`,
        hints: [
          'Load both arrays: x = tl.load(...), res = tl.load(...), then h = x + res',
          'Compute mean: tl.sum(h, axis=0) / n_cols',
          'Variance: tl.sum((h - mean)^2, axis=0) / n_cols. Normalize: (h - mean) / tl.sqrt(var + eps)',
        ],
      },
    ],
    quiz: {
      id: 'mod-6-quiz',
      questions: [
        {
          id: 'mod-6-q1',
          question:
            'Why is a fused LayerNorm kernel faster than separate PyTorch operations?',
          options: [
            'It uses fewer floating point operations',
            'It reads/writes data once instead of multiple times',
            'It uses more GPU threads',
            'LayerNorm is inherently slow in PyTorch',
          ],
          correctIndex: 1,
          explanation:
            'The fused kernel loads each row once, computes mean/var/normalize in registers, and writes once. Separate ops would read/write intermediate results to memory multiple times.',
        },
        {
          id: 'mod-6-q2',
          question:
            'What does temperature scaling do in softmax?',
          options: [
            'It normalizes the input to unit variance',
            'It controls how sharp (peaked) vs uniform the output distribution is',
            'It prevents numerical overflow',
            'It scales the output to a specific range',
          ],
          correctIndex: 1,
          explanation:
            'Temperature < 1 makes softmax sharper (more confident, highest value dominates). Temperature > 1 makes it softer (more uniform). Temperature = 1 is standard softmax.',
        },
        {
          id: 'mod-6-q3',
          question:
            'In real Triton, what does @triton.autotune do?',
          options: [
            'Automatically optimizes the algorithm',
            'Benchmarks multiple configurations and selects the fastest',
            'Reduces precision for faster execution',
            'Automatically parallelizes the kernel',
          ],
          correctIndex: 1,
          explanation:
            'Autotune runs the kernel with each provided Config (different BLOCK sizes, num_warps, etc.), benchmarks them, and caches the fastest one for each problem size.',
        },
        {
          id: 'mod-6-q4',
          question:
            'In a transformer, why is "residual add + layernorm" a good candidate for fusion?',
          options: [
            'Because it happens only once per layer',
            'Because both operations are applied to the same data row-wise with no cross-row dependencies',
            'Because layernorm is too slow otherwise',
            'Because residual connections are optional',
          ],
          correctIndex: 1,
          explanation:
            'Both the addition and layernorm operate on the same data independently per row. Fusing them eliminates the intermediate write (x+residual) and re-read, saving memory bandwidth.',
        },
      ],
    },
  },

  // ===================================================================
  // MODULE 7: Quantization
  // ===================================================================
  {
    id: 'mod-7',
    title: 'Quantization',
    description:
      'Shrink model weights from 16-bit to 4-bit. Learn bit packing, group-wise dequantization, and write a real inference kernel.',
    icon: 'archive',
    difficulty: 'advanced',
    lessons: [
      {
        id: 'mod-7-lesson-1',
        title: 'Why Quantization Matters',
        content: `## Why Quantization Matters

Large language models are memory-bound. A 70B parameter model in FP16 needs **140 GB** just to store the weights \u2014 that doesn't fit on a single GPU.

### The Math

| Precision | Bytes/param | 7B model | 70B model |
|-----------|-------------|----------|-----------|
| FP32      | 4           | 28 GB    | 280 GB    |
| FP16      | 2           | 14 GB    | 140 GB    |
| INT8      | 1           | 7 GB     | 70 GB     |
| **INT4**  | **0.5**     | **3.5 GB** | **35 GB** |

4-bit quantization gives **4x compression** over FP16. A 70B model fits on a single 48GB GPU.

### How It Works

**Quantize** (offline, once): convert FP16 weights to 4-bit integers.

\`\`\`
quantized = round((float_value / scale) + zero_point)
\`\`\`

**Dequantize** (at inference, in the kernel): convert back to float for computation.

\`\`\`
float_value = (quantized - zero_point) * scale
\`\`\`

### Group-wise Quantization

Using one scale for an entire weight matrix is too coarse. Instead, weights are split into **groups** (e.g., 128 values), each with its own scale and zero point. This preserves accuracy while still compressing 4x.

### Bit Packing

A 4-bit value only needs 4 bits, but the smallest addressable unit is 8 bits. So we **pack** 8 values into a single 32-bit integer:

\`\`\`
int32: [val7][val6][val5][val4][val3][val2][val1][val0]
        28-31 24-27 20-23 16-19 12-15  8-11  4-7   0-3
\`\`\`

To extract value \`i\`: \`(packed >> (i * 4)) & 0xF\`

Run the example to see packing and unpacking in action!`,
        code: `import numpy as np

# === Bit Packing Demo ===
# Pack 8 values (each 0-15) into one int32
values = np.array([3, 7, 1, 15, 0, 9, 5, 12], dtype=np.int32)

packed = 0
for i, v in enumerate(values):
    packed |= int(v) << (i * 4)

print(f"Original 8 values: {values}")
print(f"Packed into int32: 0x{packed:08X} ({packed})")
print()

# Unpack them back
print("Unpacking with (packed >> (i*4)) & 0xF:")
for i in range(8):
    extracted = (packed >> (i * 4)) & 0xF
    print(f"  Slot {i}: shift >> {i*4:2d}, mask 0xF -> {extracted}  {'OK' if extracted == values[i] else 'WRONG'}")

# === Group-wise Dequantization Demo ===
print("\\n=== Group-wise Dequantization ===")
group_size = 4
scales = np.array([0.1, 0.2], dtype=np.float32)   # 2 groups
zeros = np.array([8.0, 8.0], dtype=np.float32)     # zero points

print(f"Quantized values: {values}")
print(f"Group size: {group_size}, Scales: {scales}, Zeros: {zeros}")
print()

for i, qval in enumerate(values):
    group = i // group_size
    dequant = (float(qval) - zeros[group]) * scales[group]
    print(f"  val[{i}]={qval:2d}  group={group}  -> ({qval} - {zeros[group]}) * {scales[group]} = {dequant:+.1f}")
`,
      },
      {
        id: 'mod-7-lesson-2',
        title: 'Dequantization Kernel',
        content: `## Dequantization Kernel

Now let's write a Triton kernel that dequantizes packed 4-bit weights. This is exactly what runs inside vLLM, llama.cpp, and other inference engines during every forward pass.

### The Kernel's Job

Input:
- \`packed_weights\`: array of int32, each holding 8 x 4-bit values
- \`scales\`: per-group scale factors
- \`zeros\`: per-group zero points
- \`group_size\`: how many values share one scale/zero

Output:
- \`output\`: array of float32, 8x longer than packed_weights

### The Algorithm

Each program handles a block of packed int32 values. For each packed value, it extracts all 8 nibbles and dequantizes them:

\`\`\`python
for bit in range(8):
    quantized = (packed >> (bit * 4)) & 0xF
    group = output_column // group_size
    result = (quantized - zeros[group]) * scales[group]
\`\`\`

### Key Patterns

- **Bit manipulation** in Triton: \`>>\` (shift) and \`&\` (mask) work on integer tensors
- **Expansion**: 1 packed int32 becomes 8 float32 values
- **Group indexing**: computing which group each output element belongs to

This pattern appears in AWQ, GPTQ, and other quantization schemes used in production LLM inference (e.g., vLLM).

Run the example to see it work!`,
        code: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def dequantize_kernel(
    packed_ptr, scales_ptr, zeros_ptr, output_ptr,
    num_packed, group_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize 4-bit packed weights to float32."""
    pid = tl.program_id(0)
    packed_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = packed_offs < num_packed

    # Load packed int32 values (each contains 8 x 4-bit weights)
    packed = tl.load(packed_ptr + packed_offs, mask=mask, other=0)

    # Unpack each of the 8 nibbles
    for bit in range(8):
        # Extract 4-bit value: shift right and mask
        quantized = (packed >> (bit * 4)) & 0xF

        # Output position for this nibble
        out_col = packed_offs * 8 + bit

        # Group-wise dequantization
        group_idx = out_col // group_size
        scale = tl.load(scales_ptr + group_idx, mask=mask, other=1.0)
        zero = tl.load(zeros_ptr + group_idx, mask=mask, other=0.0)

        # Dequantize: float_val = (int4_val - zero) * scale
        dequantized = (quantized.astype(np.float32) - zero) * scale
        tl.store(output_ptr + out_col, dequantized, mask=mask)

# --- Driver code ---
np.random.seed(42)

# Create original float weights, then quantize them
num_values = 32
group_size = 8
num_groups = num_values // group_size

# Random weights in [-1, 1]
original = np.random.randn(num_values).astype(np.float32) * 0.5

# Quantize per group
scales = np.zeros(num_groups, dtype=np.float32)
zeros = np.full(num_groups, 8.0, dtype=np.float32)  # mid-point zero
quantized = np.zeros(num_values, dtype=np.int32)

for g in range(num_groups):
    sl = slice(g * group_size, (g + 1) * group_size)
    group_vals = original[sl]
    vmin, vmax = group_vals.min(), group_vals.max()
    scales[g] = (vmax - vmin) / 15.0  # 4-bit range is 0-15
    quantized[sl] = np.clip(np.round((group_vals - vmin) / scales[g]), 0, 15).astype(np.int32)
    zeros[g] = -vmin / scales[g]  # zero point

# Pack 8 values into each int32
num_packed = num_values // 8
packed = np.zeros(num_packed, dtype=np.int32)
for i in range(num_packed):
    for bit in range(8):
        packed[i] |= int(quantized[i * 8 + bit]) << (bit * 4)

print(f"Original weights ({num_values} values, {num_groups} groups, group_size={group_size})")
print(f"Packed into {num_packed} int32 values\\n")

# Dequantize with Triton kernel
output = np.zeros(num_values, dtype=np.float32)
grid = (triton.cdiv(num_packed, 4),)
dequantize_kernel[grid](packed, scales, zeros, output, num_packed, group_size, BLOCK_SIZE=4)

# Expected: manual dequant
expected = np.zeros(num_values, dtype=np.float32)
for i in range(num_values):
    expected[i] = (quantized[i] - zeros[i // group_size]) * scales[i // group_size]

print(f"Original:     {original[:8].round(3)}")
print(f"Dequantized:  {output[:8].round(3)}")
print(f"Expected:     {expected[:8].round(3)}")
print(f"\\nMax quant error vs original: {np.max(np.abs(output - original)):.4f}")
print(f"Kernel correct: {np.allclose(output, expected, atol=1e-5)}")
`,
      },
    ],
    puzzles: [
      {
        id: 'mod-7-puzzle-1',
        title: 'Puzzle: 4-bit Dequantize',
        description: `## 4-bit Dequantize

Write a kernel that dequantizes packed 4-bit weights.

You receive:
- \`packed_ptr\`: int32 array where each element holds 8 x 4-bit values
- \`scales_ptr\`: float32 per-group scales
- \`zeros_ptr\`: float32 per-group zero points
- \`group_size\`: number of values per group

For each packed int32, you need to:
1. Extract each of the 8 nibbles: \`(packed >> (bit * 4)) & 0xF\`
2. Compute the output column: \`packed_offset * 8 + bit\`
3. Find the group: \`output_col // group_size\`
4. Dequantize: \`(quantized_float - zero) * scale\`
5. Store the result`,
        difficulty: 'hard',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def dequant_kernel(packed_ptr, scales_ptr, zeros_ptr, output_ptr, num_packed, group_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_packed

    packed = tl.load(packed_ptr + offs, mask=mask, other=0)

    # TODO: loop over 8 bits, extract, dequantize, store
    for bit in range(8):
        # 1. Extract 4-bit value
        # 2. Compute output position
        # 3. Find group index
        # 4. Load scale and zero
        # 5. Dequantize and store
        pass

# --- Driver code (do not modify) ---
np.random.seed(0)
num_values = 16
group_size = 8
quantized = np.random.randint(0, 16, size=num_values, dtype=np.int32)
scales = np.array([0.1, 0.2], dtype=np.float32)
zeros = np.array([8.0, 7.0], dtype=np.float32)

num_packed = num_values // 8
packed = np.zeros(num_packed, dtype=np.int32)
for i in range(num_packed):
    for b in range(8):
        packed[i] |= int(quantized[i * 8 + b]) << (b * 4)

output = np.zeros(num_values, dtype=np.float32)
dequant_kernel[(triton.cdiv(num_packed, 4),)](packed, scales, zeros, output, num_packed, group_size, BLOCK_SIZE=4)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def dequant_kernel(packed_ptr, scales_ptr, zeros_ptr, output_ptr, num_packed, group_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_packed

    packed = tl.load(packed_ptr + offs, mask=mask, other=0)

    for bit in range(8):
        w = (packed >> (bit * 4)) & 0xF
        out_col = offs * 8 + bit
        group_idx = out_col // group_size
        scale = tl.load(scales_ptr + group_idx, mask=mask, other=1.0)
        zero = tl.load(zeros_ptr + group_idx, mask=mask, other=0.0)
        result = (w.astype(np.float32) - zero) * scale
        tl.store(output_ptr + out_col, result, mask=mask)`,
        testCode: `# Verify
expected = np.zeros(num_values, dtype=np.float32)
for i in range(num_values):
    expected[i] = (float(quantized[i]) - zeros[i // group_size]) * scales[i // group_size]
max_err = np.max(np.abs(output - expected))
assert max_err < 1e-5, f"Wrong! Max error: {max_err}"
print(f"Quantized: {quantized}")
print(f"Output:    {output.round(3)}")
print(f"Expected:  {expected.round(3)}")
print(f"Max error: {max_err:.2e}")
print("PASSED")`,
        hints: [
          'Extract the 4-bit value: w = (packed >> (bit * 4)) & 0xF',
          'Output position: out_col = offs * 8 + bit. Group: group_idx = out_col // group_size',
          'Dequantize: result = (w.astype(np.float32) - zero) * scale, then tl.store to output_ptr + out_col',
        ],
      },
      {
        id: 'mod-7-puzzle-2',
        title: 'Puzzle: 4-bit Quantize',
        description: `## 4-bit Quantize

This is the inverse of the dequantize puzzle. You're given a float32 vector plus per-group \`scales\` and \`zeros\`, and you must **pack** 8 quantized nibbles into each int32.

For each group of \`group_size\` contiguous values:

\`\`\`
q = clip(round(x / scale + zero), 0, 15)    # a 4-bit integer
\`\`\`

Then, for each packed output slot covering 8 consecutive quantized values, OR the nibbles into place:

\`\`\`
packed[i] = q[8i+0] | (q[8i+1] << 4) | (q[8i+2] << 8) | ... | (q[8i+7] << 28)
\`\`\`

Each program handles one packed int32 (8 input floats). Loop over \`bit in range(8)\`, load one scalar at a time, quantize it, shift it into the accumulator.

Assume \`group_size\` is a multiple of 8 so every packed slot lives inside a single group.`,
        difficulty: 'hard',
        starterCode: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def quant_kernel(x_ptr, scales_ptr, zeros_ptr, packed_ptr,
                  num_packed, group_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_packed

    # Per-packed-slot group index (group_size is a multiple of 8)
    group_idx = (offs * 8) // group_size
    scale = tl.load(scales_ptr + group_idx, mask=mask, other=1.0)
    zero = tl.load(zeros_ptr + group_idx, mask=mask, other=0.0)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # TODO: for bit in range(8):
    #   1. Load x at position offs * 8 + bit
    #   2. Quantize: q = clip(round(x / scale + zero), 0, 15), as int32
    #   3. OR into acc:  acc = acc | (q << (bit * 4))
    for bit in range(8):
        pass

    tl.store(packed_ptr + offs, acc, mask=mask)

# --- Driver code (do not modify) ---
np.random.seed(0)
num_values = 16
group_size = 8
x = np.random.randn(num_values).astype(np.float32) * 0.5

# Per-group scale / zero so quantization is well-defined
num_groups = num_values // group_size
scales = np.zeros(num_groups, dtype=np.float32)
zeros = np.zeros(num_groups, dtype=np.float32)
for g in range(num_groups):
    sl = slice(g * group_size, (g + 1) * group_size)
    vmin, vmax = float(x[sl].min()), float(x[sl].max())
    scales[g] = max((vmax - vmin) / 15.0, 1e-6)
    zeros[g] = -vmin / scales[g]

num_packed = num_values // 8
packed = np.zeros(num_packed, dtype=np.int32)
quant_kernel[(triton.cdiv(num_packed, 4),)](
    x, scales, zeros, packed, num_packed, group_size, BLOCK_SIZE=4,
)
`,
        solution: `import triton
import triton.language as tl
import numpy as np

@triton.jit
def quant_kernel(x_ptr, scales_ptr, zeros_ptr, packed_ptr,
                  num_packed, group_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_packed

    group_idx = (offs * 8) // group_size
    scale = tl.load(scales_ptr + group_idx, mask=mask, other=1.0)
    zero = tl.load(zeros_ptr + group_idx, mask=mask, other=0.0)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for bit in range(8):
        x_val = tl.load(x_ptr + offs * 8 + bit, mask=mask, other=0.0)
        q_float = np.round(x_val / scale + zero)
        q_clipped = tl.minimum(tl.maximum(q_float, 0.0), 15.0)
        q = q_clipped.astype(np.int32)
        acc = acc | (q << (bit * 4))

    tl.store(packed_ptr + offs, acc, mask=mask)`,
        testCode: `# Expected packing, computed in numpy
expected_q = np.zeros(num_values, dtype=np.int32)
for i in range(num_values):
    g = i // group_size
    expected_q[i] = int(np.clip(np.round(x[i] / scales[g] + zeros[g]), 0, 15))

expected_packed = np.zeros(num_packed, dtype=np.int32)
for i in range(num_packed):
    for b in range(8):
        expected_packed[i] |= int(expected_q[i * 8 + b]) << (b * 4)

assert np.array_equal(packed, expected_packed), (
    f"Wrong packing!\\n got:      {packed}\\n expected: {expected_packed}"
)
print(f"Input (first 8):     {x[:8].round(3)}")
print(f"Quantized (first 8): {expected_q[:8]}")
print(f"Packed:              {packed}")
print(f"Expected:            {expected_packed}")
print("PASSED")`,
        hints: [
          'Use the exact same group indexing as the dequant puzzle, just inverted: the group of packed slot i covers x[i*8 : i*8 + 8].',
          'Per bit: load one scalar x_val at position offs * 8 + bit, compute np.round(x_val / scale + zero), clip to [0, 15] with tl.maximum/tl.minimum, then cast with .astype(np.int32).',
          'Accumulate with bitwise OR and a left shift: acc = acc | (q << (bit * 4)); store acc once at the end.',
        ],
      },
    ],
    quiz: {
      id: 'mod-7-quiz',
      questions: [
        {
          id: 'mod-7-q1',
          question:
            'How many 4-bit values can be packed into a single int32?',
          options: ['4', '8', '16', '32'],
          correctIndex: 1,
          explanation:
            'An int32 has 32 bits. Each 4-bit value takes 4 bits, so 32/4 = 8 values fit in one int32.',
        },
        {
          id: 'mod-7-q2',
          question:
            'To extract the 3rd nibble (index 2) from a packed int32, what expression do you use?',
          options: [
            '(packed >> 2) & 0xF',
            '(packed >> 8) & 0xF',
            '(packed >> 6) & 0xF',
            '(packed & 0xF) >> 8',
          ],
          correctIndex: 1,
          explanation:
            'Nibble at index i starts at bit position i*4. For index 2: shift right by 2*4=8 bits, then mask with 0xF (binary 1111) to isolate the 4-bit value.',
        },
        {
          id: 'mod-7-q3',
          question:
            'Why is group-wise quantization used instead of per-tensor quantization?',
          options: [
            'It is faster to compute',
            'It uses less memory',
            'It preserves accuracy by allowing different scale/zero per group of weights',
            'It is required by GPU hardware',
          ],
          correctIndex: 2,
          explanation:
            'Different regions of a weight matrix can have very different value ranges. Per-group scale/zero tracks these local ranges, reducing quantization error compared to a single global scale.',
        },
        {
          id: 'mod-7-q4',
          question:
            'In a dequantization kernel, what is the main computational pattern?',
          options: [
            'Matrix multiplication followed by reduction',
            'Bit extraction with shifts/masks, then linear transform (subtract zero, multiply scale)',
            'Sorting values and interpolating',
            'Repeated softmax applications',
          ],
          correctIndex: 1,
          explanation:
            'Dequantization extracts packed integer values using bit shifts and masks, then applies the linear transform: float = (int_value - zero_point) * scale.',
        },
      ],
    },
  },
];
