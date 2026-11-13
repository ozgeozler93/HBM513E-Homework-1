# HBM513E-Homework-1

# High-Performance Matrix Multiplication Optimizations (DGEMM)

**Course:** HBM 513E – Parallel and Distributed Computing  
**Institution:** Istanbul Technical University (ITU)  
**Instructor:** Prof. Dr. M. Serdar Çelebi  

---

## 📌 Project Overview

Matrix multiplication ($C = A \times B$) is one of the most fundamental computational kernels in scientific computing, numerical linear algebra, and machine learning. While its naive implementation has a straightforward $\mathcal{O}(N^3)$ time complexity, achieving near-peak hardware performance requires careful attention to **memory hierarchy**, **cache locality**, **instruction-level parallelism**, and **SIMD vectorization**.

This repository contains the source code, numerical verification suite, visualization scripts, and formal LaTeX report exploring systematic single-core optimizations for Double-Precision General Matrix Multiplication (**DGEMM**) and comparing them against industry-standard **OpenBLAS** (`cblas_dgemm`).

---

## 📂 Repository Structure

```text
├── code/
│   ├── opt_bench_5000.c      # Core benchmarking suite testing loop permutations & cache blocking
│   ├── verify.c              # Numerical correctness validation against cblas_dgemm (L2-norm check)
│   └── grafik_ciz.py         # Python (Matplotlib) script for generating performance & speedup plots
├── report/
│   ├── hmw1_temp3.tex        # Formal IEEE/academic style LaTeX report source
│   ├── hmw1_temp3.pdf        # Compiled technical report
│   ├── fig1_loop_orderings.png
│   ├── fig2_optimization_steps.png
│   └── fig3_speedup_n5000.png
└── README.md
```

---

## 🧠 Key Optimization Concepts Covered

### 1. Memory Layout & Loop Permutations (Spatial Locality)
In C, matrices are stored in **row-major order**, meaning contiguous elements in a row reside in adjacent memory addresses. Different loop nest permutations drastically change memory access patterns:
* **`IJK` / `JIK` (Naive):** Accesses matrix $B$ column-wise, causing severe cache line misses (stride-$N$ access).
* **`IKJ` / `KIJ` (Cache-Friendly):** Iterates over rows of $B$ and $C$ sequentially in the innermost loop, maximizing **spatial locality** and enabling automatic compiler SIMD vectorization.

### 2. Cache Blocking / Loop Tiling (Temporal Locality)
For large matrix sizes (e.g., $N = 5000$), entire rows or matrices exceed the capacity of L1/L2 CPU caches. **Loop tiling** partitions matrices into smaller sub-blocks of size $B_{\text{size}} \times B_{\text{size}}$ that fit entirely within fast L1/L2 caches, maximizing data reuse (**temporal locality**) before eviction.

### 3. Compiler & Vectorization Flags
* **`-O3`:** Enables aggressive loop unrolling, inlining, and auto-vectorization.
* **`-march=native` / `-ffast-math`:** Targets native AVX2/AVX-512 vector registers and fused multiply-add (FMA) instructions.

### 4. BLAS Baseline (`cblas_dgemm`)
All custom implementations are benchmarked against **OpenBLAS / Apple Accelerate** `cblas_dgemm`, which utilizes hand-tuned assembly micro-kernels, register blocking, and multi-level cache packing.

---

## ✅ Numerical Correctness & Verification (`verify.c`)

Before measuring performance, every optimization must guarantee mathematical accuracy. Using `verify.c`, custom implementations are verified against `cblas_dgemm` for $N = 1000$ by computing the L2-norm of the difference matrix:

$$\|E\|_2 = \|C_{\text{custom}} - C_{\text{BLAS}}\|_2 = \sqrt{\sum_{i,j} (C_{\text{custom}}[i,j] - C_{\text{BLAS}}[i,j])^2}$$

The verification confirms $\|E\|_2 = 0.00$ (within double-precision floating-point machine epsilon), ensuring zero indexing bugs or precision loss.

---

## 🚀 Compilation & Execution Guide

### Prerequisites
* **Compiler:** `gcc` or `clang`
* **BLAS Library:** `OpenBLAS` (Linux) or `Accelerate` framework (macOS)
* **Python (Optional):** `matplotlib`, `numpy` for regenerating figures

### 1. Running Numerical Verification
* **On Linux (with OpenBLAS):**
  ```bash
  gcc -O3 code/verify.c -o verify -lopenblas -lm
  ./verify
  ```
* **On macOS (with Apple Accelerate):**
  ```bash
  gcc -O3 code/verify.c -o verify -framework Accelerate -lm
  ./verify
  ```

### 2. Running Performance Benchmarks ($N = 5000$)
```bash
gcc -O3 -march=native code/opt_bench_5000.c -o opt_bench -lopenblas -lm
./opt_bench
```

### 3. Generating Performance Plots
```bash
python3 code/grafik_ciz.py
```

---

## 📊 Key Takeaways
1. **Algorithmic memory layout matters more than raw FLOP count:** Simply changing loop order from `IJK` to `IKJ` yields massive speedups without changing a single arithmetic operation.
2. **Cache tiling bridges the memory wall:** Keeping working sets inside L1/L2 caches mitigates main memory bandwidth bottlenecks.
3. **Production BLAS libraries remain the gold standard:** Hand-tuned micro-kernels with register-level blocking outperform standard compiler auto-vectorization on large problem sizes.
