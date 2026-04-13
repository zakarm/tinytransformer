# tinytransformer

A learning project — building a transformer from scratch in C++, step by step.

The goal is to implement every piece of the transformer architecture by hand, starting from basic matrix operations up to multi-head attention and a feed-forward block.

---

## Architecture

```
Input
  └── Matrix ops (matmul, transpose, scale)       ✅
        └── Attention scores: Q * K^T / sqrt(d_k) ✅
              └── Softmax  →  Attention weights    ✅
                    └── weights * V  →  Output     ✅
                          └── Multi-head attention 🔧
                                └── Feed-forward  🔧
```

---

## Progress

| Component | File | Status |
|---|---|---|
| Matrix multiply | `src/matrix.cpp` | ✅ done |
| Transpose | `src/matrix.cpp` | ✅ done |
| Scale | `src/matrix.cpp` | ✅ done |
| Softmax | `src/attention.cpp` | ✅ done |
| Single-head attention | `src/attention.cpp` | ✅ done |
| Multi-head attention | `src/multi_head.cpp` | 🔧 todo |
| Feed-forward block | `src/feedforward.cpp` | 🔧 todo |

---

## Build

```bash
make        # builds libtinytransformer.a + tinytransformer binary
make clean  # removes build artifacts
```

Requires a C++17 compiler (`g++` or `clang++`).

---

## Run

```bash
./tinytransformer
```

Current output runs the full single-head attention — `Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V` — with 3 tokens of dimension 4:

```
softmax (attention weights):
  0.506   0.186   0.307
  0.186   0.506   0.307
  0.274   0.274   0.452

output  (weights * V):
  4.203   4.588   4.974   5.360
  5.483   5.869   6.255   6.640
  5.711   5.807   5.904   6.000
```

Each row of the output is a weighted blend of the value vectors — tokens attend mostly to themselves, with some weight on similar tokens.

---

## Structure

```
tinytransformer/
├── include/        # headers
│   ├── matrix.hpp
│   ├── attention.hpp
│   ├── multi_head.hpp
│   └── feedforward.hpp
├── src/            # implementations
│   ├── matrix.cpp
│   ├── attention.cpp
│   ├── multi_head.cpp
│   └── feedforward.cpp
├── main.cpp        # demo / scratchpad
└── Makefile
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
