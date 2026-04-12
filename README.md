# tinytransformer

A learning project — building a transformer from scratch in C++, step by step.

The goal is to implement every piece of the transformer architecture by hand, starting from basic matrix operations up to multi-head attention and a feed-forward block.

---

## Architecture

```
Input
  └── Matrix ops (matmul, transpose, scale)
        └── Attention scores: Q * K^T / sqrt(d_k)
              └── Softmax  →  Attention weights
                    └── weights * V  →  Attended output
                          └── Multi-head attention
                                └── Feed-forward block
```

---

## Progress

| Component | File | Status |
|---|---|---|
| Matrix multiply | `src/matrix.cpp` | ✅ done |
| Transpose | `src/matrix.cpp` | ✅ done |
| Scale | `src/matrix.cpp` | 🔧 todo |
| Softmax | — | 🔧 todo |
| Single-head attention | `src/attention.cpp` | 🔧 todo |
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

Current output demos the attention score computation `Q * K^T / sqrt(d_k)` with 3 tokens of dimension 4:

```
Q * K^T  (raw scores, 3x3):
  2.000   0.000   1.000
  0.000   2.000   1.000
  1.000   1.000   2.000
```

Each entry `[i][j]` is the dot-product between token `i`'s query and token `j`'s key — how much token `i` should attend to token `j`.

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
