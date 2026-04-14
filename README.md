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
                          └── Multi-head attention ✅
                                └── Feed-forward  ✅
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
| Multi-head attention | `src/multi_head.cpp` | ✅ done |
| Feed-forward + ReLU | `src/feedforward.cpp` | ✅ done |
| Layer norm | `src/feedforward.cpp` | ✅ done |
| Residual connections | `main.cpp` | ✅ done |

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

Runs a full transformer block on 4 tokens (`"The cat sat down"`) with `d_model=8`, 2 attention heads:

```
Input X  (4 tokens x d_model=8):
  0.900   0.100   0.400   0.800   0.300   0.600   0.200   0.700
  0.500   0.800   0.200   0.600   0.900   0.100   0.700   0.300
  0.300   0.600   0.900   0.200   0.700   0.400   0.800   0.100
  0.700   0.300   0.600   0.500   0.100   0.900   0.400   0.800

After add & norm (2)  →  transformer block output:
  1.861  -1.190  -0.576   1.073  -0.800   0.101  -0.896   0.428
 -0.076   0.753  -1.029   0.091   1.773  -1.289   0.764  -0.986
 -0.624  -0.181   1.675  -0.929   0.675  -0.647   1.248  -1.216
  0.926  -0.991  -0.061  -0.463  -1.327   1.652  -0.708   0.973
```

Each token starts as a raw embedding — the output is a contextualised representation where every token has mixed in information from the others via attention.

### Block pipeline

```
x  ──► MultiHeadAttention(Q=x, K=x, V=x)
         └─ split into h heads, attention per head, concat
    ──► Add & LayerNorm   (residual connection)
    ──► FeedForward       (linear → ReLU → linear)
    ──► Add & LayerNorm   (residual connection)
    ──► output
```

---

## Structure

```
tinytransformer/
├── include/
│   ├── matrix.hpp       # Matrix type + ops
│   ├── attention.hpp    # softmax, single-head attention
│   ├── multi_head.hpp   # MultiHeadAttention struct
│   └── feedforward.hpp  # relu, layer_norm, feedforward
├── src/
│   ├── matrix.cpp
│   ├── attention.cpp
│   ├── multi_head.cpp
│   └── feedforward.cpp
├── main.cpp             # end-to-end transformer block demo
└── Makefile
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
