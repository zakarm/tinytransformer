# tinytransformer

A learning project — building a transformer from scratch in C++, step by step.

The goal is to implement every piece of the transformer architecture by hand, starting from basic matrix operations up to multi-head attention, a feed-forward block, and a KV cache for autoregressive decoding.

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
                                      └── KV Cache ✅
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
| KV Cache | `src/kv_cache.cpp` | ✅ done |

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

The demo has two parts:

### Part 1 — full transformer block

All 4 tokens processed at once (`"The cat sat down"`, `d_model=8`, 2 heads):

```
After add & norm (2)  →  transformer block output:
  1.861  -1.190  -0.576   1.073  -0.800   0.101  -0.896   0.428
 -0.076   0.753  -1.029   0.091   1.773  -1.289   0.764  -0.986
 -0.624  -0.181   1.675  -0.929   0.675  -0.647   1.248  -1.216
  0.926  -0.991  -0.061  -0.463  -1.327   1.652  -0.708   0.973
```

```
x  ──► MultiHeadAttention(Q=x, K=x, V=x)
    ──► Add & LayerNorm   (residual)
    ──► FeedForward       (linear → ReLU → linear)
    ──► Add & LayerNorm   (residual)
    ──► output
```

### Part 2 — KV cache (autoregressive decoding)

Same sentence, but tokens arrive one at a time — each new token attends to all previous ones stored in the cache:

```
--- step 0:  "The"   (attends to 1 token)
--- step 1:  "cat"   (attends to 2 tokens)
--- step 2:  "sat"   (attends to 3 tokens)
--- step 3:  "down"  (attends to 4 tokens)
```

The output at step 3 matches row 3 of Part 1 exactly — proving the cache is correct.

```
Q (current token) ──► MultiHeadAttention(Q, K_cache, V_cache)
                  ──► Add & LayerNorm
                  ──► FeedForward
                  ──► Add & LayerNorm
                  ──► output
         cache.update(K, V) on every step
```

---

## Structure

```
tinytransformer/
├── include/
│   ├── matrix.hpp       # Matrix type + ops
│   ├── attention.hpp    # softmax, single-head attention
│   ├── multi_head.hpp   # MultiHeadAttention struct
│   ├── feedforward.hpp  # relu, layer_norm, feedforward
│   └── kv_cache.hpp     # KVCache struct
├── src/
│   ├── matrix.cpp
│   ├── attention.cpp
│   ├── multi_head.cpp
│   ├── feedforward.cpp
│   └── kv_cache.cpp
├── main.cpp             # end-to-end demo (block + KV cache)
└── Makefile
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
