# tinytransformer

A learning project — building a transformer from scratch in C++, step by step.

The goal is to implement every piece of the transformer architecture by hand, starting from basic matrix operations up to multi-head attention, a feed-forward block, and a KV cache for autoregressive decoding.

---

## Architecture

```
Input embeddings
  └── Positional encoding                         ✅
        └── Matrix ops (matmul, transpose, scale) ✅
              └── Q * K^T / sqrt(d_k)             ✅
                    └── Softmax → weights          ✅
                          └── weights * V          ✅
                                └── Multi-head    ✅
                                      └── FFN     ✅
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
| Positional encoding | `src/positional_encoding.cpp` | ✅ done |

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
After positional encoding:
  0.900   1.100   0.400   1.800   0.300   1.600   0.200   1.700
  1.341   1.340   0.300   1.595   0.910   1.100   0.701   1.300
  1.209   0.184   1.099   1.180   0.720   1.400   0.802   1.100
  0.841  -0.690   0.896   1.455   0.130   1.900   0.403   1.800

After add & norm (2)  →  transformer block output:
 -0.396  -0.420  -0.908   1.380  -1.012   1.147  -1.031   1.239
  0.271  -0.151  -1.328   1.698  -0.950   0.588  -1.064   0.936
  0.259  -1.282  -0.578   1.020  -0.968   1.551  -0.887   0.885
 -0.299  -1.255  -0.461   0.841  -0.896   1.489  -0.764   1.345
```

```
x  ──► PositionalEncoding
    ──► MultiHeadAttention(Q=x, K=x, V=x)
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
│   ├── matrix.hpp              # Matrix type + ops
│   ├── attention.hpp           # softmax, single-head attention
│   ├── multi_head.hpp          # MultiHeadAttention struct
│   ├── feedforward.hpp         # relu, layer_norm, feedforward
│   ├── kv_cache.hpp            # KVCache struct
│   └── positional_encoding.hpp # sinusoidal positional encoding
├── src/
│   ├── matrix.cpp
│   ├── attention.cpp
│   ├── multi_head.cpp
│   ├── feedforward.cpp
│   ├── kv_cache.cpp
│   └── positional_encoding.cpp
├── main.cpp             # end-to-end demo (block + KV cache)
└── Makefile
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
