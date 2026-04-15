#include "include/positional_encoding.hpp"
#include "include/multi_head.hpp"
#include "include/feedforward.hpp"
#include "include/kv_cache.hpp"
#include <iostream>

// helper: add two matrices element-wise (residual connection)
Matrix add(const Matrix& A, const Matrix& B) {
    Matrix result = A;
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            result[i][j] += B[i][j];
    return result;
}

int main() {
    // =================================================================
    // Part 1 — full transformer block (all tokens at once)
    //
    // "The cat sat down"  →  4 tokens, d_model=8, 2 heads
    // =================================================================

    Matrix X = {
        {0.9, 0.1, 0.4, 0.8, 0.3, 0.6, 0.2, 0.7},  // "The"
        {0.5, 0.8, 0.2, 0.6, 0.9, 0.1, 0.7, 0.3},  // "cat"
        {0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.8, 0.1},  // "sat"
        {0.7, 0.3, 0.6, 0.5, 0.1, 0.9, 0.4, 0.8},  // "down"
    };

    Matrix W1 = {
        {1,0,0,0,0,0,0,0}, {0,1,0,0,0,0,0,0},
        {0,0,1,0,0,0,0,0}, {0,0,0,1,0,0,0,0},
        {0,0,0,0,1,0,0,0}, {0,0,0,0,0,1,0,0},
        {0,0,0,0,0,0,1,0}, {0,0,0,0,0,0,0,1},
    };
    Matrix W2 = W1;

    MultiHeadAttention mha(2, 8);

    print_matrix("Input X  (4 tokens x d_model=8)", X);

    // --- step 0: positional encoding ---
    Matrix X_pe = add_positional_encoding(X);
    print_matrix("After positional encoding", X_pe);

    Matrix attn_out = mha.forward(X_pe, X_pe, X_pe);
    print_matrix("After multi-head attention", attn_out);

    Matrix norm1 = layer_norm(add(attn_out, X_pe));
    print_matrix("After add & norm (1)", norm1);

    Matrix ff_out = feedforward(norm1, W1, W2);
    print_matrix("After feed-forward", ff_out);

    Matrix norm2 = layer_norm(add(ff_out, norm1));
    print_matrix("After add & norm (2)  →  transformer block output", norm2);

    // =================================================================
    // Part 2 — KV cache: autoregressive decoding token by token
    //
    // Same sentence, but tokens arrive one at a time (like generation).
    // Each new token attends to ALL previous tokens via the cache.
    // =================================================================

    std::cout << "=================================================================\n";
    std::cout << "KV Cache — autoregressive decoding (one token at a time)\n";
    std::cout << "=================================================================\n\n";

    MultiHeadAttention mha_dec(2, 8);
    KVCache cache;

    // same 4 token embeddings, processed one by one
    std::vector<Matrix> tokens = {
        {{0.9, 0.1, 0.4, 0.8, 0.3, 0.6, 0.2, 0.7}},  // "The"
        {{0.5, 0.8, 0.2, 0.6, 0.9, 0.1, 0.7, 0.3}},  // "cat"
        {{0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.8, 0.1}},  // "sat"
        {{0.7, 0.3, 0.6, 0.5, 0.1, 0.9, 0.4, 0.8}},  // "down"
    };
    std::vector<std::string> words = {"The", "cat", "sat", "down"};

    for (size_t t = 0; t < tokens.size(); t++) {
        Matrix q = tokens[t];

        // store current token K and V in cache
        cache.update(q, q);

        // Q = current token only, K/V = all tokens seen so far
        Matrix K = cache.get_keys();
        Matrix V = cache.get_values();

        // transformer block for this single token
        Matrix a_out  = mha_dec.forward(q, K, V);
        Matrix n1     = layer_norm(add(a_out, q));
        Matrix f_out  = feedforward(n1, W1, W2);
        Matrix output = layer_norm(add(f_out, n1));

        std::cout << "--- step " << t << ":  \"" << words[t]
                  << "\"  (attends to " << t + 1 << " token"
                  << (t ? "s" : "") << ")\n\n";
        print_matrix("output", output);
    }

    cache.clear();
    std::cout << "cache cleared.\n";

    return 0;
}
