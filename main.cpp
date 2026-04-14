#include "include/multi_head.hpp"
#include "include/feedforward.hpp"
#include <iostream>

int main() {
    // -----------------------------------------------------------------
    // One transformer block:
    //   x → MultiHeadAttention(Q=x, K=x, V=x) → add & norm → FeedForward → add & norm
    //
    // 4 tokens, d_model = 8, 2 heads  (d_k per head = 4)
    // Input represents 4 words in a sentence
    // -----------------------------------------------------------------

    // Each row = one token embedding (d_model = 8)
    Matrix X = {
        {0.9, 0.1, 0.4, 0.8, 0.3, 0.6, 0.2, 0.7},  // "The"
        {0.5, 0.8, 0.2, 0.6, 0.9, 0.1, 0.7, 0.3},  // "cat"
        {0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.8, 0.1},  // "sat"
        {0.7, 0.3, 0.6, 0.5, 0.1, 0.9, 0.4, 0.8},  // "down"
    };

    // Feed-forward weight matrices  (d_model x d_model = 8x8)
    // Using identity-like weights to keep the example readable
    Matrix W1 = {
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 1},
    };
    Matrix W2 = W1;

    print_matrix("Input X  (4 tokens x d_model=8)", X);

    // --- step 1: multi-head self-attention (2 heads) ---
    MultiHeadAttention mha(2, 8);
    Matrix attn_out = mha.forward(X, X, X);
    print_matrix("After multi-head attention", attn_out);

    // --- step 2: residual connection + layer norm ---
    // add X back (residual), then normalise
    Matrix residual1 = attn_out;
    for (size_t i = 0; i < X.size(); i++)
        for (size_t j = 0; j < X[0].size(); j++)
            residual1[i][j] += X[i][j];
    Matrix norm1 = layer_norm(residual1);
    print_matrix("After add & norm (1)", norm1);

    // --- step 3: feed-forward  (ReLU hidden layer + layer norm) ---
    Matrix ff_out = feedforward(norm1, W1, W2);
    print_matrix("After feed-forward", ff_out);

    // --- step 4: residual connection + layer norm ---
    Matrix residual2 = ff_out;
    for (size_t i = 0; i < norm1.size(); i++)
        for (size_t j = 0; j < norm1[0].size(); j++)
            residual2[i][j] += norm1[i][j];
    Matrix norm2 = layer_norm(residual2);
    print_matrix("After add & norm (2)  →  transformer block output", norm2);

    return 0;
}
