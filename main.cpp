#include "include/matrix.hpp"
#include <cmath>
#include <iostream>

int main() {
    // -----------------------------------------------------------------
    // Attention score: scores = Q * K^T / sqrt(d_k)
    //
    // 3 tokens, d_k = 4  (each row = one token's query / key vector)
    // -----------------------------------------------------------------
    int d_k = 4;

    Matrix Q = {
        {1, 0, 1, 0},   // token 0
        {0, 1, 0, 1},   // token 1
        {1, 1, 0, 0},   // token 2
    };

    Matrix K = {
        {1, 0, 1, 0},   // token 0
        {0, 1, 0, 1},   // token 1
        {1, 1, 0, 0},   // token 2
    };

    print_matrix("Q (3 tokens, d_k=4)", Q);
    print_matrix("K (3 tokens, d_k=4)", K);

    // step 1: Q * K^T  →  raw dot-products between every pair of tokens
    Matrix Kt     = transpose(K);
    Matrix scores = matmul(Q, Kt);
    print_matrix("Q * K^T  (raw scores, 3x3)", scores);

    // step 2: scale by 1/sqrt(d_k)  — needs scale() to be implemented
    // scores /= sqrt(d_k)
    Matrix scaled = scale(scores, 1.0 / std::sqrt(d_k));
    if (scaled.empty()) {
        std::cout << "scaled scores: scale() not yet implemented\n";
        std::cout << "(expected each value above divided by sqrt(" << d_k
                  << ") = " << std::sqrt(d_k) << ")\n\n";
    } else {
        print_matrix("Q * K^T / sqrt(d_k)  (scaled scores, 3x3)", scaled);
    }

    // next step after scale: softmax row-wise → attention weights
    // next step after that:  weights * V       → attended output

    return 0;
}
