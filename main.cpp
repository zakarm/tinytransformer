#include "include/matrix.hpp"
#include "include/attention.hpp"
#include <iostream>

int main() {
    // -----------------------------------------------------------------
    // Full self-attention: Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k)) * V
    //
    // 3 tokens, d_k = d_v = 4
    // -----------------------------------------------------------------

    Matrix Q = {
        {1, 0, 1, 0},
        {0, 1, 0, 1},
        {1, 1, 0, 0},
    };

    Matrix K = {
        {1, 0, 1, 0},
        {0, 1, 0, 1},
        {1, 1, 0, 0},
    };

    Matrix V = {
        {1, 2, 3, 4},   // value for token 0
        {5, 6, 7, 8},   // value for token 1
        {9, 8, 7, 6},   // value for token 2
    };

    print_matrix("Q", Q);
    print_matrix("K", K);
    print_matrix("V", V);

    Matrix scores = matmul(Q, transpose(K));
    print_matrix("Q * K^T  (raw scores)", scores);

    Matrix scaled = scale(scores, 1.0 / 2.0);   // sqrt(d_k=4) = 2
    print_matrix("scaled  (/ sqrt(d_k))", scaled);

    Matrix weights = softmax(scaled);
    print_matrix("softmax (attention weights)", weights);

    Matrix out = matmul(weights, V);
    print_matrix("output  (weights * V)", out);

    std::cout << "--- full attention() call ---\n\n";
    print_matrix("attention(Q, K, V)", attention(Q, K, V));

    return 0;
}
