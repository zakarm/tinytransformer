#pragma once
#include "attention.hpp"

struct MultiHeadAttention {
    int num_heads;
    int d_model;
    int d_k;

    MultiHeadAttention(int num_heads, int d_model);

    std::vector<Matrix> split_heads(const Matrix& X) const;
    Matrix concat_heads(const std::vector<Matrix>& heads) const;
    Matrix forward(const Matrix& Q, const Matrix& K, const Matrix& V) const;
};
