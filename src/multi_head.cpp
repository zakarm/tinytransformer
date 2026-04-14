#include "include/multi_head.hpp"

MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model) {
    this->num_heads = num_heads;
    this->d_model = d_model;
    this->d_k = d_model / num_heads;
}

std::vector<Matrix> MultiHeadAttention::split_heads(const Matrix& X) const {
    std::vector<Matrix> result;
    for (int h = 0; h < this->num_heads; h++) 
    {
        Matrix head;
        for (const auto& row : X) {
            std::vector<double> head_row(row.begin() + h * this->d_k, row.begin() + (h + 1) * this->d_k);
            head.push_back(head_row);
        }
        result.push_back(head);
    }
    return result;
}

Matrix MultiHeadAttention::concat_heads(const std::vector<Matrix>& heads) const {
    Matrix result;
    for (size_t i = 0; i < heads[0].size(); i++)
    {
        std::vector<double> row;
        for (const auto& head : heads) {
            row.insert(row.end(), head[i].begin(), head[i].end());
        }
        result.push_back(row);
    }
    return result;
}

Matrix MultiHeadAttention::forward(const Matrix& Q, const Matrix& K, const Matrix& V) const {
    std::vector<Matrix> Q_heads = split_heads(Q);
    std::vector<Matrix> K_heads = split_heads(K);
    std::vector<Matrix> V_heads = split_heads(V);

    std::vector<Matrix> head_outputs;
    for (int h = 0; h < this->num_heads; h++) {
        head_outputs.push_back(attention(Q_heads[h], K_heads[h], V_heads[h]));
    }
    return concat_heads(head_outputs);
}
