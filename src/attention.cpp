#include "include/attention.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

Matrix softmax(const Matrix& A) {
    Matrix result = A;

    for (auto row = result.begin(); row != result.end(); ++row){
        double max = *std::max_element(row->begin(), row->end());
        std::transform(row->begin(), row->end(), row->begin(),
               [max](double c) { return c - max; });
        std::transform(row->begin(), row->end(), row->begin(),
               [](double c) { return std::exp(c); });
        double sum = std::accumulate(row->begin(), row->end(), 0.0);
        std::transform(row->begin(), row->end(), row->begin(),
               [sum](double c) { return c / sum; });
    }
    return result;
}

Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V) {
    int d_k = K[0].size();

    Matrix scores = matmul(Q, transpose(K));
    Matrix scaled  = scale(scores, 1.0 / sqrt(d_k));
    Matrix weights = softmax(scaled);
    return matmul(weights, V);
}