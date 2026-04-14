#include "./include/feedforward.hpp"
#include <numeric>

Matrix relu(const Matrix& X){
    Matrix result = X;
    for (auto& row : result) {
        for (auto& v : row) {
            v = std::max(0.0, v);
        }
    }
    return result;
}

Matrix layer_norm(const Matrix& X) {
    Matrix result = X;
    for (auto& row : result) {
        double mean = std::accumulate(row.begin(), row.end(), 0.0) / row.size();

        double variance = 0.0;
        for (double v : row) {
            variance += (v - mean) * (v - mean);
        }
        variance /= row.size();
        double std_dev = sqrt(variance);

        for (double& v : row) {
            v = (v - mean) / (std_dev + 1e-6);
        }
    }
    return result;
}

Matrix feedforward(const Matrix& X, const Matrix& W1, const Matrix& W2) {
    Matrix hidden = matmul(X, W1);

    Matrix activated = relu(hidden);

    Matrix result = matmul(activated, W2);

    result = layer_norm(result);

    return result;
}
