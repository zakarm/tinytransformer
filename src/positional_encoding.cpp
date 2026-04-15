#include "./include/positional_encoding.hpp"

Matrix positional_encoding(int seq_len, int d_model){
    Matrix PE(seq_len, std::vector<double>(d_model, 0.0));
    for (int pos = 0; pos < seq_len; pos++){
        for (int i = 0; i < d_model / 2; i++){
            double denom = pow(10000.0, (2.0 * i) / d_model);
            PE[pos][2*i]   = sin(pos / denom);
            PE[pos][2*i+1] = cos(pos / denom);
        }
    }
    return PE;
}

Matrix add_positional_encoding(const Matrix& X) {
    Matrix PE = positional_encoding(X.size(), X[0].size());
    for (size_t i = 0; i < X.size(); i++) {
        for (size_t j = 0; j < X[0].size(); j++) {
            PE[i][j] += X[i][j];
        }
    }
    return PE;
}