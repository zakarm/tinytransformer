#include "include/matrix.hpp"
#include <iostream>
#include <cstdio>

Matrix transpose(const Matrix& A) {
    std::vector<std::vector<double>> result(A[0].size(), std::vector<double>(A.size()));
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[i].size(); j++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t rows  = A.size();
    size_t cols  = B[0].size();
    size_t inner = B.size();

    Matrix result(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            for (size_t k = 0; k < inner; k++){
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

Matrix scale(const Matrix& A, double s)
{
    std::vector<std::vector<double>> result(A.size(), std::vector<double>(A[0].size()));
    for (size_t i = 0; i < A.size(); i++){
        for (size_t j = 0; j < A[i].size(); j++){
            result[i][j] = A[i][j] * s;
        }
    }
    return result;
}

void print_matrix(const std::string& name, const Matrix& M) {
    std::cout << name << ":\n";
    for (auto& row : M) {
        for (double v : row) printf("%7.3f ", v);
        std::cout << "\n";
    }
    std::cout << "\n";
}