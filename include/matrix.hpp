# pragma once

#include <vector>
#include <string>

using Matrix = std::vector<std::vector<double>>;

Matrix matmul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);
Matrix scale(const Matrix& A, double s);
void   print_matrix(const std::string& name, const Matrix& M);