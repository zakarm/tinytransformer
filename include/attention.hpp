#pragma once
#include "matrix.hpp"

Matrix softmax(const Matrix& A);
Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V);