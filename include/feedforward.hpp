#pragma once
#include "matrix.hpp"

Matrix relu(const Matrix& X);
Matrix layer_norm(const Matrix& X);
Matrix feedforward(const Matrix& X, const Matrix& W1, const Matrix& W2);