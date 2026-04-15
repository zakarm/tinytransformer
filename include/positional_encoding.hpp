#pragma once
#include "matrix.hpp"
#include <cmath>

Matrix positional_encoding(int seq_len, int d_model);
Matrix add_positional_encoding(const Matrix& X);