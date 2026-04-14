// include/kv_cache.hpp
#pragma once
#include "matrix.hpp"

struct KVCache {
    std::vector<Matrix> keys;
    std::vector<Matrix> values;

    void update(const Matrix& K, const Matrix& V);
    Matrix get_keys()   const;
    Matrix get_values() const;
    void clear();
};