#include "include/kv_cache.hpp"

void KVCache::update(const Matrix& K, const Matrix& V) {
    this->keys.push_back(K);
    this->values.push_back(V);
}

Matrix KVCache::get_keys() const {
    Matrix result;
    for (const auto& K : this->keys) {
        result.insert(result.end(), K.begin(), K.end());
    }
    return result;
}

Matrix KVCache::get_values() const {
    Matrix result;
    for (const auto& V : this->values) {
        result.insert(result.end(), V.begin(), V.end());
    }
    return result;
}

void KVCache::clear() {
    this->keys.clear();
    this->values.clear();
}