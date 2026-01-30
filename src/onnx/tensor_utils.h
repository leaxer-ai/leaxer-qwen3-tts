#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace leaxer {

/**
 * Tensor utilities for ONNX Runtime
 * 
 * These helpers simplify creating and extracting data from Ort::Value tensors.
 */
namespace tensor {

/**
 * Create a float tensor from raw data (copies data)
 * @param data Pointer to float data
 * @param shape Tensor shape
 * @return Ort::Value containing the tensor
 */
inline Ort::Value createFloat(
    const float* data,
    const std::vector<int64_t>& shape
) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    int64_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }
    
    // Create tensor (ORT will copy the data)
    return Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(data),
        static_cast<size_t>(total_size),
        shape.data(),
        shape.size()
    );
}

/**
 * Create a float tensor from vector
 * @param data Vector of float data
 * @param shape Tensor shape
 * @return Ort::Value containing the tensor
 */
inline Ort::Value createFloat(
    const std::vector<float>& data,
    const std::vector<int64_t>& shape
) {
    return createFloat(data.data(), shape);
}

/**
 * Create an int64 tensor from raw data
 * @param data Pointer to int64_t data
 * @param shape Tensor shape
 * @return Ort::Value containing the tensor
 */
inline Ort::Value createInt64(
    const int64_t* data,
    const std::vector<int64_t>& shape
) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    int64_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }
    
    return Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t*>(data),
        static_cast<size_t>(total_size),
        shape.data(),
        shape.size()
    );
}

/**
 * Create an int64 tensor from vector
 * @param data Vector of int64_t data
 * @param shape Tensor shape
 * @return Ort::Value containing the tensor
 */
inline Ort::Value createInt64(
    const std::vector<int64_t>& data,
    const std::vector<int64_t>& shape
) {
    return createInt64(data.data(), shape);
}

/**
 * Create an int32 tensor from raw data
 * @param data Pointer to int32_t data
 * @param shape Tensor shape
 * @return Ort::Value containing the tensor
 */
inline Ort::Value createInt32(
    const int32_t* data,
    const std::vector<int64_t>& shape
) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    int64_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }
    
    return Ort::Value::CreateTensor<int32_t>(
        memory_info,
        const_cast<int32_t*>(data),
        static_cast<size_t>(total_size),
        shape.data(),
        shape.size()
    );
}

/**
 * Extract float data from an Ort::Value tensor
 * @param value The tensor value
 * @return Vector containing copied data
 */
inline std::vector<float> extractFloat(const Ort::Value& value) {
    auto type_info = value.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= static_cast<size_t>(dim);
    }
    
    const float* data = value.GetTensorData<float>();
    return std::vector<float>(data, data + total_size);
}

/**
 * Extract int64 data from an Ort::Value tensor
 * @param value The tensor value
 * @return Vector containing copied data
 */
inline std::vector<int64_t> extractInt64(const Ort::Value& value) {
    auto type_info = value.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= static_cast<size_t>(dim);
    }
    
    const int64_t* data = value.GetTensorData<int64_t>();
    return std::vector<int64_t>(data, data + total_size);
}

/**
 * Extract int32 data from an Ort::Value tensor
 * @param value The tensor value
 * @return Vector containing copied data
 */
inline std::vector<int32_t> extractInt32(const Ort::Value& value) {
    auto type_info = value.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= static_cast<size_t>(dim);
    }
    
    const int32_t* data = value.GetTensorData<int32_t>();
    return std::vector<int32_t>(data, data + total_size);
}

/**
 * Get tensor shape from Ort::Value
 * @param value The tensor value
 * @return Shape vector
 */
inline std::vector<int64_t> getShape(const Ort::Value& value) {
    auto type_info = value.GetTensorTypeAndShapeInfo();
    return type_info.GetShape();
}

/**
 * Get total element count from Ort::Value
 * @param value The tensor value
 * @return Number of elements
 */
inline size_t getElementCount(const Ort::Value& value) {
    auto type_info = value.GetTensorTypeAndShapeInfo();
    return type_info.GetElementCount();
}

/**
 * Get raw float pointer from tensor (zero-copy access)
 * WARNING: Pointer is only valid while Ort::Value exists
 * @param value The tensor value
 * @return Pointer to float data
 */
inline const float* getFloatPtr(const Ort::Value& value) {
    return value.GetTensorData<float>();
}

/**
 * Get raw int64 pointer from tensor (zero-copy access)
 * WARNING: Pointer is only valid while Ort::Value exists
 * @param value The tensor value
 * @return Pointer to int64 data
 */
inline const int64_t* getInt64Ptr(const Ort::Value& value) {
    return value.GetTensorData<int64_t>();
}

/**
 * Get mutable float pointer from tensor (for in-place modification)
 * WARNING: Pointer is only valid while Ort::Value exists
 * @param value The tensor value (non-const reference)
 * @return Pointer to float data
 */
inline float* getMutableFloatPtr(Ort::Value& value) {
    return value.GetTensorMutableData<float>();
}

} // namespace tensor
} // namespace leaxer
