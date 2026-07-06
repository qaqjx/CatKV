/**
 * @file tensor.h
 * @brief Lightweight tensor definition with basic serialization helpers.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace disk_manager {

enum class DType {
  FLOAT32,
  FLOAT16,
  INT8,
  UINT8,
  BFLOAT16,
  INT32,
  FLOAT64,
};

struct Tensor {
  DType dtype{};
  std::size_t ndim{};
  std::vector<std::size_t> shape;
  std::size_t data_size{};
  std::unique_ptr<std::byte[]> data;
  void* external_data_ = nullptr;
  std::shared_ptr<void> external_owner_;

  Tensor() = default;

  Tensor(DType dtype_, std::initializer_list<std::size_t> dims)
      : dtype(dtype_), ndim(dims.size()), shape(dims), data_size(ComputeSize()) {
    data = std::make_unique<std::byte[]>(data_size);
  }

  Tensor(DType dtype_, std::vector<std::size_t> dims)
      : dtype(dtype_), ndim(dims.size()), shape(std::move(dims)), data_size(ComputeSize()) {
    data = std::make_unique<std::byte[]>(data_size);
  }

  Tensor(DType dtype_, std::vector<std::size_t> dims, void* external_data)
      : dtype(dtype_),
        ndim(dims.size()),
        shape(std::move(dims)),
        data_size(ComputeSize()),
        external_data_(external_data) {}

  Tensor(const Tensor& other)
      : dtype(other.dtype),
        ndim(other.ndim),
        shape(other.shape),
        data_size(other.data_size),
        data(other.data_size ? std::make_unique<std::byte[]>(other.data_size) : nullptr),
        external_data_(nullptr),
        external_owner_() {
    if (data && other.data_size) {
      std::memcpy(data.get(), other.data_ptr(), data_size);
    }
  }

  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor& other) {
    if (this == &other) {
      return *this;
    }
    dtype = other.dtype;
    ndim = other.ndim;
    shape = other.shape;
    data_size = other.data_size;
    data.reset(other.data_size ? new std::byte[other.data_size] : nullptr);
    external_data_ = nullptr;
    external_owner_.reset();
    if (data && other.data_size) {
      std::memcpy(data.get(), other.data_ptr(), data_size);
    }
    return *this;
  }

  Tensor& operator=(Tensor&&) noexcept = default;
  ~Tensor() = default;

  static Tensor from_raw(DType dtype_, std::vector<std::size_t> dims, const void* src, std::size_t bytes) {
    Tensor tensor(dtype_, std::move(dims));
    if (tensor.data_size != bytes) {
      throw std::invalid_argument("Byte size mismatch when creating tensor from raw data.");
    }
    if (src && tensor.data_ptr()) {
      std::memcpy(tensor.data_ptr(), src, bytes);
    }
    return tensor;
  }

  std::vector<std::byte> to_bytes() const {
    const std::byte* ptr = data_ptr();
    if (!ptr) {
      return {};
    }
    return {ptr, ptr + data_size};
  }

  std::byte* data_ptr() {
    return external_data_ ? static_cast<std::byte*>(external_data_) : data.get();
  }

  const std::byte* data_ptr() const {
    return external_data_ ? static_cast<const std::byte*>(external_data_) : data.get();
  }

private:
  std::size_t ComputeSize() const {
    std::size_t element_size = ElementSize(dtype);
    std::size_t total_elements = 1;
    for (auto dim : shape) {
      total_elements *= dim;
    }
    return element_size * total_elements;
  }

  static std::size_t ElementSize(DType t) {
    switch (t) {
      case DType::FLOAT32:
        return 4;
      case DType::FLOAT16:
        return 2;
      case DType::INT8:
        return 1;
      case DType::UINT8:
        return 1;
      case DType::BFLOAT16:
        return 2;
      case DType::INT32:
        return 4;
      case DType::FLOAT64:
        return 8;
    }
    throw std::runtime_error("Unsupported dtype.");
  }
};

}  // namespace disk_manager
