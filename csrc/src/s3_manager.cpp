#include "s3_manager.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <sys/mman.h>

namespace s3 {
namespace {

constexpr std::array<char, 8> kMagic{'T', 'E', 'N', 'S', 'D', 'I', 'C', 'T'};
constexpr std::uint32_t kVersion = 2;
constexpr std::size_t kTensorAlignment = 256;
constexpr std::size_t kBlockSize = 4096;
constexpr std::size_t kGpuArenaSize = 512ULL * 1024 * 1024;
constexpr std::size_t kGpuAlignment = 256;

std::once_flag g_curl_global_flag;

std::size_t align_up(std::size_t size, std::size_t block_size) {
  if (block_size == 0) {
    return size;
  }
  return (size + block_size - 1) / block_size * block_size;
}

std::tm gmtime_utc(std::time_t t) {
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  return tm;
}

std::string uri_encode(const std::string& input, bool encode_slash) {
  static constexpr char hex[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(input.size() * 3);
  for (unsigned char c : input) {
    const bool safe =
        (c >= 'A' && c <= 'Z') ||
        (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9') ||
        c == '-' || c == '_' || c == '.' || c == '~' ||
        (!encode_slash && c == '/');
    if (safe) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 0x0F]);
    }
  }
  return out;
}

std::string to_hex(const unsigned char* data, std::size_t len) {
  static constexpr char hex_digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(len * 2);
  for (std::size_t i = 0; i < len; ++i) {
    unsigned char c = data[i];
    out.push_back(hex_digits[c >> 4]);
    out.push_back(hex_digits[c & 0x0F]);
  }
  return out;
}

std::string sha256_hex(const void* data, std::size_t len) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(data), len, hash);
  return to_hex(hash, SHA256_DIGEST_LENGTH);
}

std::string sha256_hex(const std::string& data) {
  return sha256_hex(data.data(), data.size());
}

std::vector<unsigned char> hmac_sha256(const unsigned char* key, std::size_t key_len,
                                       std::string_view data) {
  unsigned int len = 0;
  std::vector<unsigned char> out(EVP_MAX_MD_SIZE);
  HMAC(EVP_sha256(),
       key,
       static_cast<int>(key_len),
       reinterpret_cast<const unsigned char*>(data.data()),
       data.size(),
       out.data(),
       &len);
  out.resize(len);
  return out;
}

std::vector<unsigned char> hmac_sha256(const std::vector<unsigned char>& key,
                                       std::string_view data) {
  if (key.empty()) {
    return hmac_sha256(nullptr, 0, data);
  }
  return hmac_sha256(key.data(), key.size(), data);
}

torch::ScalarType DTypeToTorchScalar(disk_manager::DType dtype) {
  using disk_manager::DType;
  switch (dtype) {
    case DType::FLOAT32: return torch::kFloat32;
    case DType::FLOAT16: return torch::kFloat16;
    case DType::INT8: return torch::kInt8;
    case DType::UINT8: return torch::kUInt8;
    case DType::BFLOAT16: return torch::kBFloat16;
    case DType::INT32: return torch::kInt32;
    case DType::FLOAT64: return torch::kFloat64;
  }
  throw std::runtime_error("Unsupported dtype for torch conversion");
}

class MemoryReader {
 public:
  MemoryReader(char* begin, char* end) : cur_(begin), end_(end) {}

  template <typename T>
  T read_value() {
    constexpr std::size_t size = sizeof(T);
    ensure_available(size);
    T out{};
    std::memcpy(&out, cur_, size);
    cur_ += size;
    return out;
  }

  void read_bytes(void* dest, std::size_t len) {
    if (len == 0) return;
    ensure_available(len);
    std::memcpy(dest, cur_, len);
    cur_ += len;
  }

  void skip(std::size_t n) {
    if (n == 0) return;
    ensure_available(n);
    cur_ += n;
  }

  char* current_ptr() const { return cur_; }

 private:
  void ensure_available(std::size_t len) const {
    if (static_cast<std::size_t>(end_ - cur_) < len) {
      throw std::runtime_error("Unexpected end of buffer during parse");
    }
  }

  char* cur_;
  char* end_;
};

disk_manager::Tensor read_tensor(MemoryReader& reader, const std::shared_ptr<void>& owner) {
  using disk_manager::Tensor;
  const auto dtype_raw = reader.read_value<std::uint32_t>();
  const auto dtype = static_cast<disk_manager::DType>(dtype_raw);
  const auto ndim = reader.read_value<std::uint64_t>();
  std::vector<std::size_t> shape;
  shape.reserve(static_cast<std::size_t>(ndim));
  for (std::uint64_t i = 0; i < ndim; ++i) {
    shape.push_back(static_cast<std::size_t>(reader.read_value<std::uint64_t>()));
  }
  const auto data_size = static_cast<std::size_t>(reader.read_value<std::uint64_t>());
  Tensor tensor(dtype, std::move(shape));
  if (tensor.data_size != data_size) {
    throw std::runtime_error("Tensor byte-size mismatch");
  }
  if (tensor.data_size > 0) {
    const auto padding_size = reader.read_value<std::uint32_t>();
    if (padding_size > 0) {
      reader.skip(padding_size);
    }
    tensor.external_data_ = reader.current_ptr();
    tensor.external_owner_ = owner;
    reader.skip(tensor.data_size);
  }
  return tensor;
}

S3Manager::TensorMap parse_from_memory_zero_copy(
    char* begin,
    char* end,
    const std::shared_ptr<void>& owner) {
  MemoryReader reader(begin, end);
  std::array<char, kMagic.size()> magic{};
  reader.read_bytes(magic.data(), magic.size());
  if (!std::equal(magic.begin(), magic.end(), kMagic.begin(), kMagic.end())) {
    throw std::runtime_error("Invalid tensor dictionary magic");
  }
  const auto version = reader.read_value<std::uint32_t>();
  if (version != kVersion) {
    throw std::runtime_error("Unsupported tensor dictionary version");
  }
  const auto tensor_count = reader.read_value<std::uint32_t>();
  S3Manager::TensorMap tensors;
  tensors.reserve(tensor_count);
  for (std::uint32_t i = 0; i < tensor_count; ++i) {
    const auto key_size = reader.read_value<std::uint32_t>();
    std::string key(key_size, '\0');
    reader.read_bytes(key.data(), key.size());
    tensors.emplace(std::move(key), read_tensor(reader, owner));
  }
  return tensors;
}

void write_value(std::ostream& os, std::uint64_t value, std::size_t bytes) {
  for (std::size_t i = 0; i < bytes; ++i) {
    os.put(static_cast<char>((value >> (i * 8)) & 0xFF));
  }
  if (!os) {
    throw std::runtime_error("Failed to write binary value");
  }
}

void append_value(std::string& out, std::uint64_t value, std::size_t bytes) {
  const std::size_t start = out.size();
  out.resize(start + bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[start + i] = static_cast<char>((value >> (i * 8)) & 0xFF);
  }
}

std::size_t serialized_tensor_size(std::size_t cursor, const disk_manager::Tensor& tensor) {
  std::size_t total = 0;
  total += sizeof(std::uint32_t);  // dtype
  total += sizeof(std::uint64_t);  // ndim
  total += tensor.shape.size() * sizeof(std::uint64_t);
  total += sizeof(std::uint64_t);  // data_size
  cursor += total;
  if (tensor.data_size == 0) {
    return total;
  }
  total += sizeof(std::uint32_t);  // padding size
  cursor += sizeof(std::uint32_t);
  const std::uint32_t padding = static_cast<std::uint32_t>(
      (kTensorAlignment - (cursor % kTensorAlignment)) % kTensorAlignment);
  total += padding;
  total += tensor.data_size;
  return total;
}

std::string serialize_tensor_map(const S3Manager::TensorMap& tensors) {
  std::size_t total_size = 0;
  total_size += kMagic.size();
  total_size += sizeof(kVersion);
  total_size += sizeof(std::uint32_t);  // tensor count
  for (const auto& [name, tensor] : tensors) {
    total_size += sizeof(std::uint32_t);
    total_size += name.size();
    total_size += serialized_tensor_size(total_size + name.size() + sizeof(std::uint32_t), tensor);
  }

  std::string out;
  out.reserve(total_size);
  out.append(kMagic.data(), kMagic.size());
  append_value(out, kVersion, sizeof(kVersion));
  append_value(out, static_cast<std::uint32_t>(tensors.size()), sizeof(std::uint32_t));

  for (const auto& [name, tensor] : tensors) {
    append_value(out, static_cast<std::uint32_t>(name.size()), sizeof(std::uint32_t));
    out.append(name.data(), name.size());
    append_value(out, static_cast<std::uint32_t>(tensor.dtype), sizeof(std::uint32_t));
    append_value(out, static_cast<std::uint64_t>(tensor.ndim), sizeof(std::uint64_t));
    for (std::size_t dim : tensor.shape) {
      append_value(out, static_cast<std::uint64_t>(dim), sizeof(std::uint64_t));
    }
    append_value(out, static_cast<std::uint64_t>(tensor.data_size), sizeof(std::uint64_t));
    if (tensor.data_size == 0) {
      continue;
    }
    const std::uint32_t padding = static_cast<std::uint32_t>(
        (kTensorAlignment - ((out.size() + sizeof(std::uint32_t)) % kTensorAlignment)) % kTensorAlignment);
    append_value(out, padding, sizeof(std::uint32_t));
    if (padding > 0) {
      out.append(padding, '\0');
    }
    const std::byte* ptr = tensor.data_ptr();
    if (!ptr) {
      throw std::runtime_error("Tensor data missing during serialization");
    }
    out.append(reinterpret_cast<const char*>(ptr), tensor.data_size);
  }

  return out;
}

void write_tensor(std::ostream& os, const disk_manager::Tensor& tensor) {
  write_value(os, static_cast<std::uint32_t>(tensor.dtype), sizeof(std::uint32_t));
  write_value(os, static_cast<std::uint64_t>(tensor.ndim), sizeof(std::uint64_t));
  for (std::size_t dim : tensor.shape) {
    write_value(os, static_cast<std::uint64_t>(dim), sizeof(std::uint64_t));
  }
  write_value(os, static_cast<std::uint64_t>(tensor.data_size), sizeof(std::uint64_t));
  if (tensor.data_size == 0) {
    return;
  }
  const auto cur_pos = static_cast<std::size_t>(os.tellp());
  const std::uint32_t padding = static_cast<std::uint32_t>(
      (kTensorAlignment - ((cur_pos + sizeof(std::uint32_t)) % kTensorAlignment)) % kTensorAlignment);
  write_value(os, padding, sizeof(std::uint32_t));
  if (padding > 0) {
    std::array<char, kTensorAlignment> zeros{};
    os.write(zeros.data(), padding);
  }
  const std::byte* ptr = tensor.data_ptr();
  if (!ptr || tensor.data_size == 0) {
    throw std::runtime_error("Tensor data missing during serialization");
  }
  os.write(reinterpret_cast<const char*>(ptr), tensor.data_size);
  if (!os) {
    throw std::runtime_error("Failed to serialize tensor payload");
  }
}

struct UploadBuffer {
  const char* data = nullptr;
  std::size_t size = 0;
  std::size_t offset = 0;
};

size_t read_callback(char* buffer, size_t size, size_t nmemb, void* userdata) {
  UploadBuffer* upload = static_cast<UploadBuffer*>(userdata);
  const std::size_t space = size * nmemb;
  if (!upload || space == 0) {
    return 0;
  }
  const std::size_t remaining = upload->size - upload->offset;
  const std::size_t to_copy = std::min(space, remaining);
  if (to_copy > 0) {
    std::memcpy(buffer, upload->data + upload->offset, to_copy);
    upload->offset += to_copy;
  }
  return to_copy;
}

struct RangeWriter {
  char* dest = nullptr;
  std::size_t capacity = 0;
  std::size_t written = 0;
};

size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  RangeWriter* writer = static_cast<RangeWriter*>(userdata);
  if (!writer || !writer->dest) {
    return 0;
  }
  const std::size_t total = size * nmemb;
  const std::size_t remaining = writer->capacity - writer->written;
  const std::size_t to_copy = std::min(total, remaining);
  if (to_copy > 0) {
    std::memcpy(writer->dest + writer->written, ptr, to_copy);
    writer->written += to_copy;
  }
  return to_copy;
}

}  // namespace

std::vector<std::size_t> ComputeAlignedObjectOffsets(
    const std::vector<std::size_t>& sizes,
    std::size_t alignment) {
  if (alignment == 0) {
    throw std::invalid_argument("alignment must be positive");
  }

  std::vector<std::size_t> offsets;
  offsets.reserve(sizes.size());
  std::size_t cursor = 0;
  for (std::size_t size : sizes) {
    cursor = align_up(cursor, alignment);
    offsets.push_back(cursor);
    cursor += size;
  }
  return offsets;
}

S3Manager::S3Manager(const std::string& config_path,
                     std::size_t thread_count,
                     std::size_t preallocate_buffer_size,
                     std::size_t connection_pool_size)
    : thread_pool_(thread_count == 0 ? 256 : thread_count) {
  if (!load_s3_config(config_path, config_)) {
    throw std::runtime_error("Failed to load S3 config: " + config_path);
  }

  if (preallocate_buffer_size > 0) {
    void* raw = nullptr;
    const std::size_t alloc_size = align_up(preallocate_buffer_size, kBlockSize);
    cudaError_t err = cudaHostAlloc(&raw, alloc_size, cudaHostAllocDefault);
    if (err == cudaSuccess) {
      arena_buffer_ = std::shared_ptr<void>(raw, [](void* p) { cudaFreeHost(p); });
      arena_capacity_ = alloc_size;
      arena_offset_ = 0;
    } else if (posix_memalign(&raw, kBlockSize, alloc_size) == 0) {
      arena_buffer_ = std::shared_ptr<void>(raw, [](void* p) { std::free(p); });
      arena_capacity_ = alloc_size;
      arena_offset_ = 0;
      if (madvise(raw, alloc_size, MADV_POPULATE_WRITE) != 0) {
        std::memset(raw, 0, alloc_size);
      }
    }
  }

  curl_pool_size_.store(connection_pool_size);
  init_curl();
}

S3Manager::S3Manager(const S3Config& config,
                     std::size_t thread_count,
                     std::size_t preallocate_buffer_size,
                     std::size_t connection_pool_size)
    : config_(config),
      thread_pool_(thread_count == 0 ? 256 : thread_count) {
  if (preallocate_buffer_size > 0) {
    void* raw = nullptr;
    const std::size_t alloc_size = align_up(preallocate_buffer_size, kBlockSize);
    cudaError_t err = cudaHostAlloc(&raw, alloc_size, cudaHostAllocDefault);
    if (err == cudaSuccess) {
      arena_buffer_ = std::shared_ptr<void>(raw, [](void* p) { cudaFreeHost(p); });
      arena_capacity_ = alloc_size;
      arena_offset_ = 0;
    } else if (posix_memalign(&raw, kBlockSize, alloc_size) == 0) {
      arena_buffer_ = std::shared_ptr<void>(raw, [](void* p) { std::free(p); });
      arena_capacity_ = alloc_size;
      arena_offset_ = 0;
      if (madvise(raw, alloc_size, MADV_POPULATE_WRITE) != 0) {
        std::memset(raw, 0, alloc_size);
      }
    }
  }
  curl_pool_size_.store(connection_pool_size);
  init_curl();
}

S3Manager::~S3Manager() {
  cleanup_curl();
  if (gpu_arena_) {
    if (current_device_id_ >= 0) {
      cudaSetDevice(current_device_id_);
    }
    cudaFree(gpu_arena_);
    gpu_arena_ = nullptr;
  }
  arena_buffer_.reset();
  arena_capacity_ = 0;
  arena_offset_ = 0;
}

void S3Manager::init_curl() {
  std::call_once(g_curl_global_flag, []() {
    curl_global_init(CURL_GLOBAL_ALL);
  });

  std::lock_guard<std::mutex> lock(curl_pool_mutex_);
  const std::size_t desired = std::max<std::size_t>(1, curl_pool_size_.load());
  while (curl_pool_.size() < desired) {
    CURL* handle = curl_easy_init();
    if (!handle) {
      throw std::runtime_error("Failed to create CURL easy handle");
    }
    // TCP connection optimizations
    curl_easy_setopt(handle, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(handle, CURLOPT_TCP_KEEPIDLE, 60L);     // Start keepalive after 60s idle
    curl_easy_setopt(handle, CURLOPT_TCP_KEEPINTVL, 30L);    // Keepalive interval 30s
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_TCP_NODELAY, 1L);       // Disable Nagle's algorithm
    curl_easy_setopt(handle, CURLOPT_BUFFERSIZE, 256L * 1024L);
    // Connection reuse
    curl_easy_setopt(handle, CURLOPT_FORBID_REUSE, 0L);      // Allow connection reuse
    curl_easy_setopt(handle, CURLOPT_FRESH_CONNECT, 0L);     // Don't force new connections
    curl_pool_.push_back(handle);
  }
}

void S3Manager::cleanup_curl() {
  std::lock_guard<std::mutex> lock(curl_pool_mutex_);
  for (auto* handle : curl_pool_) {
    curl_easy_cleanup(handle);
  }
  curl_pool_.clear();
}

CURL* S3Manager::acquire_handle() {
  std::lock_guard<std::mutex> lock(curl_pool_mutex_);
  if (!curl_pool_.empty()) {
    CURL* handle = curl_pool_.back();
    curl_pool_.pop_back();
    return handle;
  }
  CURL* handle = curl_easy_init();
  if (!handle) {
    throw std::runtime_error("Failed to allocate CURL handle");
  }
  // TCP connection optimizations
  curl_easy_setopt(handle, CURLOPT_TCP_KEEPALIVE, 1L);
  curl_easy_setopt(handle, CURLOPT_TCP_KEEPIDLE, 60L);
  curl_easy_setopt(handle, CURLOPT_TCP_KEEPINTVL, 30L);
  curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(handle, CURLOPT_TCP_NODELAY, 1L);
  curl_easy_setopt(handle, CURLOPT_BUFFERSIZE, 256L * 1024L);
  // Connection reuse
  curl_easy_setopt(handle, CURLOPT_FORBID_REUSE, 0L);
  curl_easy_setopt(handle, CURLOPT_FRESH_CONNECT, 0L);
  return handle;
}

void S3Manager::release_handle(CURL* handle) {
  if (!handle) {
    return;
  }
  // Don't fully reset - preserve TCP connection state
  // Only clear request-specific options
  curl_easy_setopt(handle, CURLOPT_HTTPHEADER, nullptr);
  curl_easy_setopt(handle, CURLOPT_RANGE, nullptr);
  curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, nullptr);
  curl_easy_setopt(handle, CURLOPT_WRITEDATA, nullptr);
  curl_easy_setopt(handle, CURLOPT_READFUNCTION, nullptr);
  curl_easy_setopt(handle, CURLOPT_READDATA, nullptr);
  curl_easy_setopt(handle, CURLOPT_UPLOAD, 0L);
  curl_easy_setopt(handle, CURLOPT_NOBODY, 0L);
  curl_easy_setopt(handle, CURLOPT_CUSTOMREQUEST, nullptr);
  curl_easy_setopt(handle, CURLOPT_HTTPGET, 1L);  // Reset to GET
  curl_easy_setopt(handle, CURLOPT_INFILESIZE_LARGE, -1L);

  std::lock_guard<std::mutex> lock(curl_pool_mutex_);
  if (curl_pool_.size() < curl_pool_size_.load()) {
    curl_pool_.push_back(handle);
  } else {
    curl_easy_cleanup(handle);
  }
}

std::string S3Manager::build_url(const std::string& key) const {
  std::string normalized = key;
  if (!normalized.empty() && normalized.front() == '/') {
    normalized.erase(0, 1);
  }
  const std::string scheme = config_.use_ssl ? "https://" : "http://";
  std::string path = config_.bucket;
  if (!normalized.empty()) {
    path.push_back('/');
    path += uri_encode(normalized, false);
  }
  return scheme + config_.endpoint + "/" + path;
}

void S3Manager::sign_request(CURL* curl,
                             const std::string& method,
                             const std::string& key,
                             const std::string& content_sha256,
                             curl_slist** headers) {
  std::string normalized = key;
  if (!normalized.empty() && normalized.front() == '/') {
    normalized.erase(0, 1);
  }
  const std::string encoded_key = uri_encode(normalized, false);
  std::string canonical_uri = "/" + config_.bucket;
  if (!encoded_key.empty()) {
    canonical_uri.push_back('/');
    canonical_uri += encoded_key;
  }

  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  const std::tm tm = gmtime_utc(now_time);

  std::ostringstream ts_stream;
  ts_stream << std::put_time(&tm, "%Y%m%dT%H%M%SZ");
  const std::string amz_date = ts_stream.str();
  const std::string date = amz_date.substr(0, 8);

  std::string canonical_headers =
      "host:" + config_.endpoint + "\n" +
      "x-amz-content-sha256:" + content_sha256 + "\n" +
      "x-amz-date:" + amz_date + "\n";
  const std::string signed_headers = "host;x-amz-content-sha256;x-amz-date";
  const std::string canonical_request =
      method + "\n" +
      canonical_uri + "\n\n" +
      canonical_headers + "\n" +
      signed_headers + "\n" +
      content_sha256;

  const std::string scope = date + "/" + config_.region + "/s3/aws4_request";
  const std::string string_to_sign =
      "AWS4-HMAC-SHA256\n" + amz_date + "\n" + scope + "\n" + sha256_hex(canonical_request);

  // Use cached signing key if date matches (reduces 4 HMAC ops to 1)
  std::vector<unsigned char> k_signing;
  {
    std::lock_guard<std::mutex> lock(signing_key_mutex_);
    if (cached_date_ == date && !cached_signing_key_.empty()) {
      k_signing = cached_signing_key_;
    } else {
      std::string secret = "AWS4" + config_.secret_key;
      std::vector<unsigned char> k_date = hmac_sha256(
          reinterpret_cast<const unsigned char*>(secret.data()), secret.size(), date);
      std::vector<unsigned char> k_region = hmac_sha256(k_date, config_.region);
      std::vector<unsigned char> k_service = hmac_sha256(k_region, "s3");
      k_signing = hmac_sha256(k_service, "aws4_request");
      cached_date_ = date;
      cached_signing_key_ = k_signing;
    }
  }
  std::vector<unsigned char> signature_bytes = hmac_sha256(k_signing, string_to_sign);
  const std::string signature = to_hex(signature_bytes.data(), signature_bytes.size());

  curl_slist* list = headers ? *headers : nullptr;
  const std::string host_header = "Host: " + config_.endpoint;
  const std::string date_header = "x-amz-date: " + amz_date;
  const std::string hash_header = "x-amz-content-sha256: " + content_sha256;
  const std::string auth_header =
      "Authorization: AWS4-HMAC-SHA256 Credential=" + config_.access_key + "/" + scope +
      ", SignedHeaders=" + signed_headers + ", Signature=" + signature;

  list = curl_slist_append(list, host_header.c_str());
  list = curl_slist_append(list, date_header.c_str());
  list = curl_slist_append(list, hash_header.c_str());
  list = curl_slist_append(list, auth_header.c_str());
  if (headers) {
    *headers = list;
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
}

std::size_t S3Manager::download_range(const std::string& key,
                                      char* dest,
                                      std::size_t offset,
                                      std::size_t size) {
  if (size == 0) {
    return 0;
  }
  CURL* curl = acquire_handle();
  curl_slist* headers = nullptr;
  const std::string url = build_url(key);
  const std::size_t end = offset + size - 1;
  std::string range = "bytes=" + std::to_string(offset) + "-" + std::to_string(end);

  RangeWriter writer{dest, size, 0};
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &writer);
  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
  sign_request(curl, "GET", key, "UNSIGNED-PAYLOAD", &headers);

  CURLcode res = curl_easy_perform(curl);
  long response_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
  curl_slist_free_all(headers);
  release_handle(curl);

  if (res != CURLE_OK) {
    throw std::runtime_error("CURL range download failed: " + std::string(curl_easy_strerror(res)));
  }
  if (response_code != 200 && response_code != 206) {
    throw std::runtime_error("Unexpected HTTP code for range download: " + std::to_string(response_code));
  }
  if (writer.written < size) {
    throw std::runtime_error("Short read while downloading range");
  }
  return writer.written;
}

void S3Manager::parallel_download(const std::string& key, char* dest, std::size_t total_size) {
  if (total_size == 0) {
    return;
  }
  const std::size_t chunk_count = (total_size + kChunkSize - 1) / kChunkSize;
  if (chunk_count == 1) {
    download_range(key, dest, 0, total_size);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(chunk_count);
  std::exception_ptr first_error;
  std::mutex error_mutex;

  for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const std::size_t offset = chunk * kChunkSize;
    const std::size_t chunk_size = std::min(kChunkSize, total_size - offset);
    workers.emplace_back([&, offset, chunk_size]() {
      try {
        download_range(key, dest + offset, offset, chunk_size);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
          first_error = std::current_exception();
        }
      }
    });
  }

  for (auto& worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

void S3Manager::batch_download_curl_multi(const std::vector<std::string>& keys,
                                          const std::vector<char*>& destinations,
                                          const std::vector<std::size_t>& sizes) {
  if (keys.size() != destinations.size() || keys.size() != sizes.size()) {
    throw std::runtime_error("batch_download_curl_multi parameter mismatch");
  }
  if (keys.empty()) {
    return;
  }

  struct CurlMultiDeleter {
    void operator()(CURLM* handle) const noexcept {
      if (handle) {
        curl_multi_cleanup(handle);
      }
    }
  };

  std::unique_ptr<CURLM, CurlMultiDeleter> multi(curl_multi_init());
  if (!multi) {
    throw std::runtime_error("Failed to create CURLM handle");
  }

  struct DownloadState {
    RangeWriter writer{};
    curl_slist* headers = nullptr;
    CURL* handle = nullptr;
    std::string range;
    bool added = false;
  };

  std::vector<DownloadState> states(keys.size());
  std::unordered_map<CURL*, std::size_t> handle_to_index;
  handle_to_index.reserve(keys.size());
  std::vector<std::string> errors(keys.size());

  auto release_state = [&](std::size_t idx) {
    if (idx >= states.size()) {
      return;
    }
    if (states[idx].handle) {
      if (states[idx].added) {
        curl_multi_remove_handle(multi.get(), states[idx].handle);
        states[idx].added = false;
      }
      curl_slist_free_all(states[idx].headers);
      states[idx].headers = nullptr;
      release_handle(states[idx].handle);
      states[idx].handle = nullptr;
    } else if (states[idx].headers) {
      curl_slist_free_all(states[idx].headers);
      states[idx].headers = nullptr;
    }
  };

  auto release_all = [&]() {
    for (std::size_t i = 0; i < states.size(); ++i) {
      release_state(i);
    }
    handle_to_index.clear();
  };

  std::size_t active = 0;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (sizes[i] == 0) {
      continue;
    }
    states[i].writer = RangeWriter{destinations[i], sizes[i], 0};
    states[i].handle = acquire_handle();
    CURL* easy = states[i].handle;
    const std::string url = build_url(keys[i]);
    curl_easy_setopt(easy, CURLOPT_URL, url.c_str());
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &states[i].writer);
    curl_easy_setopt(easy, CURLOPT_HTTPGET, 1L);
    states[i].range = "bytes=0-" + std::to_string(sizes[i] - 1);
    curl_easy_setopt(easy, CURLOPT_RANGE, states[i].range.c_str());
    sign_request(easy, "GET", keys[i], "UNSIGNED-PAYLOAD", &states[i].headers);
    CURLMcode add_res = curl_multi_add_handle(multi.get(), easy);
    if (add_res != CURLM_OK) {
      release_all();
      throw std::runtime_error("curl_multi_add_handle failed: " +
                               std::string(curl_multi_strerror(add_res)));
    }
    states[i].added = true;
    handle_to_index[easy] = i;
    ++active;
  }

  if (active == 0) {
    return;
  }

  auto drain_messages = [&]() {
    int msgs_left = 0;
    while (CURLMsg* msg = curl_multi_info_read(multi.get(), &msgs_left)) {
      if (msg->msg != CURLMSG_DONE) {
        continue;
      }
      auto it = handle_to_index.find(msg->easy_handle);
      if (it == handle_to_index.end()) {
        continue;
      }
      const std::size_t idx = it->second;
      handle_to_index.erase(it);

      long response_code = 0;
      curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &response_code);
      if (msg->data.result != CURLE_OK) {
        errors[idx] = keys[idx] + ": download failed - " +
                      std::string(curl_easy_strerror(msg->data.result));
      } else if (response_code < 200 || response_code >= 300) {
        errors[idx] = keys[idx] + ": HTTP " + std::to_string(response_code);
      } else if (states[idx].writer.written < sizes[idx]) {
        errors[idx] = keys[idx] + ": short read (" +
                      std::to_string(states[idx].writer.written) + "/" +
                      std::to_string(sizes[idx]) + ")";
      }
      release_state(idx);
    }
  };

  auto perform = [&](int& still_running) {
    CURLMcode rc;
    do {
      rc = curl_multi_perform(multi.get(), &still_running);
    } while (rc == CURLM_CALL_MULTI_PERFORM);
    if (rc != CURLM_OK) {
      release_all();
      throw std::runtime_error("curl_multi_perform failed: " +
                               std::string(curl_multi_strerror(rc)));
    }
    drain_messages();
  };

  int still_running = 0;
  perform(still_running);
  while (still_running > 0) {
    CURLMcode poll_res = curl_multi_poll(multi.get(), nullptr, 0, 100, nullptr);
    if (poll_res != CURLM_OK) {
      release_all();
      throw std::runtime_error("curl_multi_poll failed: " +
                               std::string(curl_multi_strerror(poll_res)));
    }
    perform(still_running);
  }
  drain_messages();

  for (const auto& err : errors) {
    if (!err.empty()) {
      release_all();
      throw std::runtime_error(err);
    }
  }
  release_all();
}

void S3Manager::upload_object(const std::string& key, const char* data, std::size_t size) {
  CURL* curl = acquire_handle();
  curl_slist* headers = nullptr;
  const std::string url = build_url(key);

  UploadBuffer upload{data, size, 0};
  const std::string payload_hash = sha256_hex(data, size);

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);
  curl_easy_setopt(curl, CURLOPT_READDATA, &upload);
  curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
  curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(size));
  sign_request(curl, "PUT", key, payload_hash, &headers);

  CURLcode res = curl_easy_perform(curl);
  long response_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
  curl_slist_free_all(headers);
  release_handle(curl);

  if (res != CURLE_OK) {
    throw std::runtime_error("CURL upload failed: " + std::string(curl_easy_strerror(res)));
  }
  if (response_code < 200 || response_code >= 300) {
    throw std::runtime_error("Upload failed with HTTP " + std::to_string(response_code));
  }
}

char* S3Manager::allocate_from_arena(std::size_t required_size,
                                     std::shared_ptr<void>& out_owner) {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  const std::size_t aligned_size =
      required_size == 0 ? kBlockSize : align_up(required_size, kBlockSize);
  if (!arena_buffer_ || arena_capacity_ < aligned_size) {
    void* raw = nullptr;
    if (posix_memalign(&raw, kBlockSize, aligned_size) != 0) {
      throw std::runtime_error("Failed to allocate arena buffer");
    }
    out_owner = std::shared_ptr<void>(raw, [](void* p) { std::free(p); });
    return static_cast<char*>(raw);
  }

  if (arena_offset_ + aligned_size > arena_capacity_) {
    arena_offset_ = 0;
  }
  char* ptr = static_cast<char*>(arena_buffer_.get()) + arena_offset_;
  arena_offset_ += aligned_size;
  out_owner = arena_buffer_;
  return ptr;
}

void* S3Manager::allocate_from_gpu_arena(std::size_t size,
                                         int device_id,
                                         std::shared_ptr<void>& out_owner) {
  if (size == 0) {
    out_owner = nullptr;
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(gpu_arena_mutex_);
  return allocate_from_gpu_arena_unlocked(size, device_id, out_owner);
}

void* S3Manager::allocate_from_gpu_arena_unlocked(std::size_t size,
                                                  int device_id,
                                                  std::shared_ptr<void>& out_owner) {
  if (size == 0) {
    out_owner = nullptr;
    return nullptr;
  }

  const std::size_t aligned_size = align_up(size, kGpuAlignment);
  if (current_device_id_ != device_id || gpu_arena_capacity_ < aligned_size) {
    if (current_device_id_ >= 0 && current_device_id_ != device_id) {
      cudaSetDevice(current_device_id_);
    }
    if (gpu_arena_) {
      cudaFree(gpu_arena_);
      gpu_arena_ = nullptr;
    }
    const std::size_t alloc_size = std::max(kGpuArenaSize, aligned_size);
    cudaSetDevice(device_id);
    cudaError_t err = cudaMalloc(&gpu_arena_, alloc_size);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    gpu_arena_capacity_ = alloc_size;
    gpu_arena_offset_ = 0;
    current_device_id_ = device_id;
  }

  auto align_offset = [](std::size_t offset) {
    return (offset + kGpuAlignment - 1) & ~(kGpuAlignment - 1);
  };

  std::size_t alloc_offset = align_offset(gpu_arena_offset_);
  if (alloc_offset + aligned_size > gpu_arena_capacity_) {
    alloc_offset = 0;
    alloc_offset = align_offset(alloc_offset);
  }
  void* ptr = static_cast<char*>(gpu_arena_) + alloc_offset;
  gpu_arena_offset_ = alloc_offset + aligned_size;
  out_owner = nullptr;
  return ptr;
}

void S3Manager::save(const std::string& key, const TensorMap& tensors) {
  const std::string blob = serialize_tensor_map(tensors);
  upload_object(key, blob.data(), blob.size());
}

S3Manager::OptionalTensorMap S3Manager::load(const std::string& key) {
  auto results = batch_load(std::vector<std::string>{key});
  if (results.empty()) {
    return std::nullopt;
  }
  return std::move(results.front());
}

std::vector<S3Manager::OptionalTensorMap> S3Manager::batch_load(
    const std::vector<std::string>& keys) {
  return batch_load(keys, -1).first;
}

std::vector<S3Manager::OptionalTensorMap> S3Manager::batch_load_with_sizes(
    const std::vector<std::string>& keys,
    const std::vector<std::size_t>& sizes) {
  return batch_load(keys, sizes, -1).first;
}

S3Manager::BatchLoadResult S3Manager::batch_load(const std::vector<std::string>& keys,
                                                 int device_id) {
  if (keys.empty()) {
    return {};
  }
  std::vector<std::future<ProbeResult>> head_tasks;
  head_tasks.reserve(keys.size());
  for (const auto& key : keys) {
    head_tasks.emplace_back(thread_pool_.submit([this, key]() { return probe_object(key); }));
  }
  std::vector<std::size_t> sizes(keys.size());
  std::vector<bool> present_mask(keys.size(), false);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    ProbeResult probe = head_tasks[i].get();
    if (probe.status == ProbeStatus::kError) {
      throw std::runtime_error("HEAD failed for key " + keys[i] + ": " + probe.error);
    }
    if (probe.status == ProbeStatus::kPresent) {
      sizes[i] = probe.size;
      present_mask[i] = true;
    }
  }
  return batch_load_internal(keys, sizes, present_mask, device_id);
}

S3Manager::BatchLoadResult S3Manager::batch_load(const std::vector<std::string>& keys,
                                                 const std::vector<std::size_t>& sizes,
                                                 int device_id) {
  if (keys.size() != sizes.size()) {
    throw std::runtime_error("Keys/sizes length mismatch for batch_load");
  }
  if (keys.empty()) {
    return {};
  }
  std::vector<bool> present_mask(keys.size(), false);
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    present_mask[i] = sizes[i] > 0;
  }
  return batch_load_internal(keys, sizes, present_mask, device_id);
}

S3Manager::BatchLoadResult S3Manager::batch_load_internal(
    const std::vector<std::string>& keys,
    const std::vector<std::size_t>& sizes,
    const std::vector<bool>& present_mask,
    int device_id) {
  BatchLoadResult result;
  if (keys.empty()) {
    return result;
  }
  if (sizes.size() != keys.size() || present_mask.size() != keys.size()) {
    throw std::runtime_error("Keys/sizes/present_mask length mismatch for batch_load_internal");
  }

  const bool transfer_to_gpu = device_id >= 0;

  struct ObjectInfo {
    std::size_t size = 0;
    std::size_t offset = 0;
    std::size_t aligned_size = 0;
    std::size_t gpu_offset = 0;
  };

  std::vector<ObjectInfo> objects(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    objects[i].size = sizes[i];
    objects[i].aligned_size = align_up(sizes[i], kBlockSize);
  }

  std::size_t current_offset = 0;
  std::vector<std::size_t> gpu_object_sizes;
  gpu_object_sizes.reserve(objects.size());
  for (auto& obj : objects) {
    obj.offset = align_up(current_offset, kBlockSize);
    current_offset = obj.offset + obj.aligned_size;
    gpu_object_sizes.push_back(obj.size);
  }

  const auto gpu_offsets =
      ComputeAlignedObjectOffsets(gpu_object_sizes, kGpuAlignment);
  std::size_t gpu_total = 0;
  for (std::size_t i = 0; i < objects.size(); ++i) {
    objects[i].gpu_offset = gpu_offsets[i];
    gpu_total = std::max(gpu_total, objects[i].gpu_offset + objects[i].size);
  }

  const std::size_t allocation_size =
      current_offset == 0 ? kBlockSize : align_up(current_offset, kBlockSize);
  std::shared_ptr<void> buffer_owner;
  char* base_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (arena_buffer_ && arena_capacity_ >= allocation_size) {
      if (arena_offset_ + allocation_size > arena_capacity_) {
        arena_offset_ = 0;
      }
      base_ptr = static_cast<char*>(arena_buffer_.get()) + arena_offset_;
      arena_offset_ += allocation_size;
      buffer_owner = arena_buffer_;
    }
  }
  if (!base_ptr) {
    base_ptr = allocate_from_arena(allocation_size, buffer_owner);
  }

  std::vector<char*> destinations(keys.size(), nullptr);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    destinations[i] = base_ptr + objects[i].offset;
  }

  const bool has_downloads = std::any_of(
      present_mask.begin(), present_mask.end(), [](bool present) { return present; });
  if (has_downloads) {
    // Use thread pool for parallel downloads (faster than curl_multi for this workload)
    std::vector<std::future<void>> download_tasks;
    download_tasks.reserve(keys.size());
    std::exception_ptr first_error;
    std::mutex error_mutex;

    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (!present_mask[i] || sizes[i] == 0) continue;
      download_tasks.emplace_back(thread_pool_.submit([this, &keys, &destinations, &sizes, &first_error, &error_mutex, i]() {
        try {
          download_range(keys[i], destinations[i], 0, sizes[i]);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error) {
            first_error = std::current_exception();
          }
        }
      }));
    }

    for (auto& task : download_tasks) {
      task.get();
    }
    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }

  result.first.resize(keys.size());
  std::vector<std::future<std::string>> parse_tasks;
  parse_tasks.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    parse_tasks.emplace_back(thread_pool_.submit([&, i]() -> std::string {
      if (!present_mask[i] || objects[i].size == 0) {
        return std::string{};
      }
      try {
        result.first[i] = parse_from_memory_zero_copy(
            base_ptr + objects[i].offset,
            base_ptr + objects[i].offset + objects[i].size,
            buffer_owner);
      } catch (const std::exception& ex) {
        return keys[i] + ": parse failed - " + ex.what();
      }
      return std::string{};
    }));
  }

  for (auto& task : parse_tasks) {
    const std::string err = task.get();
    if (!err.empty()) {
      throw std::runtime_error(err);
    }
  }

  if (!transfer_to_gpu || gpu_total == 0) {
    result.second.clear();
    result.second.resize(keys.size());
    return result;
  }

  std::shared_ptr<void> gpu_owner;
  void* gpu_base = nullptr;
  {
    std::lock_guard<std::mutex> lock(gpu_arena_mutex_);
    gpu_base = allocate_from_gpu_arena_unlocked(gpu_total, device_id, gpu_owner);
  }

  std::vector<cudaStream_t> streams(keys.size(), nullptr);
  cudaSetDevice(device_id);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (objects[i].size == 0) {
      continue;
    }
    cudaError_t err = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamCreateWithFlags failed: ") +
                               cudaGetErrorString(err));
    }
  }

  auto destroy_streams = [&](int device) {
    if (device >= 0) {
      cudaSetDevice(device);
    }
    for (auto& stream : streams) {
      if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
      }
    }
  };

  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (objects[i].size == 0) {
      continue;
    }
    char* cpu_ptr = base_ptr + objects[i].offset;
    char* gpu_ptr = static_cast<char*>(gpu_base) + objects[i].gpu_offset;
    cudaError_t err = cudaMemcpyAsync(
        gpu_ptr,
        cpu_ptr,
        objects[i].size,
        cudaMemcpyHostToDevice,
        streams[i]);
    if (err != cudaSuccess) {
      destroy_streams(device_id);
      throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                               cudaGetErrorString(err));
    }
  }

  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (!streams[i]) continue;
    cudaError_t err = cudaStreamSynchronize(streams[i]);
    if (err != cudaSuccess) {
      destroy_streams(device_id);
      throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                               cudaGetErrorString(err));
    }
  }

  destroy_streams(device_id);

  result.second.resize(keys.size());
  torch::Device device(torch::kCUDA, device_id);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (!result.first[i].has_value()) {
      continue;
    }
    char* gpu_base_ptr = static_cast<char*>(gpu_base) + objects[i].gpu_offset;
    char* cpu_base_ptr = base_ptr + objects[i].offset;
    for (const auto& entry : *result.first[i]) {
      const auto& tensor = entry.second;
      auto& gpu_map = result.second[i];
      std::vector<int64_t> sizes;
      for (auto dim : tensor.shape) {
        sizes.push_back(static_cast<int64_t>(dim));
      }
      auto options = torch::TensorOptions().dtype(DTypeToTorchScalar(tensor.dtype)).device(device);
      if (tensor.data_size == 0 || !tensor.data_ptr()) {
        gpu_map[entry.first] = torch::empty(sizes, options);
        continue;
      }
      const char* tensor_ptr = reinterpret_cast<const char*>(tensor.data_ptr());
      const std::size_t tensor_offset =
          static_cast<std::size_t>(tensor_ptr - cpu_base_ptr);
      void* gpu_tensor_ptr = gpu_base_ptr + tensor_offset;
      auto owner = gpu_owner;
      gpu_map[entry.first] = torch::from_blob(
          gpu_tensor_ptr,
          sizes,
          [owner](void*) {},
          options);
    }
  }

  return result;
}

S3Manager::ProbeResult S3Manager::probe_object(const std::string& key) {
  CURL* curl = acquire_handle();
  curl_slist* headers = nullptr;
  const std::string url = build_url(key);

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "HEAD");
  sign_request(curl, "HEAD", key, "UNSIGNED-PAYLOAD", &headers);

  CURLcode res = curl_easy_perform(curl);
  long response_code = 0;
  curl_off_t content_length = -1;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
  curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);

  curl_slist_free_all(headers);
  release_handle(curl);

  if (res != CURLE_OK) {
    return ProbeResult{
        ProbeStatus::kError,
        0,
        std::string(curl_easy_strerror(res)),
    };
  }
  if (response_code == 404) {
    return ProbeResult{ProbeStatus::kMissing, 0, {}};
  }
  if (response_code < 200 || response_code >= 300) {
    return ProbeResult{
        ProbeStatus::kError,
        0,
        "HTTP " + std::to_string(response_code),
    };
  }
  if (content_length <= 0) {
    return ProbeResult{ProbeStatus::kMissing, 0, {}};
  }
  return ProbeResult{
      ProbeStatus::kPresent,
      static_cast<std::size_t>(content_length),
      {},
  };
}

std::size_t S3Manager::get_object_size(const std::string& key) {
  ProbeResult probe = probe_object(key);
  if (probe.status != ProbeStatus::kPresent) {
    return 0;
  }
  return probe.size;
}

bool S3Manager::exists(const std::string& key) {
  CURL* curl = acquire_handle();
  curl_slist* headers = nullptr;
  const std::string url = build_url(key);
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "HEAD");
  sign_request(curl, "HEAD", key, "UNSIGNED-PAYLOAD", &headers);
  CURLcode res = curl_easy_perform(curl);
  long response_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
  curl_slist_free_all(headers);
  release_handle(curl);
  if (res != CURLE_OK) {
    return false;
  }
  return response_code >= 200 && response_code < 300;
}

void S3Manager::remove(const std::string& key) {
  CURL* curl = acquire_handle();
  curl_slist* headers = nullptr;
  const std::string url = build_url(key);
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  sign_request(curl, "DELETE", key, "UNSIGNED-PAYLOAD", &headers);
  CURLcode res = curl_easy_perform(curl);
  long response_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
  curl_slist_free_all(headers);
  release_handle(curl);
  if (res != CURLE_OK) {
    throw std::runtime_error("CURL delete failed: " + std::string(curl_easy_strerror(res)));
  }
  if (response_code >= 400) {
    throw std::runtime_error("Delete failed with HTTP " + std::to_string(response_code));
  }
}

void S3Manager::warmup_connections(std::size_t num_connections) {
  // Default to connection pool size
  if (num_connections == 0) {
    num_connections = curl_pool_size_.load();
  }
  num_connections = std::min(num_connections, static_cast<std::size_t>(128));

  // Parallel HEAD requests to establish TCP connections
  std::vector<std::future<void>> tasks;
  tasks.reserve(num_connections);

  // Use a dummy key that will return 404 but still establishes connection
  const std::string warmup_url = build_url("__warmup__");

  auto warmup_one = [this, &warmup_url]() {
    CURL* curl = acquire_handle();
    curl_slist* headers = nullptr;

    curl_easy_setopt(curl, CURLOPT_URL, warmup_url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "HEAD");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1000L);  // 1 second timeout
    sign_request(curl, "HEAD", "__warmup__", "UNSIGNED-PAYLOAD", &headers);

    // Ignore result - we just want to establish TCP connection
    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    release_handle(curl);
  };

  for (std::size_t i = 0; i < num_connections; ++i) {
    tasks.emplace_back(thread_pool_.submit(warmup_one));
  }

  // Wait for all warmup requests to complete
  for (auto& task : tasks) {
    task.get();
  }
}

}  // namespace s3
