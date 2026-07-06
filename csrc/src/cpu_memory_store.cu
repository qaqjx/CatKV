#include "cpu_memory_store.h"
#include "remote_pipeline.h"

#include <cuda_runtime.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace catkv_ops {

namespace {

struct OffloadTicket {
  torch::Tensor source_tensor;
  torch::Tensor cpu_tensor;
  cudaEvent_t copy_done = nullptr;
};

struct PendingTensor {
  std::string name;
  torch::Tensor tensor;
  cudaEvent_t producer_ready = nullptr;
};

inline void CheckCuda(cudaError_t err, const char* message) {
  TORCH_CHECK(err == cudaSuccess, message, ": ", cudaGetErrorString(err));
}

void ValidateInputTensor(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.defined(), "tensor must be defined");
  TORCH_CHECK(tensor.is_cuda(), "tensor must be on CUDA");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kBFloat16,
      "tensor must have dtype torch.bfloat16");
}

PendingTensor PreparePendingTensor(
    const std::string& name,
    const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.defined(), "tensor must be defined");
  PendingTensor pending;
  pending.name = name;
  if (tensor.is_cuda()) {
    c10::cuda::CUDAGuard device_guard(tensor.device());
    pending.tensor = tensor.contiguous().clone();
    auto producer_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    CheckCuda(
        cudaEventCreateWithFlags(&pending.producer_ready, cudaEventDisableTiming),
        "failed to create producer ready event");
    CheckCuda(
        cudaEventRecord(pending.producer_ready, producer_stream.stream()),
        "failed to record producer ready event");
    return pending;
  }
  pending.tensor = tensor.contiguous();
  return pending;
}

void DestroyEventIfNeeded(cudaEvent_t* event) {
  if (*event != nullptr) {
    cudaEventDestroy(*event);
    *event = nullptr;
  }
}

OffloadTicket ScheduleOffload(
    const torch::Tensor& tensor,
    bool pin_memory,
    cudaEvent_t producer_ready = nullptr) {
  ValidateInputTensor(tensor);
  c10::cuda::CUDAGuard device_guard(tensor.device());

  auto copy_stream = c10::cuda::getStreamFromPool(false, tensor.device().index());
  cudaEvent_t done_event = nullptr;

  if (producer_ready != nullptr) {
    CheckCuda(
        cudaStreamWaitEvent(copy_stream.stream(), producer_ready, 0),
        "failed to wait on producer event");
  }

  auto source = tensor;
  auto options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kBFloat16).pinned_memory(pin_memory);
  auto cpu_tensor = torch::empty(source.sizes(), options);

  c10::cuda::CUDACachingAllocator::recordStream(source.storage().data_ptr(), copy_stream);
  CheckCuda(
      cudaMemcpyAsync(
          cpu_tensor.data_ptr(),
          source.data_ptr(),
          static_cast<size_t>(source.nbytes()),
          cudaMemcpyDeviceToHost,
          copy_stream.stream()),
      "failed to schedule D2H copy");
  CheckCuda(
      cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming),
      "failed to create completion event");
  CheckCuda(
      cudaEventRecord(done_event, copy_stream.stream()),
      "failed to record completion event");

  OffloadTicket ticket;
  ticket.source_tensor = std::move(source);
  ticket.cpu_tensor = std::move(cpu_tensor);
  ticket.copy_done = done_event;
  return ticket;
}

}  // namespace

torch::Tensor OffloadToCpu(const torch::Tensor& tensor, bool pin_memory) {
  auto ticket = ScheduleOffload(tensor, pin_memory);
  CheckCuda(
      cudaEventSynchronize(ticket.copy_done),
      "failed while waiting for CPU offload completion");
  CheckCuda(cudaEventDestroy(ticket.copy_done), "failed to destroy completion event");
  ticket.copy_done = nullptr;
  return ticket.cpu_tensor;
}

struct CPUMemoryStore::Impl {
  struct Entry {
    TensorMap data;
    bool failed = false;
    std::string error;
    mutable std::mutex mutex;
    std::condition_variable cv;
  };

  struct OffloadCPUJob {
    std::string path;
    std::string group_uuid;
    std::shared_ptr<Entry> entry;
    std::vector<PendingTensor> tensors;
  };

  explicit Impl(bool pin_memory_, size_t offload_workers_)
      : pin_memory(pin_memory_),
        offload_workers(offload_workers_) {
    TORCH_CHECK(offload_workers > 0, "offload worker count must be positive");
    workers.reserve(offload_workers);
    for (size_t idx = 0; idx < offload_workers; ++idx) {
      workers.emplace_back([this]() { WorkerLoop(); });
    }
  }

  ~Impl() { Shutdown(); }

  void Offload(
      const std::string& path,
      const TensorMap& data,
      const std::string& uuid) {
    TORCH_CHECK(!path.empty(), "path must not be empty");
    TORCH_CHECK(!data.empty(), "data must not be empty");

    for (const auto& [name, tensor] : data) {
      TORCH_CHECK(!name.empty(), "tensor name must not be empty");
      TORCH_CHECK(tensor.defined(), "tensor must be defined");
    }

    auto entry = std::make_shared<Entry>();
    bool all_cpu = true;
    for (const auto& [_, tensor] : data) {
      if (!tensor.device().is_cpu()) {
        all_cpu = false;
        break;
      }
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      TORCH_CHECK(!stopping, "store is shutting down");
      if (remote_pipeline != nullptr) {
        auto [_, inserted] = remote_seen_paths.insert(path);
        if (!inserted) {
          return;
        }
      }
      storage[path] = entry;
    }

    if (all_cpu) {
      TensorMap cpu_data;
      cpu_data.reserve(data.size());
      for (const auto& [name, tensor] : data) {
        cpu_data.emplace(name, tensor.contiguous());
      }
      {
        std::lock_guard<std::mutex> lock(entry->mutex);
        entry->data = std::move(cpu_data);
      }
      entry->cv.notify_all();
      SubmitRemote(path, entry, uuid);
      return;
    }

    std::vector<PendingTensor> pending;
    pending.reserve(data.size());
    for (const auto& [name, tensor] : data) {
      pending.push_back(PreparePendingTensor(name, tensor));
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      jobs.push(OffloadCPUJob{path, uuid, entry, std::move(pending)});
    }
    queue_cv.notify_one();
  }

  TensorMap Load(const std::string& path, const std::string& device) const {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = storage.find(path);
      if (it != storage.end()) {
        entry = it->second;
      }
    }

    // if key not in cpu memory , return the nullptr data
    if (entry == nullptr) {
      return {};
    }
    // if (entry == nullptr) {
    //   auto pipeline = [&]() {
    //     std::lock_guard<std::mutex> lock(mutex);
    //     return remote_pipeline;
    //   }();
    //   TORCH_CHECK(pipeline != nullptr, "path not found: ", path);
    //   return pipeline->Load(path, device);
    // }

    TensorMap data;
    {
      std::unique_lock<std::mutex> lock(entry->mutex);
      entry->cv.wait(lock, [&entry]() { return !entry->data.empty() || entry->failed; });
      TORCH_CHECK(!entry->failed, "offload failed for path ", path, ": ", entry->error);
      data = entry->data;
    }

    if (device == "cpu") {
      return data;
    }

    TensorMap loaded;
    loaded.reserve(data.size());
    for (const auto& [name, tensor] : data) {
      loaded.emplace(name, tensor.to(torch::Device(device)));
    }
    return loaded;
  }

  std::vector<TensorMap> LoadBatch(
      const std::vector<std::string>& paths,
      const std::string& device) const {
    std::vector<TensorMap> result;
    result.reserve(paths.size());
    for (const auto& path : paths) {
      try {
        result.push_back(Load(path, device));
      } catch (const c10::Error& error) {
        result.emplace_back();
      } catch (const std::runtime_error& error) {
        result.emplace_back();
      }
    }
    return result;
  }

  void EnableRemoteUpload(
      const std::string& config_path,
      double ratio,
      torch::ScalarType dtype,
      size_t num_workers,
      size_t max_queue_bytes,
      bool skip_remote_save) {
    std::lock_guard<std::mutex> lock(mutex);
    TORCH_CHECK(!stopping, "store is shutting down");
    TORCH_CHECK(remote_pipeline == nullptr, "remote upload is already enabled");
    remote_pipeline = std::make_shared<RemotePipeline>(
        config_path,
        ratio,
        dtype,
        num_workers,
        num_workers,
        max_queue_bytes,
        max_queue_bytes,
        skip_remote_save);
  }

  TensorMap LoadRemote(const std::string& path, const std::string& device) const {
    return GetRemotePipeline()->Load(path, device);
  }

  TensorMap WaitRemote(
      const std::string& path,
      const std::string& device,
      double timeout_seconds) const {
    return GetRemotePipeline()->Wait(path, device, timeout_seconds);
  }

  void WaitRemoteReady(
      const std::string& path,
      double timeout_seconds) const {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = storage.find(path);
      if (it != storage.end()) {
        entry = it->second;
      }
    }
    if (entry != nullptr) {
      std::unique_lock<std::mutex> lock(entry->mutex);
      entry->cv.wait(lock, [&entry]() { return !entry->data.empty() || entry->failed; });
      TORCH_CHECK(!entry->failed, "offload failed for path ", path, ": ", entry->error);
    }
    GetRemotePipeline()->WaitReady(path, timeout_seconds);
  }

  bool RemoteContains(const std::string& path) const {
    return GetRemotePipeline()->Contains(path);
  }

  void WaitRemoteAll(double timeout_seconds) const {
    GetRemotePipeline()->WaitAll(timeout_seconds);
  }

  size_t RemotePendingCount() const {
    return GetRemotePipeline()->PendingCount();
  }

  size_t RemoteCurrentQueueBytes() const {
    return GetRemotePipeline()->CurrentQueueBytes();
  }

  std::unordered_map<std::string, uint64_t> RemoteQueueStats() const {
    return GetRemotePipeline()->QueueStats();
  }

  void ResetRemoteQueueStats() const {
    GetRemotePipeline()->ResetQueueStats();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex);
    storage.clear();
    remote_seen_paths.clear();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return storage.size();
  }

  void Shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (stopping) {
        return;
      }
      stopping = true;
    }
    queue_cv.notify_all();
    for (auto& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  std::shared_ptr<Entry> Lookup(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = storage.find(path);
    TORCH_CHECK(it != storage.end(), "path not found: ", path);
    return it->second;
  }

  std::shared_ptr<RemotePipeline> GetRemotePipeline() const {
    std::lock_guard<std::mutex> lock(mutex);
    TORCH_CHECK(remote_pipeline != nullptr, "remote upload is not enabled");
    return remote_pipeline;
  }

  void SubmitRemote(
      const std::string& path,
      const std::shared_ptr<Entry>& entry,
      const std::string& group_uuid) {
    auto pipeline = [&]() {
      std::lock_guard<std::mutex> lock(mutex);
      return remote_pipeline;
    }();
    if (pipeline != nullptr) {
      TensorMap data;
      {
        std::lock_guard<std::mutex> lock(entry->mutex);
        data = entry->data;
      }
      std::weak_ptr<Entry> weak_entry = entry;
      pipeline->Submit(
          path,
          data,
          group_uuid,
          [this, path, weak_entry]() {
            auto strong_entry = weak_entry.lock();
            if (strong_entry == nullptr) {
              return;
            }
            std::lock_guard<std::mutex> lock(mutex);
            auto it = storage.find(path);
            if (it != storage.end() && it->second == strong_entry) {
              storage.erase(it);
            }
          },
          [this, path, weak_entry]() {
            auto strong_entry = weak_entry.lock();
            if (strong_entry == nullptr) {
              return;
            }
            std::lock_guard<std::mutex> lock(mutex);
            auto it = storage.find(path);
            if (it != storage.end() && it->second == strong_entry) {
              storage.erase(it);
            }
          });
    }
  }

  void WorkerLoop() {
    while (true) {
      OffloadCPUJob job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        queue_cv.wait(lock, [this]() { return stopping || !jobs.empty(); });
        if (jobs.empty()) {
          if (stopping) {
            return;
          }
          continue;
        }
        job = std::move(jobs.front());
        jobs.pop();
      }
      RunOffload(job.path, job.group_uuid, job.entry, std::move(job.tensors));
    }
  }

  void RunOffload(
      const std::string& path,
      const std::string& group_uuid,
      const std::shared_ptr<Entry>& entry,
      std::vector<PendingTensor> pending) {
    TensorMap cpu_data;
    cpu_data.reserve(pending.size());
    for (auto& item : pending) {
      torch::Tensor cpu_tensor;
      if (item.tensor.is_cuda()) {
        auto ticket = ScheduleOffload(item.tensor, pin_memory, item.producer_ready);
        DestroyEventIfNeeded(&item.producer_ready);
        CheckCuda(
            cudaEventSynchronize(ticket.copy_done),
            "failed while waiting for CPU offload completion");
        DestroyEventIfNeeded(&ticket.copy_done);
        cpu_tensor = std::move(ticket.cpu_tensor);
      } else {
        cpu_tensor = item.tensor.contiguous();
      }
      cpu_data.emplace(item.name, std::move(cpu_tensor));
    }
    {
      std::lock_guard<std::mutex> lock(entry->mutex);
      entry->data = std::move(cpu_data);
    }
    entry->cv.notify_all();
    SubmitRemote(path, entry, group_uuid);
  }

  bool pin_memory;
  size_t offload_workers;
  mutable std::mutex mutex;
  std::condition_variable queue_cv;
  std::unordered_map<std::string, std::shared_ptr<Entry>> storage;
  std::unordered_set<std::string> remote_seen_paths;
  std::queue<OffloadCPUJob> jobs;
  std::shared_ptr<RemotePipeline> remote_pipeline;
  std::vector<std::thread> workers;
  bool stopping = false;
};

CPUMemoryStore::CPUMemoryStore(bool pin_memory, size_t offload_workers)
    : impl_(std::make_shared<Impl>(pin_memory, offload_workers)) {}

CPUMemoryStore::~CPUMemoryStore() = default;

void CPUMemoryStore::Offload(
    const std::string& path,
    const TensorMap& data,
    const std::string& uuid) {
  impl_->Offload(path, data, uuid);
}

TensorMap CPUMemoryStore::Load(const std::string& path, const std::string& device) const {
  return impl_->Load(path, device);
}

std::vector<TensorMap> CPUMemoryStore::LoadBatch(
    const std::vector<std::string>& paths,
    const std::string& device) const {
  return impl_->LoadBatch(paths, device);
}

void CPUMemoryStore::EnableRemoteUpload(
    const std::string& config_path,
    double ratio,
    torch::ScalarType dtype,
    size_t num_workers,
    size_t max_queue_bytes,
    bool skip_remote_save) {
  impl_->EnableRemoteUpload(
      config_path, ratio, dtype, num_workers, max_queue_bytes, skip_remote_save);
}

TensorMap CPUMemoryStore::LoadRemote(const std::string& path, const std::string& device) const {
  return impl_->LoadRemote(path, device);
}

TensorMap CPUMemoryStore::WaitRemote(
    const std::string& path,
    const std::string& device,
    double timeout_seconds) const {
  return impl_->WaitRemote(path, device, timeout_seconds);
}

void CPUMemoryStore::WaitRemoteReady(
    const std::string& path,
    double timeout_seconds) const {
  impl_->WaitRemoteReady(path, timeout_seconds);
}

bool CPUMemoryStore::RemoteContains(const std::string& path) const {
  return impl_->RemoteContains(path);
}

void CPUMemoryStore::WaitRemoteAll(double timeout_seconds) const {
  impl_->WaitRemoteAll(timeout_seconds);
}

size_t CPUMemoryStore::RemotePendingCount() const {
  return impl_->RemotePendingCount();
}

size_t CPUMemoryStore::RemoteCurrentQueueBytes() const {
  return impl_->RemoteCurrentQueueBytes();
}

std::unordered_map<std::string, uint64_t> CPUMemoryStore::RemoteQueueStats() const {
  return impl_->RemoteQueueStats();
}

void CPUMemoryStore::ResetRemoteQueueStats() const {
  impl_->ResetRemoteQueueStats();
}

void CPUMemoryStore::Clear() {
  impl_->Clear();
}

size_t CPUMemoryStore::Size() const {
  return impl_->Size();
}

}  // namespace catkv_ops
