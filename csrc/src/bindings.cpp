#include "cpu_memory_store.h"
#include "compressor.h"
#include "fused_dequant.h"
#include "s3_manager.h"
#include "s3_schedule.h"
#include "shared_key_sv_path.h"
#include "tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

disk_manager::DType TorchToDType(torch::ScalarType type) {
  switch (type) {
    case torch::kFloat32:
      return disk_manager::DType::FLOAT32;
    case torch::kFloat16:
      return disk_manager::DType::FLOAT16;
    case torch::kInt8:
      return disk_manager::DType::INT8;
    case torch::kUInt8:
      return disk_manager::DType::UINT8;
    case torch::kBFloat16:
      return disk_manager::DType::BFLOAT16;
    case torch::kInt32:
      return disk_manager::DType::INT32;
    case torch::kFloat64:
      return disk_manager::DType::FLOAT64;
    default:
      throw std::invalid_argument("Unsupported torch dtype supplied.");
  }
}

torch::ScalarType DTypeToTorch(disk_manager::DType dtype) {
  switch (dtype) {
    case disk_manager::DType::FLOAT32:
      return torch::kFloat32;
    case disk_manager::DType::FLOAT16:
      return torch::kFloat16;
    case disk_manager::DType::INT8:
      return torch::kInt8;
    case disk_manager::DType::UINT8:
      return torch::kUInt8;
    case disk_manager::DType::BFLOAT16:
      return torch::kBFloat16;
    case disk_manager::DType::INT32:
      return torch::kInt32;
    case disk_manager::DType::FLOAT64:
      return torch::kFloat64;
  }
  throw std::runtime_error("Unsupported DType");
}

disk_manager::Tensor TensorToCpp(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    throw std::invalid_argument("Expected a defined torch.Tensor.");
  }
  torch::Tensor normalized = tensor.detach().cpu().contiguous();
  const disk_manager::DType dtype = TorchToDType(normalized.scalar_type());

  std::vector<std::size_t> shape;
  shape.reserve(static_cast<std::size_t>(normalized.dim()));
  for (int64_t i = 0; i < normalized.dim(); ++i) {
    shape.push_back(static_cast<std::size_t>(normalized.size(i)));
  }

  const std::size_t bytes = static_cast<std::size_t>(normalized.element_size()) *
                            static_cast<std::size_t>(normalized.numel());
  return disk_manager::Tensor::from_raw(dtype, std::move(shape), normalized.data_ptr(), bytes);
}

torch::Tensor CppToTensor(const disk_manager::Tensor& tensor) {
  const torch::ScalarType scalar_type = DTypeToTorch(tensor.dtype);
  std::vector<int64_t> sizes;
  sizes.reserve(tensor.shape.size());
  for (std::size_t dim : tensor.shape) {
    sizes.push_back(static_cast<int64_t>(dim));
  }
  auto options = torch::TensorOptions().dtype(scalar_type).device(torch::kCPU);

  if (tensor.data_size == 0 || !tensor.data_ptr()) {
    return torch::empty(sizes, options);
  }

  if (tensor.external_owner_) {
    std::shared_ptr<void> owner = tensor.external_owner_;
    torch::Tensor view = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(tensor.data_ptr())),
        torch::IntArrayRef(sizes),
        [owner](void*) {},
        options);
    return view;
  }

  torch::Tensor view = torch::from_blob(
      const_cast<void*>(static_cast<const void*>(tensor.data_ptr())),
      torch::IntArrayRef(sizes),
      [](void*) {},
      options);
  return view.clone();
}

s3::S3Manager::TensorMap DictToTensorMap(const py::dict& tensors_dict) {
  s3::S3Manager::TensorMap tensors;
  tensors.reserve(tensors_dict.size());
  for (auto item : tensors_dict) {
    const std::string key = py::cast<std::string>(item.first);
    torch::Tensor tensor = py::cast<torch::Tensor>(item.second);
    tensors.emplace(key, TensorToCpp(tensor));
  }
  return tensors;
}

py::dict TensorMapToDict(const s3::S3Manager::TensorMap& tensor_map) {
  py::dict result;
  for (const auto& [key, tensor] : tensor_map) {
    result[py::str(key)] = py::cast(CppToTensor(tensor));
  }
  return result;
}

py::object OptionalTensorMapToPyObject(const s3::S3Manager::OptionalTensorMap& tensor_map) {
  if (!tensor_map.has_value()) {
    return py::none();
  }
  return TensorMapToDict(*tensor_map);
}

py::list OptionalTensorMapsToPyList(
    const std::vector<s3::S3Manager::OptionalTensorMap>& tensor_maps) {
  py::list results(static_cast<py::ssize_t>(tensor_maps.size()));
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(tensor_maps.size()); ++i) {
    results[i] = OptionalTensorMapToPyObject(tensor_maps[static_cast<std::size_t>(i)]);
  }
  return results;
}

std::vector<std::string> SequenceToKeys(const py::sequence& seq) {
  std::vector<std::string> keys;
  keys.reserve(static_cast<std::size_t>(seq.size()));
  for (py::ssize_t i = 0; i < seq.size(); ++i) {
    keys.emplace_back(py::cast<std::string>(seq[i]));
  }
  return keys;
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "Offline GPU bf16 KV cache offload to CPU";

  py::module_::import("torch");

  m.def(
      "shared_key_sv_path",
      &catkv_ops::SharedKeySvPath,
      py::arg("uuid"),
      py::arg("layer_idx"),
      py::arg("base_path"));

  py::class_<disk_manager::CatKVCompressor>(m, "CatKVCompressor")
      .def(
          py::init<double, torch::ScalarType>(),
          py::arg("ratio") = 0.2,
          py::arg("dtype") = torch::kBFloat16)
      .def("compress", &disk_manager::CatKVCompressor::compress, py::arg("data"))
      .def(
          "compress_multi",
          &disk_manager::CatKVCompressor::compress_multi,
          py::arg("data"),
          py::arg("indices"),
          py::arg("uuids") = std::vector<std::string>{});

  py::class_<catkv_ops::CPUMemoryStore>(m, "CPUMemoryStore")
      .def(
          py::init<bool, size_t>(),
          py::arg("pin_memory") = true,
          py::arg("offload_workers") = 1)
      .def(
          "offload",
          &catkv_ops::CPUMemoryStore::Offload,
          py::arg("path"),
          py::arg("data"),
          py::arg("uuid"),
          py::call_guard<py::gil_scoped_release>())
      .def("load", &catkv_ops::CPUMemoryStore::Load, py::arg("path"), py::arg("device") = "cpu")
      .def(
          "load_batch",
          &catkv_ops::CPUMemoryStore::LoadBatch,
          py::arg("paths"),
          py::arg("device") = "cpu",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "enable_remote_upload",
          &catkv_ops::CPUMemoryStore::EnableRemoteUpload,
          py::arg("config_path"),
          py::arg("ratio") = 0.2,
          py::arg("dtype") = torch::kBFloat16,
          py::arg("num_workers") = 1,
          py::arg("max_queue_bytes") = 0,
          py::arg("skip_remote_save") = false)
      .def(
          "load_remote",
          &catkv_ops::CPUMemoryStore::LoadRemote,
          py::arg("path"),
          py::arg("device") = "cpu",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait_remote",
          &catkv_ops::CPUMemoryStore::WaitRemote,
          py::arg("path"),
          py::arg("device") = "cpu",
          py::arg("timeout_seconds") = -1.0,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait_remote_ready",
          &catkv_ops::CPUMemoryStore::WaitRemoteReady,
          py::arg("path"),
          py::arg("timeout_seconds") = -1.0,
          py::call_guard<py::gil_scoped_release>())
      .def("remote_contains", &catkv_ops::CPUMemoryStore::RemoteContains, py::arg("path"))
      .def(
          "wait_remote_all",
          &catkv_ops::CPUMemoryStore::WaitRemoteAll,
          py::arg("timeout_seconds") = -1.0,
          py::call_guard<py::gil_scoped_release>())
      .def("remote_pending_count", &catkv_ops::CPUMemoryStore::RemotePendingCount)
      .def(
          "remote_current_queue_bytes",
          &catkv_ops::CPUMemoryStore::RemoteCurrentQueueBytes)
      .def("remote_queue_stats", &catkv_ops::CPUMemoryStore::RemoteQueueStats)
      .def("reset_remote_queue_stats", &catkv_ops::CPUMemoryStore::ResetRemoteQueueStats)
      .def("clear", &catkv_ops::CPUMemoryStore::Clear)
      .def("__len__", &catkv_ops::CPUMemoryStore::Size);

  py::class_<s3::S3Manager>(m, "S3Manager")
      .def(py::init<const std::string&>(), py::arg("config_path"))
      .def(
          "save",
          [](s3::S3Manager& self, const std::string& key, const py::dict& tensors_dict) {
            auto tensors = DictToTensorMap(tensors_dict);
            py::gil_scoped_release release;
            self.save(key, tensors);
          },
          py::arg("key"),
          py::arg("tensors"))
      .def(
          "load",
          [](s3::S3Manager& self, const std::string& key) {
            s3::S3Manager::OptionalTensorMap tensors;
            {
              py::gil_scoped_release release;
              tensors = self.load(key);
            }
            return OptionalTensorMapToPyObject(tensors);
          },
          py::arg("key"))
      .def(
          "batch_load",
          [](s3::S3Manager& self, const py::sequence& keys) {
            auto cpp_keys = SequenceToKeys(keys);
            std::vector<s3::S3Manager::OptionalTensorMap> tensor_maps;
            {
              py::gil_scoped_release release;
              tensor_maps = self.batch_load(cpp_keys);
            }
            py::list results(static_cast<py::ssize_t>(tensor_maps.size()));
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(tensor_maps.size()); ++i) {
              results[i] = OptionalTensorMapToPyObject(
                  tensor_maps[static_cast<std::size_t>(i)]);
            }
            return results;
          },
          py::arg("keys"))
      .def(
          "batch_load_to_gpu",
          [](s3::S3Manager& self, const py::sequence& keys, int device_id) {
            auto cpp_keys = SequenceToKeys(keys);
            auto result = self.batch_load(cpp_keys, device_id);

            py::list cpu_results(static_cast<py::ssize_t>(result.first.size()));
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.first.size()); ++i) {
              cpu_results[i] = OptionalTensorMapToPyObject(
                  result.first[static_cast<std::size_t>(i)]);
            }

            py::list gpu_results(static_cast<py::ssize_t>(result.second.size()));
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.second.size()); ++i) {
              py::dict gpu_dict;
              for (const auto& [key, tensor] : result.second[static_cast<std::size_t>(i)]) {
                gpu_dict[py::str(key)] = py::cast(tensor);
              }
              gpu_results[i] = gpu_dict;
            }
            return py::make_tuple(cpu_results, gpu_results);
          },
          py::arg("keys"),
          py::arg("device_id") = 0)
      .def(
          "get_object_size",
          [](s3::S3Manager& self, const std::string& key) {
            py::gil_scoped_release release;
            return self.get_object_size(key);
          },
          py::arg("key"))
      .def(
          "exists",
          [](s3::S3Manager& self, const std::string& key) {
            py::gil_scoped_release release;
            return self.exists(key);
          },
          py::arg("key"))
      .def(
          "remove",
          [](s3::S3Manager& self, const std::string& key) {
            py::gil_scoped_release release;
            self.remove(key);
          },
          py::arg("key"))
      .def(
          "warmup_connections",
          [](s3::S3Manager& self, std::size_t num_connections) {
            py::gil_scoped_release release;
            self.warmup_connections(num_connections);
          },
          py::arg("num_connections") = 0);

  py::class_<s3::S3Schedule>(m, "S3Schedule")
      .def(py::init<const std::string&>(), py::arg("config_path"))
      .def(py::init<const std::string&, int>(), py::arg("config_path"), py::arg("num_threads"))
      .def(
          "submit_load",
          [](s3::S3Schedule& self, const std::string& key) {
            py::gil_scoped_release release;
            return self.submit_load(key);
          },
          py::arg("key"))
      .def(
          "submit_load_to_gpu",
          [](s3::S3Schedule& self, const std::string& key, int device_id) {
            py::gil_scoped_release release;
            return self.submit_load_to_gpu(key, device_id);
          },
          py::arg("key"),
          py::arg("device_id") = 0)
      .def(
          "submit_save",
          [](s3::S3Schedule& self, const std::string& key, const py::dict& tensors_dict) {
            auto tensors = DictToTensorMap(tensors_dict);
            py::gil_scoped_release release;
            return self.submit_save(key, tensors);
          },
          py::arg("key"),
          py::arg("tensors"))
      .def(
          "submit_multi_save",
          [](s3::S3Schedule& self,
             const py::sequence& paths,
             const std::vector<std::pair<int64_t, int64_t>>& indices,
             const py::list& data_list,
             double ratio,
             const std::vector<std::string>& uuids) {
            std::vector<std::string> cpp_paths;
            cpp_paths.reserve(static_cast<std::size_t>(paths.size()));
            for (py::ssize_t i = 0; i < paths.size(); ++i) {
              cpp_paths.emplace_back(py::cast<std::string>(paths[i]));
            }

            std::vector<torch::Tensor> data;
            data.reserve(static_cast<std::size_t>(data_list.size()));
            for (auto item : data_list) {
              data.push_back(py::cast<torch::Tensor>(item));
            }

            py::gil_scoped_release release;
            return self.submit_multi_save(cpp_paths, indices, data, ratio, uuids);
          },
          py::arg("paths"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("ratio") = 0.2,
          py::arg("uuids") = std::vector<std::string>{})
      .def(
          "submit_batch_load",
          [](s3::S3Schedule& self, const py::sequence& keys) {
            auto cpp_keys = SequenceToKeys(keys);
            py::gil_scoped_release release;
            return self.submit_batch_load(cpp_keys);
          },
          py::arg("keys"))
      .def(
          "submit_batch_load_to_gpu",
          [](s3::S3Schedule& self, const py::sequence& keys, int device_id) {
            auto cpp_keys = SequenceToKeys(keys);
            py::gil_scoped_release release;
            return self.submit_batch_load_to_gpu(cpp_keys, device_id);
          },
          py::arg("keys"),
          py::arg("device_id") = 0)
      .def(
          "get_load_result",
          [](s3::S3Schedule& self, int task_id) {
            s3::S3Manager::OptionalTensorMap tensors;
            {
              py::gil_scoped_release release;
              tensors = self.get_load_result(task_id);
            }
            return OptionalTensorMapToPyObject(tensors);
          },
          py::arg("task_id"))
      .def(
          "get_batch_load_result",
          [](s3::S3Schedule& self, int task_id) {
            std::vector<s3::S3Manager::OptionalTensorMap> tensor_maps;
            {
              py::gil_scoped_release release;
              tensor_maps = self.get_batch_load_result(task_id);
            }
            return OptionalTensorMapsToPyList(tensor_maps);
          },
          py::arg("task_id"))
      .def(
          "get_batch_load_to_gpu_result",
          [](s3::S3Schedule& self, int task_id) {
            s3::S3Manager::BatchLoadResult result;
            {
              py::gil_scoped_release release;
              result = self.get_batch_load_to_gpu_result(task_id);
            }

            py::list cpu_results = OptionalTensorMapsToPyList(result.first);
            py::list gpu_results(static_cast<py::ssize_t>(result.second.size()));
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.second.size()); ++i) {
              py::dict gpu_dict;
              for (const auto& [key, tensor] : result.second[static_cast<std::size_t>(i)]) {
                gpu_dict[py::str(key)] = py::cast(tensor);
              }
              gpu_results[i] = gpu_dict;
            }

            return py::make_tuple(cpu_results, gpu_results);
          },
          py::arg("task_id"))
      .def(
          "is_ready",
          [](s3::S3Schedule& self, int task_id) {
            py::gil_scoped_release release;
            return self.is_ready(task_id);
          },
          py::arg("task_id"))
      .def(
          "wait",
          [](s3::S3Schedule& self, int task_id) {
            py::gil_scoped_release release;
            self.wait(task_id);
          },
          py::arg("task_id"))
      .def(
          "get_status",
          [](s3::S3Schedule& self, int task_id) {
            py::gil_scoped_release release;
            return self.get_status(task_id);
          },
          py::arg("task_id"));

  m.def(
      "offload_to_cpu",
      &catkv_ops::OffloadToCpu,
      py::arg("tensor"),
      py::arg("pin_memory") = true,
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "fused_dequant_u_transposed",
      &fused_dequant_u_transposed,
      py::arg("u_quantized"),
      py::arg("u_meta"),
      py::arg("kv_len"),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "fused_dequant_v_residual",
      &fused_dequant_v_residual,
      py::arg("v_quantized"),
      py::arg("v_meta"),
      py::arg("key_residual"),
      py::arg("val_residual"),
      py::arg("hidden_dim"),
      py::call_guard<py::gil_scoped_release>());
}
