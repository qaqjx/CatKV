#!/usr/bin/env python3
import sys
import json


def main() -> int:
    try:
        import torch  # type: ignore
    except Exception as e:
        print("{\"error\": \"PyTorch not installed or failed to import\", \"detail\": %s}" % json.dumps(str(e)))
        return 1

    info: dict[str, object] = {}
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_runtime_version"] = getattr(torch.version, "cuda", None)

    # cuDNN info
    cudnn_version = None
    try:
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    info["cudnn_version"] = cudnn_version

    device_count = torch.cuda.device_count() if info["cuda_available"] else 0
    info["device_count"] = device_count

    devices: list[dict[str, object]] = []
    if info["cuda_available"]:
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            dev: dict[str, object] = {
                "index": i,
                "name": props.name,
                "major": props.major,
                "minor": props.minor,
                "sm_arch": f"sm_{props.major}{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                # total_memory from props is per-device total
                "total_memory_MB": round(props.total_memory / 1024 / 1024, 1),
            }
            # Try to get free/total memory via runtime query
            try:
                with torch.cuda.device(i):
                    free, total = torch.cuda.mem_get_info()
                dev["free_memory_MB"] = round(free / 1024 / 1024, 1)
                dev["runtime_total_memory_MB"] = round(total / 1024 / 1024, 1)
            except Exception:
                # mem_get_info may not be available in some environments
                pass
            devices.append(dev)
    info["devices"] = devices

    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
