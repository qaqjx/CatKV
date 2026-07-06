from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="catkv-ops",
    version="0.1.0",
    description="Offline GPU KV cache offload for torch.bfloat16 tensors",
    packages=["catkv_ops"],
    install_requires=["torch"],
    ext_modules=[
        CUDAExtension(
            name="catkv_ops._C",
            sources=[
                str(ROOT / "src" / "bindings.cpp"),
                str(ROOT / "src" / "cpu_memory_store.cu"),
                str(ROOT / "src" / "remote_pipeline.cpp"),
                str(ROOT / "src" / "compression_manager.cpp"),
                str(ROOT / "src" / "remote_storage_manager.cpp"),
                str(ROOT / "src" / "compressor.cpp"),
                str(ROOT / "src" / "compression_runner.cpp"),
                str(ROOT / "src" / "shared_key_sv_path.cpp"),
                str(ROOT / "src" / "s3_manager.cpp"),
                str(ROOT / "src" / "s3_schedule.cpp"),
                str(ROOT / "src" / "fused_dequant_kernel.cu"),
            ],
            include_dirs=[str(ROOT / "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
            libraries=["curl", "ssl", "crypto"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
