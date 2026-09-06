"""Fail closed unless PyTorch can execute tensor math on the RTX 5050."""

from __future__ import annotations

import json

import torch


def verify_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 5050" not in device_name:
        raise RuntimeError(f"expected RTX 5050, found {device_name}")
    left = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    right = torch.eye(4, dtype=torch.float32, device="cuda")
    result = torch.matmul(left, right)
    torch.cuda.synchronize()
    if not torch.equal(result.cpu(), left.cpu()):
        raise RuntimeError("CUDA matrix multiplication returned an unexpected result")
    details = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": True,
        "device_name": device_name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "matrix_sum": float(result.sum().item()),
    }
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return details


if __name__ == "__main__":
    verify_cuda()

