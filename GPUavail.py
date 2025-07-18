import torch

def check_gpu_available():
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No GPU available. Check your driver installation.")


def get_gpu_info():
    if not torch.cuda.is_available():
        print("❌ No GPU detected.")
        return

    print("🔍 GPU Info:")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

import time

def test_gpu_performance():
    if not torch.cuda.is_available():
        print("❌ No GPU available.")
        return

    print("🚀 Testing GPU matrix multiplication (4096x4096)...")
    device = torch.device("cuda")
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    print(f"✅ Matrix multiplication took {end - start:.4f} seconds.")

import subprocess

def run_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode()
        print("🖥️ nvidia-smi output:\n")
        print(output)
    except FileNotFoundError:
        print("❌ `nvidia-smi` not found. Make sure the NVIDIA driver is installed.")


def run_all_tests():
    check_gpu_available()
    get_gpu_info()
    test_gpu_performance()
    run_nvidia_smi()


run_all_tests()