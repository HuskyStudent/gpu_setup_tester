# GPU Availability and Hugging Face Test Suite

This repository contains a set of scripts for testing GPU availability, PyTorch CUDA performance, and Hugging Face model loading with GPU acceleration.

## Scripts

- **GPUavail.py**  
  Checks if PyTorch can detect and utilize the GPU.

- **GPUavail2.py**  
  Tests CUDA performance with a large matrix multiplication to benchmark GPU speed.

- **HFavail.py**  
  Loads a small Hugging Face model (`gpt2`) and verifies GPU inference functionality.

- **HFavail2.py**  
  Alternative Hugging Face test using a custom or larger model (e.g., `Qwen`), useful for more realistic workloads.

## Run All Scripts
Need to install torch and HF
To run all GPU and Hugging Face tests in order, use the provided shell script:

```bash
bash tester.sh
