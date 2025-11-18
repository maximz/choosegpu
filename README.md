# Choose GPU

[![](https://img.shields.io/pypi/v/choosegpu.svg)](https://pypi.python.org/pypi/choosegpu)
[![CI](https://github.com/maximz/choosegpu/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/maximz/choosegpu/actions/workflows/ci.yaml)
[![](https://img.shields.io/badge/docs-here-blue.svg)](https://choosegpu.maximz.com)
[![](https://img.shields.io/github/stars/maximz/choosegpu?style=social)](https://github.com/maximz/choosegpu)

Automatically configure GPU usage for PyTorch and TensorFlow. Supports both NVIDIA CUDA GPUs and Apple Silicon (M series) MPS.

## Overview

`choosegpu` helps you manage GPU device selection and configuration before importing deep learning frameworks. It:

- **On NVIDIA systems**: Automatically detects available CUDA GPUs and sets `CUDA_VISIBLE_DEVICES` to use free GPUs
- **On Apple Silicon (Mac M series)**: Configures PyTorch to use the Metal Performance Shaders (MPS) backend
- Respects your preferences for which GPUs to use
- Can enable memory pooling on NVIDIA GPUs (requires RMM and CuPy)

**Important behavioral difference between platforms:**

- **NVIDIA CUDA**: When you call `configure_gpu(enable=False)`, it sets `CUDA_VISIBLE_DEVICES="-1"`, which makes `torch.cuda.is_available()` return `False`. The GPU is truly "hidden" from PyTorch/TensorFlow.

- **Apple Silicon MPS**: When you call `configure_gpu(enable=False)`, there is no effect because the hardware is always available. **The GPU is not actually disabled at the PyTorch level** - your code must check `choosegpu.get_gpu_config()` to determine whether to use it.

## Installation

```bash
pip install choosegpu
```

This installs `choosegpu` with all dependencies, including `nvitop` for NVIDIA GPU detection. On Mac Silicon, `nvitop` will be installed but not used (the library automatically detects the platform and uses MPS instead).

### Optional: Memory Pooling (NVIDIA only)

If you want to use `memory_pool=True` on NVIDIA GPUs, also install RMM and CuPy:

```bash
pip install rmm-cu11 cupy-cuda11x  # Adjust CUDA version as needed
```

## Usage

### Basic Usage

```python
import choosegpu

# Enable GPU (uses MPS on Mac Silicon, CUDA on NVIDIA)
choosegpu.configure_gpu(enable=True)

# Now import your deep learning framework
import torch

# The appropriate GPU backend will be configured
```

### Disable GPU (only effective on NVIDIA, not on Mac Silicon)

```python
import choosegpu

# Disable GPU - use CPU only
choosegpu.configure_gpu(enable=False)

import torch

# On Mac Silicon: check choosegpu.get_gpu_config() to see if GPU should not be used
```

### Check GPU Configuration

```python
import choosegpu

choosegpu.configure_gpu(enable=True)

# Get the current GPU configuration
gpu_config = choosegpu.get_gpu_config()
# Returns: ["mps"] on Mac Silicon, or ["GPU-UUID"] on NVIDIA
# Returns: ["-1"] when disabled
# Returns: None if not configured

# Check if GPU settings have been configured
if choosegpu.are_gpu_settings_configured():
    print(f"GPU configured: {gpu_config}")
```

### Advanced: Prefer Specific GPUs (NVIDIA only)

```python
import choosegpu

# Prefer specific GPU IDs if they're available
choosegpu.configure_gpu(enable=True, gpu_device_ids=[2, 3])

# Or set global preference
choosegpu.preferred_gpu_ids = [2, 3]
choosegpu.configure_gpu(enable=True)
```

### Memory Pooling (NVIDIA only)

```python
import choosegpu

# Enable memory pooling with RMM (requires rmm and cupy)
choosegpu.configure_gpu(enable=True, memory_pool=True)
```

## Platform Support

| Platform | GPU Backend | Hardware Detection | GPU Disable Behavior |
|----------|-------------|-------------------|---------------------|
| Apple Silicon (M series) | MPS (Metal Performance Shaders) | Always available if hardware supports it | Sets configuration flag, but `torch.backends.mps.is_available()` remains `True` |
| NVIDIA | CUDA | Uses `nvitop` to detect available GPUs | Sets `CUDA_VISIBLE_DEVICES="-1"`, making `torch.cuda.is_available()` return `False` |

## API Reference

### `configure_gpu(enable=True, desired_number_of_gpus=1, memory_pool=False, gpu_device_ids=None, overwrite_existing_configuration=True)`

Configure GPU usage. Must be called before importing PyTorch/TensorFlow.

**Parameters:**
- `enable` (bool): Enable or disable GPU
- `desired_number_of_gpus` (int): Number of GPUs to use (NVIDIA only, ignored on Mac Silicon)
- `memory_pool` (bool): Enable memory pooling with RMM (NVIDIA only)
- `gpu_device_ids` (list): Preferred GPU IDs to use if available (NVIDIA only)
- `overwrite_existing_configuration` (bool): Whether to overwrite existing GPU configuration

**Returns:** List of GPU device identifiers (e.g., `["mps"]` or `["GPU-UUID-123"]`)

### `get_gpu_config()`

Get current GPU configuration.

**Returns:**
- `["mps"]` if MPS is enabled on Mac Silicon
- `["-1"]` if GPU is disabled
- `["GPU-UUID", ...]` for NVIDIA GPUs
- `None` if not configured

### `are_gpu_settings_configured()`

Check if GPU settings have been configured by this library.

**Returns:** `True` if `configure_gpu()` has been called, `False` otherwise

## Development

Submit PRs against `develop` branch, then make a release pull request to `master`.

```bash
# Install requirements
pip install --upgrade pip wheel
pip install -r requirements_dev.txt

# Install local package
pip install -e .

# Install pre-commit
pre-commit install

# Run tests
make test

# Run lint
make lint

# bump version before submitting a PR against master (all master commits are deployed)
bump2version patch # possible: major / minor / patch

# also ensure CHANGELOG.md updated
```
