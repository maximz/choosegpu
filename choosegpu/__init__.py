"""Choose GPU."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.3"

import logging
import os
import platform
import sys
from typing import List, Tuple, Union, Optional
import math

logger = logging.getLogger(__name__)

# Set default logging handler to avoid "No handler found" warnings.
logger.addHandler(logging.NullHandler())


def _is_mac_silicon() -> bool:
    """Detect if running on Mac Silicon (Apple Silicon / ARM)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


# Override on module initialization for global configuration of which GPUs to favor.
# Ignored if gpu_device_ids parameter is set in configure_gpu(...) call.
preferred_gpu_ids: List[Union[int, Tuple[int, int]]] = []

# Track Mac Silicon GPU configuration state internally (since we don't use environment variables)
_mac_silicon_gpu_config: Optional[List[str]] = None


def get_available_gpus() -> List[Tuple[Union[int, Tuple[int, int]], str]]:
    """
    Detect which GPUs are free (allowing for ambient RAM usage and GPU utilization spikes -- unclear why those happen).
    Returns available GPUs as tuples of (GPU ID, GPU UUID). The UUID is a string. The ID is an int in the case of a single physical device, or a tuple of ints in MIG mode.

    On Mac Silicon, returns a single MPS device if available.
    On NVIDIA systems, uses NVML to detect available CUDA devices.
    """
    # Mac Silicon: Use MPS (Metal Performance Shaders) backend
    if _is_mac_silicon():
        # All Mac Silicon devices have GPUs, so we always return the MPS device
        # We don't check PyTorch availability here because choosegpu shouldn't require PyTorch
        # Use ID 0 and a special UUID to indicate MPS
        return [(0, "mps")]

    # NVIDIA GPUs: Use NVML
    # Note: nvitop is a regular dependency and should be installed, but we wrap in try/except
    # for graceful handling if something goes wrong with the installation
    try:
        from nvitop import Device, Snapshot
        from nvitop.api import libnvml
    except ImportError as e:
        logger.error(f"nvitop not available - cannot detect NVIDIA GPUs: {e}")
        return []

    def _is_available(gpu_device: Snapshot):
        return (
            gpu_device.memory_percent < 5
            and (
                math.isnan(gpu_device.memory_utilization)
                or gpu_device.memory_utilization < 5
            )
            and (
                math.isnan(gpu_device.gpu_utilization)
                or gpu_device.gpu_utilization < 10
            )
            and len(gpu_device.processes) == 0
        )

    # Use libnvml context manager to ensure nvmlShutdown() is called
    # Otherwise, we might have issues with multiprocessing:
    # If you call nvmlInit() in a parent process (called when running any qwuery),
    # the "initialized" global in the libnvml module will be set to True in forked child processes,
    # but the NVML library won't actually have been initialized in child processes,
    # leading to pynvml.NVML_ERROR_UNINITIALIZED errors.

    # Just for illustration, here's how we would do it without a context manager:
    # from pynvml import NVML_ERROR_UNINITIALIZED
    # try:
    #     libnvml.nvmlQuery('nvmlDeviceGetCount', ignore_errors=False)
    # except NVML_ERROR_UNINITIALIZED:
    #     libnvml.nvmlShutdown()
    #     libnvml.nvmlInit()
    try:
        with libnvml:
            gpu_devices = [
                leaf_device.as_snapshot()
                for device in Device.all()
                for leaf_device in device.to_leaf_devices()
            ]

            return [
                (gpu_device.index, gpu_device.uuid)
                for gpu_device in gpu_devices
                if _is_available(gpu_device)
            ]
    except Exception as e:
        logger.error(f"Failed to detect NVIDIA GPUs: {e}")
        return []


# Track whether configure_gpu has been called (regardless of whether enabled or disabled)
# TODO: Refactor this to avoid using a global variable.
# Instead make config a class with self methods.
# in __init__.py, set config = Configuration() which instantiates the class
# then malid.config.configure_gpu() can read/write to self.has_user_configured_gpus without requiring "global"
# see also https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py

has_user_configured_gpus = False


def are_gpu_settings_configured() -> bool:
    global has_user_configured_gpus
    # Check if user has called configure_gpu()
    # On Mac Silicon, we just check the global flag since there's no environment variable we set
    # On NVIDIA, we check both the flag and the CUDA env var
    if _is_mac_silicon():
        return has_user_configured_gpus
    else:
        return has_user_configured_gpus and "CUDA_VISIBLE_DEVICES" in os.environ.keys()


def get_gpu_config() -> Optional[List[str]]:
    """Get current GPU configuration.

    Returns list of GPU device IDs/UUIDs that are currently configured.
    On Mac Silicon, returns ["mps"] if GPU is enabled, ["-1"] if disabled, or None if not configured.
    On NVIDIA, returns CUDA device IDs/UUIDs from CUDA_VISIBLE_DEVICES.
    """
    global has_user_configured_gpus

    if _is_mac_silicon():
        # On Mac Silicon, we don't set environment variables
        # We need to track the state internally via a module-level variable
        if not has_user_configured_gpus:
            return None
        # Check the internal state - we'll store it in a module variable
        return _mac_silicon_gpu_config
    else:
        if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
            return None
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def is_gpu_enabled() -> Optional[bool]:
    """Check if GPU is enabled.

    Returns:
    - True: GPU is enabled (configured to use GPU)
    - False: GPU is disabled (configured to use CPU only)
    - None: GPU configuration has not been set yet

    This is a convenience method that interprets get_gpu_config() for you.
    """
    config = get_gpu_config()
    if config is None:
        return None
    if config == ["-1"]:
        return False
    # Any other value means GPU is enabled (["mps"] or GPU UUIDs)
    return True


def check_if_gpu_libraries_see_gpu() -> bool:
    """Check if PyTorch can see GPU hardware (CUDA or MPS).

    This checks hardware availability from PyTorch's perspective, which is different
    from checking choosegpu's configuration state (use is_gpu_enabled() for that).

    Important behavioral differences:
    - On Mac Silicon: Returns True if PyTorch MPS is available, REGARDLESS of
      configure_gpu() settings. Mac hardware cannot be "hidden" like CUDA can.
    - On NVIDIA: Returns True if CUDA is available AND not disabled by configure_gpu().
      When configure_gpu(enable=False) is called, CUDA_VISIBLE_DEVICES="-1" makes
      torch.cuda.is_available() return False.

    If PyTorch is not installed, returns False.

    If GPU settings have not been configured yet, this will call ensure_gpu_settings_configured()
    to default to CPU mode before checking availability.
    """
    # Ensure GPU settings are configured before checking (defaults to CPU if not set)
    # The reason this is important is we are about to import torch.
    # If we don't configure settings prior to importing torch, the import will by default cause using all GPUs.
    ensure_gpu_settings_configured()

    # Check if PyTorch is installed
    try:
        import torch
    except ImportError:
        return False

    if _is_mac_silicon():
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else:
        return torch.cuda.is_available()


def ensure_gpu_settings_configured() -> None:
    """
    This function ensures some GPU settings are configured:
    If the user has not yet called configure_gpu(), this will call it with configure_gpu(enable=False).
    Use this before importing Tensorflow or Torch to disable GPU usage if the user has not explicitly made a decision about GPUs.
    """
    return configure_gpu(enable=False, overwrite_existing_configuration=False)


def configure_gpu(
    enable: bool = True,
    desired_number_of_gpus: int = 1,
    memory_pool: bool = False,
    gpu_device_ids: Optional[List[Union[int, Tuple[int, int]]]] = None,
    overwrite_existing_configuration: bool = True,
) -> Optional[List[str]]:
    """GPU bootstrap: configures GPU device IDs.

    On NVIDIA systems: overwrites CUDA_VISIBLE_DEVICES env var. Run before importing Tensorflow/PyTorch.
    On Mac Silicon: configures PyTorch to use MPS (Metal Performance Shaders) backend.

    IMPORTANT BEHAVIORAL DIFFERENCE:
    - On NVIDIA: configure_gpu(enable=False) sets CUDA_VISIBLE_DEVICES="-1", which makes
                 torch.cuda.is_available() return False. GPU is truly disabled.
    - On Mac Silicon: configure_gpu(enable=False) will not prevent the hardware from being detectable.
                      Your code must check get_gpu_config() to determine whether to actually use the GPU.

    Arguments:
    - enable: enable or disable GPU
    - desired_number_of_gpus: number of GPUs to use (ignored on Mac Silicon, which has a single integrated GPU)
    - memory_pool: enable memory pooling (only works on NVIDIA GPUs with RMM)
    - gpu_device_ids: preferred GPU ID(s), will be chosen if available. The ID is an int in the case of a single physical device, or a tuple of ints in MIG mode. (NVIDIA only)
    - overwrite_existing_configuration: whether to overwrite existing GPU configuration

    Returns:
    - List of GPU device identifiers (e.g., ["mps"] on Mac Silicon, ["GPU-UUID"] on NVIDIA, ["-1"] when disabled)
    """

    global has_user_configured_gpus

    if are_gpu_settings_configured() and not overwrite_existing_configuration:
        # This may be innocuous - e.g. may be from a safety check call to ensure_gpu_settings_configured()
        logger.debug(
            "GPU settings already configured, ignoring call to configure_gpu(..., overwrite_existing_configuration=False)"
        )
        return None

    # preferred GPU IDs to use, if they're available
    if gpu_device_ids is None:
        gpu_device_ids = preferred_gpu_ids

    # Mac Silicon: Use MPS backend
    if _is_mac_silicon():
        global _mac_silicon_gpu_config

        if enable:
            # Note: Unlike NVIDIA CUDA, there's no environment variable to "enable" MPS.
            # PyTorch will automatically use MPS when available if the user moves tensors to the MPS device.
            # We just track the configuration state internally.
            # Note: We don't check if PyTorch is installed or has MPS support here - choosegpu just
            # tracks configuration state. If PyTorch isn't installed or doesn't support MPS,
            # that's the user's responsibility to handle.
            logger.info(
                "Configured Mac Silicon to use MPS (Metal Performance Shaders) backend"
            )

            if memory_pool:
                logger.warning(
                    "Memory pooling with RMM is not supported on Mac Silicon (NVIDIA-only). Ignoring memory_pool=True."
                )

            has_user_configured_gpus = True
            _mac_silicon_gpu_config = ["mps"]
            return ["mps"]
        else:
            # Disable GPU
            # IMPORTANT: On Mac Silicon, we cannot actually disable GPU at the PyTorch level like we can with CUDA.
            # torch.backends.mps.is_available() will still return True because the hardware is always available.
            # We just track the configuration state internally. Your code should check get_gpu_config()
            # to determine whether to use the GPU.
            # This is fundamentally different from NVIDIA CUDA, where setting CUDA_VISIBLE_DEVICES="-1"
            # makes torch.cuda.is_available() return False.
            logger.info("Disabled GPU on Mac Silicon")
            has_user_configured_gpus = True
            _mac_silicon_gpu_config = ["-1"]
            return ["-1"]

    # NVIDIA GPUs: Original CUDA logic
    if enable:
        # Detect which GPUs are free.
        # Sort in reverse order because we prefer to use higher-numbered GPUs (other people's code may try to grab GPU 0 by default)
        available_gpus: List[Tuple[Union[int, Tuple[int, int]], str]] = list(
            reversed(get_available_gpus())
        )

        if len(available_gpus) == 0:
            raise ValueError("No GPUs available.")

        # intersect available and preferred
        preferred_available_gpus = [
            gpu for gpu in available_gpus if gpu[0] in gpu_device_ids
        ]

        chosen_gpus = (
            preferred_available_gpus
            if len(preferred_available_gpus) > 0
            else available_gpus
        )

        # extract the number we want
        # note that MIG seems to be limited to one single MIG at a time, but doesn't hurt to provide multiple
        chosen_gpus = chosen_gpus[:desired_number_of_gpus]

        # extract UUID
        chosen_gpu_uuids: List[str] = [chosen_gpu[1] for chosen_gpu in chosen_gpus]

    else:
        # disable GPUs
        chosen_gpu_uuids: List[str] = ["-1"]

    # set env vars
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # format as comma-separated list of GPU UUIDs or indexes
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(chosen_gpu_uuids)

    if enable and memory_pool:
        import rmm
        import cupy as cp

        # use first device in GPU whitelist set in environment variables
        device_id = 0
        cp.cuda.Device(device_id).use()
        cp.cuda.runtime.setDevice(device_id)
        # cp.cuda.device.get_device_id()

        # pool allocator = preallocate memory? https://blog.blazingdb.com/gpu-memory-pools-and-performance-with-blazingsql-9034c427a591

        rmm.reinitialize(
            managed_memory=True,  # Allows oversubscription
            pool_allocator=True,  # False, # default is False
            devices=device_id,  # GPU device IDs to register
        )

        cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

        # cp.cuda.get_device_id()
        # cp.cuda.device.get_device_id()
        # cp.cuda.device.get_compute_capability()

    has_user_configured_gpus = True
    return chosen_gpu_uuids
