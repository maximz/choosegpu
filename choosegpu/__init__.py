"""Choose GPU."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.1"

import logging
import os
from typing import List, Tuple, Union, Optional
import math

logger = logging.getLogger(__name__)

# Set default logging handler to avoid "No handler found" warnings.
logger.addHandler(logging.NullHandler())


# Override on module initialization for global configuration of which GPUs to favor.
# Ignored if gpu_device_ids parameter is set in configure_gpu(...) call.
preferred_gpu_ids: List[Union[int, Tuple[int, int]]] = []


def get_available_gpus() -> List[Tuple[Union[int, Tuple[int, int]], str]]:
    """
    Detect which GPUs are free (allowing for ambient RAM usage and GPU utilization spikes -- unclear why those happen).
    Returns available GPUs as tuples of (GPU ID, GPU UUID). The UUID is a string. The ID is an int in the case of a single physical device, or a tuple of ints in MIG mode.
    """
    # Get leaf devices (i.e. finds MIG devices in MIG mode, ignoring the parent device); take snapshot of current utilization
    from nvitop import Device, Snapshot
    from nvitop.api import libnvml

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


# Track whether configure_gpu has been called (regardless of whether enabled or disabled)
# TODO: Refactor this to avoid using a global variable.
# Instead make config a class with self methods.
# in __init__.py, set config = Configuration() which instantiates the class
# then malid.config.configure_gpu() can read/write to self.has_user_configured_gpus without requiring "global"
# see also https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py

has_user_configured_gpus = False

def are_gpu_settings_configured() -> bool:
    global has_user_configured_gpus
    # Check if user has called configure_gpu() and whether env vars were not set outside our code
    return has_user_configured_gpus and "CUDA_VISIBLE_DEVICES" in os.environ.keys()


def get_gpu_config() -> Optional[List[str]]:
    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        return None
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


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
    """GPU bootstrap: configures GPU device IDs (overwrites CUDA_VISIBLE_DEVICES env var). Run before importing Tensorflow.

    Arguments:
    - enable: enable or disable GPU
    - gpu_device_ids: preferred GPU ID(s), will be chosen if available. The ID is an int in the case of a single physical device, or a tuple of ints in MIG mode.
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
