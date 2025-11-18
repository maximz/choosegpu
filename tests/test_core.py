import pytest
import os
from collections import namedtuple
import dataclasses
from typing import Union, Tuple
import copy
import math
import choosegpu


def _save_gpu_config():
    # get original values
    return (
        os.environ.get("CUDA_VISIBLE_DEVICES", None),
        copy.copy(choosegpu.preferred_gpu_ids),
        copy.copy(choosegpu._mac_silicon_gpu_config),
        choosegpu.has_user_configured_gpus,
    )


def _restore_gpu_config(original_values):
    # Restore CUDA env var (may not exist on Mac Silicon)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if original_values[0] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_values[0]

    # Restore module state
    choosegpu.preferred_gpu_ids = original_values[1]
    choosegpu._mac_silicon_gpu_config = original_values[2]
    choosegpu.has_user_configured_gpus = original_values[3]


@pytest.fixture
def mocked_pynvml(mocker):
    # Mock the nvidia library calls
    # libnvml context manager's call of nvmlInit will fail in CI
    # https://stackoverflow.com/q/28850070/130164
    mocker.patch("nvitop.api.libnvml").return_value.__enter__.return_value = None


def test_gpu_ids(mocker, mocked_pynvml):
    # This test is an exception to the rule of "don't call config.configure_gpu() in tests; let conftest.py handle it"
    # But we restore the environment variable after the test.

    # Mock platform detection to test NVIDIA GPU selection logic even on Mac
    mocker.patch("choosegpu._is_mac_silicon", return_value=False)

    original_values = _save_gpu_config()

    @dataclasses.dataclass
    class MockGpuDevice:
        def as_snapshot(self):
            return self

        def to_leaf_devices(self):
            return [self]

        index: Union[int, Tuple[int, int]]
        uuid: str
        memory_percent: float
        memory_utilization: float
        gpu_utilization: float
        processes: dict

    # preferred is available, but lower order than other available
    choosegpu.preferred_gpu_ids = [2]
    mocker.patch(
        "nvitop.Device.all",
        return_value=[
            MockGpuDevice(
                index=2,
                uuid="gpu-2",
                memory_percent=0,
                memory_utilization=math.nan,
                gpu_utilization=math.nan,
                processes={},
            ),
            MockGpuDevice(
                index=3,
                uuid="gpu-3",
                memory_percent=0,
                memory_utilization=0,
                gpu_utilization=0,
                processes={},
            ),
        ],
    )
    choosegpu.configure_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "gpu-2"

    # argument overrides global preferred list
    mocker.patch(
        "nvitop.Device.all",
        return_value=[
            MockGpuDevice(
                index=(1, 1),
                uuid="gpu-1",
                memory_percent=0,
                memory_utilization=math.nan,
                gpu_utilization=math.nan,
                processes={},
            ),
            MockGpuDevice(
                index=2,
                uuid="gpu-2",
                memory_percent=0,
                memory_utilization=math.nan,
                gpu_utilization=math.nan,
                processes={},
            ),
            MockGpuDevice(
                index=3,
                uuid="gpu-3",
                memory_percent=0,
                memory_utilization=0,
                gpu_utilization=0,
                processes={},
            ),
        ],
    )
    choosegpu.configure_gpu(gpu_device_ids=[(1, 1)])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "gpu-1"

    # preferred is unavailable
    mocker.patch(
        "nvitop.Device.all",
        return_value=[
            MockGpuDevice(
                index=3,
                uuid="gpu-3",
                memory_percent=0,
                memory_utilization=0,
                gpu_utilization=0,
                processes={},
            )
        ],
    )
    choosegpu.configure_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "gpu-3"

    # highest index order is chosen among available
    mocker.patch(
        "nvitop.Device.all",
        return_value=[
            MockGpuDevice(
                index=3,
                uuid="gpu-3",
                memory_percent=0,
                memory_utilization=0,
                gpu_utilization=0,
                processes={},
            ),
            MockGpuDevice(
                index=4,
                uuid="gpu-4",
                memory_percent=0,
                memory_utilization=0,
                gpu_utilization=0,
                processes={},
            ),
        ],
    )
    choosegpu.configure_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "gpu-4"

    # clean up
    _restore_gpu_config(original_values)


@pytest.mark.xfail(raises=ValueError)
def test_gpu_ids_none_available(mocker, mocked_pynvml):
    # This test is an exception to the rule of "don't call config.configure_gpu() in tests; let conftest.py handle it"
    # But we restore the environment variable after the test.

    # Mock platform detection to test NVIDIA GPU selection logic even on Mac
    mocker.patch("choosegpu._is_mac_silicon", return_value=False)

    original_values = _save_gpu_config()

    mocker.patch("nvitop.Device.all", return_value=[])
    choosegpu.configure_gpu()

    # clean up
    _restore_gpu_config(original_values)


@pytest.mark.skipif(not choosegpu._is_mac_silicon(), reason="Only runs on Mac Silicon")
def test_mac_silicon_mps_support():
    """Test that MPS support is detected correctly on Mac Silicon with PyTorch installed."""
    # This test only runs on actual Mac Silicon hardware
    # It verifies that PyTorch with MPS support is properly detected

    # Verify platform detection
    assert choosegpu._is_mac_silicon()

    # Verify MPS support detection (requires PyTorch to be installed)
    # If this fails, PyTorch may not be installed or may not have MPS support
    assert choosegpu._has_mps_support(), (
        "PyTorch with MPS support should be available on Mac Silicon. "
        "Make sure PyTorch is installed in test requirements."
    )

    # Test configuration
    original_values = _save_gpu_config()

    # Test enabling GPU
    result = choosegpu.configure_gpu(enable=True)
    assert result == ["mps"]
    assert choosegpu.get_gpu_config() == ["mps"]
    assert choosegpu.are_gpu_settings_configured()

    # Test disabling GPU
    result = choosegpu.configure_gpu(
        enable=False, overwrite_existing_configuration=True
    )
    assert result == ["-1"]
    assert choosegpu.get_gpu_config() == ["-1"]
    assert choosegpu.are_gpu_settings_configured()

    # Clean up
    _restore_gpu_config(original_values)
