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

    # Verify MPS support detection
    import torch

    assert (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ), "PyTorch with MPS support should be available on Mac Silicon. "

    # Test configuration
    original_values = _save_gpu_config()

    # Test enabling GPU
    result = choosegpu.configure_gpu(enable=True)
    assert result == ["mps"]
    assert choosegpu.get_gpu_config() == ["mps"]
    assert choosegpu.are_gpu_settings_configured()
    assert choosegpu.is_gpu_enabled() is True

    # Test disabling GPU
    result = choosegpu.configure_gpu(
        enable=False, overwrite_existing_configuration=True
    )
    assert result == ["-1"]
    assert choosegpu.get_gpu_config() == ["-1"]
    assert choosegpu.are_gpu_settings_configured()
    assert choosegpu.is_gpu_enabled() is False

    # Clean up
    _restore_gpu_config(original_values)


def test_is_gpu_enabled_helper():
    """Test the is_gpu_enabled() convenience method.

    This tests the basic logic of the helper on the current platform.
    """
    original_values = _save_gpu_config()

    # Test with explicit GPU config values
    # Simulate disabled GPU
    choosegpu.has_user_configured_gpus = True
    if choosegpu._is_mac_silicon():
        choosegpu._mac_silicon_gpu_config = ["-1"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    assert choosegpu.is_gpu_enabled() is False

    # Simulate enabled GPU
    if choosegpu._is_mac_silicon():
        choosegpu._mac_silicon_gpu_config = ["mps"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-123"

    assert choosegpu.is_gpu_enabled() is True

    # Simulate not configured
    choosegpu.has_user_configured_gpus = False
    if not choosegpu._is_mac_silicon() and "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    choosegpu._mac_silicon_gpu_config = None

    assert choosegpu.is_gpu_enabled() is None

    # Clean up
    _restore_gpu_config(original_values)


def test_check_if_gpu_libraries_see_gpu(mocker):
    """Test the check_if_gpu_libraries_see_gpu() method that checks PyTorch's view of GPU hardware.

    This tests the behavioral differences between Mac Silicon and NVIDIA platforms.
    """
    original_values = _save_gpu_config()

    # Test on Mac Silicon
    if choosegpu._is_mac_silicon():
        # On Mac, MPS is always available if hardware supports it, regardless of configure_gpu
        # Configure GPU to be disabled
        choosegpu.configure_gpu(enable=False, overwrite_existing_configuration=True)
        assert choosegpu.is_gpu_enabled() is False

        # But hardware is still available from PyTorch's perspective
        # (assuming PyTorch with MPS is installed, which it should be per requirements_dev.txt)
        assert choosegpu.check_if_gpu_libraries_see_gpu() is True, (
            "On Mac Silicon, MPS hardware should be available even when configure_gpu(enable=False). "
            "This is a fundamental difference from NVIDIA CUDA."
        )

        # Even with GPU enabled, hardware is available
        choosegpu.configure_gpu(enable=True, overwrite_existing_configuration=True)
        assert choosegpu.is_gpu_enabled() is True
        assert choosegpu.check_if_gpu_libraries_see_gpu() is True

    # Test on NVIDIA (or mock it on Mac)
    else:
        # Test with GPU enabled - only if GPUs are actually available
        mocker.patch("choosegpu._is_mac_silicon", return_value=False)

        # Check if any GPUs are available first
        available_gpus = choosegpu.get_available_gpus()
        if len(available_gpus) > 0:
            # When GPU is enabled, CUDA should be available (if hardware exists)
            choosegpu.configure_gpu(enable=True, overwrite_existing_configuration=True)
            assert choosegpu.is_gpu_enabled() is True
            assert (
                choosegpu.check_if_gpu_libraries_see_gpu() is True
            ), "On NVIDIA with GPUs present, CUDA should be available when configure_gpu(enable=True)"

        # When GPU is disabled, CUDA should NOT be available
        # This test works even without real GPUs
        choosegpu.configure_gpu(enable=False, overwrite_existing_configuration=True)
        assert choosegpu.is_gpu_enabled() is False
        # With CUDA_VISIBLE_DEVICES="-1", torch.cuda.is_available() should return False
        assert (
            choosegpu.check_if_gpu_libraries_see_gpu() is False
        ), "On NVIDIA, CUDA should not be available when configure_gpu(enable=False)"

    # Clean up
    _restore_gpu_config(original_values)


def test_check_if_gpu_libraries_see_gpu_no_pytorch(mocker):
    """Test check_if_gpu_libraries_see_gpu() when PyTorch is not installed."""
    original_values = _save_gpu_config()

    # Mock PyTorch import to fail
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("PyTorch not installed")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    # Should return False when PyTorch is not installed
    assert choosegpu.check_if_gpu_libraries_see_gpu() is False

    # Clean up
    _restore_gpu_config(original_values)


def test_get_available_gpus_no_pytorch_on_mac(mocker):
    """Test that get_available_gpus() works on Mac Silicon even without PyTorch installed."""
    if not choosegpu._is_mac_silicon():
        pytest.skip("This test only runs on Mac Silicon")

    # Mock PyTorch import to fail
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("PyTorch not installed")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    # get_available_gpus() should still work and return MPS device
    # even without PyTorch installed (all Mac Silicon have GPUs)
    available_gpus = choosegpu.get_available_gpus()
    assert available_gpus == [(0, "mps")], (
        "On Mac Silicon, get_available_gpus() should return MPS device "
        "even when PyTorch is not installed, since all Mac Silicon have GPUs"
    )
