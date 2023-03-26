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
    return os.environ.get("CUDA_VISIBLE_DEVICES", None), copy.copy(
        choosegpu.preferred_gpu_ids
    )


def _restore_gpu_config(original_values):
    del os.environ["CUDA_VISIBLE_DEVICES"]
    if original_values[0] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_values[0]
    choosegpu.preferred_gpu_ids = original_values[1]


@pytest.fixture
def mocked_pynvml(mocker):
    # Mock the nvidia library calls
    # libnvml context manager's call of nvmlInit will fail in CI
    # https://stackoverflow.com/q/28850070/130164
    mocker.patch("nvitop.api.libnvml").return_value.__enter__.return_value = None


def test_gpu_ids(mocker, mocked_pynvml):
    # This test is an exception to the rule of "don't call config.configure_gpu() in tests; let conftest.py handle it"
    # But we restore the environment variable after the test.

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

    original_values = _save_gpu_config()

    mocker.patch("nvitop.Device.all", return_value=[])
    choosegpu.configure_gpu()

    # clean up
    _restore_gpu_config(original_values)
