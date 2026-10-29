"""Choosing the compute device: CPU baseline, optional GPU, graceful failure."""

from pathlib import Path

import pytest

from azhi import acceleration
from azhi.acceleration import (
    DEFAULT_MODE,
    MODE_CPU,
    MODE_CUDA,
    MODE_LABELS,
    Device,
    build_supports_cuda,
    cpu_device,
    default_mode,
    describe_hardware,
    fallback_to_cpu,
    offered_modes,
    resolve,
)


@pytest.fixture
def cpu_only_build(monkeypatch):
    """A build with the CUDA libraries stripped out of it - i.e. any bundle."""
    monkeypatch.setattr(acceleration, "build_supports_cuda", lambda: False)


@pytest.fixture
def cuda_capable_build(monkeypatch):
    monkeypatch.setattr(acceleration, "build_supports_cuda", lambda: True)


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(acceleration, "cuda_device_count", lambda: 0)


@pytest.fixture
def one_gpu(monkeypatch):
    monkeypatch.setattr(acceleration, "cuda_device_count", lambda: 1)


def test_cpu_is_the_default_when_no_gpu_is_present(no_gpu):
    device = resolve(DEFAULT_MODE)
    assert device.device == "cpu"
    assert device.compute_type == "int8"
    assert not device.is_gpu
    # Auto falling back to CPU is normal, not a warning.
    assert device.note == ""


def test_auto_prefers_a_gpu_when_one_is_available(one_gpu):
    device = resolve("auto")
    assert device.device == "cuda"
    assert device.compute_type == "float16"
    assert device.is_gpu


def test_cpu_is_honoured_even_with_a_gpu_present(one_gpu):
    # CPU stays fully supported; asking for it must not be overridden.
    assert resolve(MODE_CPU).device == "cpu"


def test_an_impossible_gpu_request_says_so_instead_of_failing_the_run(no_gpu):
    device = resolve(MODE_CUDA)
    assert device.device == "cpu"
    # Not a silent downgrade: the operator is told why their choice was not used.
    assert "No CUDA device was detected" in device.note
    assert "Results are unaffected" in device.note


def test_an_unknown_or_empty_mode_falls_back_to_auto(no_gpu):
    assert resolve("").device == "cpu"
    assert resolve("wildly-invalid").device == "cpu"


def test_a_failed_gpu_run_offers_a_cpu_retry():
    gpu = Device("cuda", "float16", "NVIDIA GPU (CUDA)")
    retry = fallback_to_cpu(gpu, RuntimeError("out of memory"))
    assert retry is not None
    assert retry.device == "cpu"
    assert "out of memory" in retry.note
    assert "retried on the CPU" in retry.note


def test_a_failed_cpu_run_has_nothing_to_fall_back_to():
    # Retrying the CPU on the CPU would just fail again; the caller re-raises.
    assert fallback_to_cpu(cpu_device(), RuntimeError("boom")) is None


def test_an_error_with_no_message_still_produces_a_usable_note():
    retry = fallback_to_cpu(Device("cuda", "float16", "GPU"), RuntimeError())
    assert retry is not None and "RuntimeError" in retry.note


def test_detection_is_local_and_treats_any_failure_as_no_gpu(monkeypatch):
    class _Broken:
        @staticmethod
        def get_cuda_device_count():
            raise OSError("no driver")

    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", _Broken)
    # A missing driver, a CPU-only wheel or a broken install all mean "no GPU"
    # operationally - never an exception that stops the app.
    assert acceleration.cuda_device_count() == 0
    assert not acceleration.cuda_available()


def test_hardware_description_is_honest_either_way(no_gpu):
    assert "Processor only" in describe_hardware()


def test_hardware_description_counts_devices(one_gpu, monkeypatch):
    assert "1 device" in describe_hardware()
    monkeypatch.setattr(acceleration, "cuda_device_count", lambda: 2)
    assert "2 devices" in describe_hardware()


def test_every_offered_mode_resolves(no_gpu):
    for mode, label in MODE_LABELS:
        assert resolve(mode).device in {"cpu", "cuda"}
        assert label  # each mode is presented with a human label


def test_describe_names_the_device_and_precision():
    assert Device("cuda", "float16", "NVIDIA GPU (CUDA)").describe() == (
        "NVIDIA GPU (CUDA) (cuda/float16)"
    )


# -- A build that cannot use a GPU at all ----------------------------------
#
# The bundle strips the CUDA libraries, so selecting a GPU in a shipped copy
# used to fall back to the CPU without saying so: the setting looked honoured
# and the run was simply slower than the operator believed it to be. These hold
# the option shut where it cannot work, and keep the reason given accurate.


def test_a_frozen_bundle_reports_no_gpu_support(monkeypatch):
    monkeypatch.setattr(acceleration.sys, "frozen", True, raising=False)
    assert build_supports_cuda() is False


def test_running_from_source_keeps_gpu_support(monkeypatch):
    monkeypatch.delattr(acceleration.sys, "frozen", raising=False)
    assert build_supports_cuda() is True


def test_a_cpu_only_build_does_not_offer_a_gpu(cpu_only_build):
    modes = [mode for mode, _ in offered_modes()]
    assert modes == [MODE_CPU]
    assert MODE_CUDA not in modes


def test_a_cpu_only_build_does_not_offer_auto_either(cpu_only_build):
    """Auto would resolve to the CPU and its label promises a graphics card."""
    labels = [label for _, label in offered_modes()]
    assert labels == ["CPU"]
    assert not any("GPU" in label for label in labels)
    assert default_mode() == MODE_CPU


def test_a_cuda_capable_build_offers_everything(cuda_capable_build):
    assert offered_modes() == MODE_LABELS
    assert default_mode() == DEFAULT_MODE


def test_offered_modes_cannot_be_mutated_through_the_returned_list(
    cuda_capable_build,
):
    offered_modes().clear()
    assert offered_modes() == MODE_LABELS


def test_a_gpu_request_in_a_cpu_only_build_blames_the_build(cpu_only_build):
    """Not "no device detected": that sends someone off to fit a card."""
    device = resolve(MODE_CUDA)
    assert device.device == "cpu"
    assert "does not include GPU support" in device.note
    assert "detected" not in device.note


def test_auto_in_a_cpu_only_build_is_quiet_about_it(cpu_only_build):
    """Auto asked for the best available and got it. There is nothing to report."""
    device = resolve(acceleration.MODE_AUTO)
    assert device.device == "cpu"
    assert device.note == ""


def test_a_cpu_only_build_does_not_even_look_for_a_card(cpu_only_build, monkeypatch):
    def refuse():
        raise AssertionError("asked the driver about a GPU this build cannot use")

    monkeypatch.setattr(acceleration, "cuda_device_count", refuse)
    for mode in (MODE_CPU, MODE_CUDA, acceleration.MODE_AUTO):
        assert resolve(mode).device == "cpu"


def test_the_hardware_line_blames_the_build_not_the_machine(cpu_only_build):
    text = describe_hardware()
    assert "this copy has no graphics-card support" in text
    assert "found" not in text


def test_every_mode_a_cpu_only_build_offers_resolves(cpu_only_build):
    for mode, _ in offered_modes():
        assert resolve(mode).device == "cpu"


def test_a_saved_gpu_setting_from_another_build_still_runs(cpu_only_build):
    """A profile or settings file can outlive the build that offered the mode."""
    device = resolve(MODE_CUDA)
    assert device.compute_type == acceleration.CPU_COMPUTE_TYPE
    assert device.note, "it must say why the setting was not honoured"


def test_the_packaging_still_strips_cuda_so_the_claim_holds():
    """build_supports_cuda() says "no" in a bundle *because* the spec strips CUDA.

    If that stripping is ever removed to ship a GPU-capable bundle, this fails
    rather than leaving the application hiding an option that would by then work.
    """
    spec = Path(__file__).resolve().parent.parent / "packaging" / "azhi.spec"
    text = spec.read_text(encoding="utf-8")
    assert "_CUDA_MARKERS" in text
    for marker in ("cudnn", "cublas", "cudart", "onnxruntime_providers_cuda"):
        assert f'"{marker}"' in text, f"{marker} is no longer stripped"
    # The lines that actually do the removing, not just the list of names.
    assert "if not _is_cuda_lib(b[0])" in text
    assert "build_supports_cuda" in text, (
        "the spec should name the function that depends on this stripping"
    )
