"""Options confined to the optional GPU diagnostics suite."""
import gc
from datetime import datetime
from pathlib import Path
import pytest
import torch


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.call_report = report


@pytest.fixture(scope="session")
def gpu_output(request):
    directory = request.config.getoption("--gpu-output")
    return Path(directory or ("runs/check1080/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))


@pytest.fixture(autouse=True)
def cpu_threads(request):
    previous = torch.get_num_threads()
    if request.config.getoption("--run-gpu"):
        count = request.config.getoption("--gpu-threads")
        if count < 1:
            pytest.fail("gpu-threads must be positive")
        torch.set_num_threads(count)
    yield
    torch.set_num_threads(previous)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
