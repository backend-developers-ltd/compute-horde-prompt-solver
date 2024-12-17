import hashlib
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest
import requests
from huggingface_hub import snapshot_download

TIMEOUT = 180


@pytest.fixture(scope="session", autouse=True)
def download_model():
    snapshot_download(
        repo_id="microsoft/Phi-3.5-mini-instruct",
        local_dir=pathlib.Path(__file__).parent.parent.parent
        / "src"
        / "compute_horde_prompt_solver"
        / "saved_model/",
        revision="cd6881a82d62252f5a84593c61acf290f15d89e3",
    )


@pytest.mark.parametrize(
    "seed,expected_output_file",
    [
        ("1234567891", "expected_real_1234567891_output.json"),
        ("99", "expected_real_99_output.json"),
    ],
)
def test_cli(input_file, seed, expected_output_file):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "src.compute_horde_prompt_solver",
            "--temperature",
            "0.5",
            "--top-p",
            "0.8",
            "--max-tokens",
            "256",
            "--seed",
            seed,
            "--output-dir",
            tempfile.gettempdir(),
            input_file,
        ],
        timeout=TIMEOUT,
    )
    expected = (
        pathlib.Path(__file__).parent.parent / "payload" / expected_output_file
    ).read_text()
    actual = pathlib.Path(input_file + ".json").read_text()
    assert expected == actual


def get_url_within_time(url, timeout=TIMEOUT):
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except (requests.HTTPError, requests.ConnectionError):
            pass

        time.sleep(
            0.5
        )  # Wait a bit before trying again to not overload the server and your machine.

    raise TimeoutError(f"Could not get data from {url} within {timeout} seconds")


@pytest.mark.parametrize(
    "seed,expected_output_file",
    [
        ("1234567891", "expected_real_1234567891_output.json"),
        ("99", "expected_real_99_output.json"),
    ],
)
def test_http(input_file, seed, expected_output_file):
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.compute_horde_prompt_solver",
            "--temperature",
            "0.5",
            "--top-p",
            "0.8",
            "--max-tokens",
            "256",
            "--output-dir",
            tempfile.gettempdir(),
            "--server",
            input_file,
        ],
    )
    try:
        base_url = "http://localhost:8000/"
        get_url_within_time(base_url + "health")

        with requests.post(base_url + "execute-job", json={"seed": seed}) as resp:
            resp.raise_for_status()
            hashes = resp.json()
        try:
            requests.get(base_url + "terminate")
        except Exception:
            pass
        assert hashes == {
            input_file + ".json": hashlib.sha256(
                pathlib.Path(input_file + ".json").read_bytes()
            ).hexdigest()
        }
    finally:
        server.terminate()
        server.wait()
