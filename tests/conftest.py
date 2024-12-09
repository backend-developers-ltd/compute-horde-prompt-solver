import os
import pathlib
import shutil
import tempfile

import pytest


@pytest.fixture()
def input_file() -> str:
    path = pathlib.Path(__file__).parent / "payload" / "prompts.txt"
    tmp_path = tempfile.mktemp()
    shutil.copy(path.as_posix(), tmp_path)
    yield tmp_path
    if os.path.isfile(tmp_path):
        os.remove(tmp_path)

    output_file = tmp_path + '.json'
    if os.path.isfile(output_file):
        os.remove(output_file)
