# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path
from subprocess import run
from numpy import loadtxt
from numpy.testing import assert_allclose
import pytest
import sys


@pytest.mark.parametrize(
    "example_file, params",
    [
        ("Lbracket.py", "30 10 0.5 .8 3 1 20 0 0 0"),
    ],
)
def test_stress_lbracket(tmp_path, example_file, params):
    """
    Small test for the 2D stress Lbracket example.

    The example is run as a script, writes obj to CSV files in
    tmp_path, and the results are compared against reference CSVs.
    """
    test_path = Path(__file__).resolve().parent
    file_path = (
        test_path.parent
        / "examples"
        / "topology_optimization"
        / "stress_constraint"
        / example_file
    )
    cmd = [sys.executable, str(file_path)] + params.split(" ")
    run(cmd, cwd=tmp_path, shell=False, check=True)
    obj = loadtxt(tmp_path / "stress_lbracket_obj.csv", delimiter=",")
    obj_ref = loadtxt(test_path / "test_files" / "stress_lbracket_obj.csv", delimiter=",")
    assert_allclose(obj, obj_ref,rtol=1e-6)