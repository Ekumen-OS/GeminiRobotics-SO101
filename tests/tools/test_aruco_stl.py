"""Smoke tests for tools/aruco_stl.py 3D-printable ArUco marker STL generator."""

import struct
from pathlib import Path

from tools.aruco_stl import generate_jaw_tag_stls, main


def _verify_binary_stl(file_path: Path) -> int:
    """Verify that a file is a valid binary STL and return its triangle count.

    Args:
        file_path: Path to the STL file.

    Returns:
        Number of triangles declared and stored in the file.
    """
    assert file_path.is_file(), f"Expected file {file_path} to exist."
    data = file_path.read_bytes()
    assert len(data) >= 84, f"File {file_path} is too small to be a binary STL ({len(data)} bytes)."

    (num_triangles,) = struct.unpack("<I", data[80:84])
    assert num_triangles > 0, f"Expected non-zero triangle count in {file_path}."

    expected_len = 84 + num_triangles * 50
    assert len(data) == expected_len, (
        f"File {file_path} size ({len(data)} bytes) does not match "
        f"header triangle count {num_triangles} (expected {expected_len} bytes)."
    )
    return num_triangles


def test_generate_jaw_tag_stls(tmp_path: Path) -> None:
    """Generate grooved and two-part STLs for ID 10 and verify binary STL integrity."""
    result = generate_jaw_tag_stls(output_dir=tmp_path, marker_id=10)

    expected_keys = {"grooved", "white", "black"}
    assert set(result.keys()) == expected_keys

    expected_files = {
        "grooved": tmp_path / "aruco_jaw_id_10_25mm_grooved.stl",
        "white": tmp_path / "aruco_jaw_id_10_25mm_white.stl",
        "black": tmp_path / "aruco_jaw_id_10_25mm_black.stl",
    }

    for role, path in expected_files.items():
        assert result[role] == path
        tri_count = _verify_binary_stl(path)
        assert tri_count > 0


def test_generate_aruco_stl_cli(tmp_path: Path) -> None:
    """Verify CLI entrypoint parses arguments and generates binary STL files."""
    cli_out = tmp_path / "cli_out"
    main(["--marker-id", "10", "--output-dir", str(cli_out)])

    for suffix in ("grooved", "white", "black"):
        stl_path = cli_out / f"aruco_jaw_id_10_25mm_{suffix}.stl"
        _verify_binary_stl(stl_path)
