"""3D printable ArUco marker STL generator (grooved and AMS multicolor parts).

Generates watertight binary STL meshes for ArUco fiducial markers inspired by
`imontesino/3D-aruco-generator`, without requiring heavy CAD kernel dependencies.

For each marker, supports generating:
1. A single-body grooved/embossed STL (`aruco_id_<id>_grooved.stl`) where black
   cells are recessed by `groove_depth_mm` into the white base tile (for single-color
   printing + painting or filament pause).
2. A two-part multicolor STL pair (`aruco_id_<id>_white.stl` and `aruco_id_<id>_black.stl`)
   designed for Bambu Studio / AMS multi-material printing: load both STLs together as a
   single assembly and assign White filament to the base/inlay and Black filament to the
   marker cells.
"""

import argparse
import logging
import struct
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ARUCO_DICT_MAP: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def generate_aruco_occupancy_grid(
    dictionary_name: str = "DICT_4X4_50",
    marker_id: int = 10,
) -> np.ndarray:
    """Generate a binary occupancy grid for an ArUco marker (0 = black, 1 = white).

    Includes the 1-bit black border around the inner data bits (e.g. a 4x4 dictionary
    produces a 6x6 grid).

    Args:
        dictionary_name: Predefined OpenCV ArUco dictionary name.
        marker_id: Marker ID within the dictionary.

    Returns:
        2D NumPy integer array of shape (N, N) with values 0 (black) and 1 (white).

    Raises:
        ValueError: If `dictionary_name` is unknown or `marker_id` is out of range.
    """
    if dictionary_name not in ARUCO_DICT_MAP:
        raise ValueError(
            f"Invalid ArUco dictionary '{dictionary_name}'. "
            f"Valid options: {sorted(ARUCO_DICT_MAP.keys())}"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dictionary_name])
    max_id = int(aruco_dict.bytesList.shape[0])
    if marker_id < 0 or marker_id >= max_id:
        raise ValueError(
            f"Marker ID {marker_id} out of range for {dictionary_name} (0..{max_id - 1})."
        )

    grid_n = int(aruco_dict.markerSize) + 2
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, grid_n)
    return np.where(marker_img == 255, 1, 0).astype(np.int32)


def format_aruco_ascii(grid: np.ndarray) -> str:
    """Format a binary ArUco occupancy grid (0=black, 1=white) as ASCII art.

    Follows the convention of `3D-aruco-generator` where the white quiet-zone border
    and white cells are rendered as solid blocks (`██`) and black cells as spaces (`  `).

    Args:
        grid: 2D array of 0s and 1s.

    Returns:
        Multi-line string representation of the marker.
    """
    rows, cols = grid.shape
    border_line = "██" * (cols + 2)
    lines = [border_line]
    for r in range(rows):
        row_str = "██" + "".join("  " if grid[r, c] == 0 else "██" for c in range(cols)) + "██"
        lines.append(row_str)
    lines.append(border_line)
    return "\n".join(lines)


def _quad_triangles(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> list[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
]:
    """Build two triangles for a counter-clockwise quad (p0, p1, p2, p3) with outward normal."""
    return [
        (normal, p0, p1, p2),
        (normal, p0, p2, p3),
    ]


def _heightmap_to_triangles(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_top: np.ndarray,
    z_bottom: float = 0.0,
) -> list[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
]:
    """Convert a 2D heightmap over a rectilinear XY grid into a watertight triangle list.

    Cells with `z_top[iy, ix] <= z_bottom` are treated as empty (omitted), allowing both
    solid grooved plates and disjoint multi-part inlays to be meshed cleanly with zero
    internal faces or non-manifold edges where cells share boundaries.

    Args:
        x_coords: Monotonically increasing X grid boundaries of length `nx + 1`.
        y_coords: Monotonically increasing Y grid boundaries of length `ny + 1`.
        z_top: 2D array of shape `(ny, nx)` giving the top Z height of each grid cell.
        z_bottom: Bottom Z plane of the solid cells.

    Returns:
        List of `(normal, v0, v1, v2)` triangle tuples.
    """
    ny, nx = z_top.shape
    triangles: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    eps = 1e-9

    for iy in range(ny):
        y0 = float(y_coords[iy])
        y1 = float(y_coords[iy + 1])
        for ix in range(nx):
            zt = float(z_top[iy, ix])
            if zt <= z_bottom + eps:
                continue
            x0 = float(x_coords[ix])
            x1 = float(x_coords[ix + 1])

            # Top face (+Z normal)
            triangles.extend(
                _quad_triangles(
                    (x0, y0, zt),
                    (x1, y0, zt),
                    (x1, y1, zt),
                    (x0, y1, zt),
                    (0.0, 0.0, 1.0),
                )
            )
            # Bottom face (-Z normal)
            triangles.extend(
                _quad_triangles(
                    (x0, y1, z_bottom),
                    (x1, y1, z_bottom),
                    (x1, y0, z_bottom),
                    (x0, y0, z_bottom),
                    (0.0, 0.0, -1.0),
                )
            )

            # -X neighbor wall
            z_nx0 = float(z_top[iy, ix - 1]) if ix > 0 else z_bottom
            z_low = max(z_bottom, z_nx0)
            if zt > z_low + eps:
                triangles.extend(
                    _quad_triangles(
                        (x0, y0, z_low),
                        (x0, y0, zt),
                        (x0, y1, zt),
                        (x0, y1, z_low),
                        (-1.0, 0.0, 0.0),
                    )
                )

            # +X neighbor wall
            z_nx1 = float(z_top[iy, ix + 1]) if ix + 1 < nx else z_bottom
            z_low = max(z_bottom, z_nx1)
            if zt > z_low + eps:
                triangles.extend(
                    _quad_triangles(
                        (x1, y1, z_low),
                        (x1, y1, zt),
                        (x1, y0, zt),
                        (x1, y0, z_low),
                        (1.0, 0.0, 0.0),
                    )
                )

            # -Y neighbor wall
            z_ny0 = float(z_top[iy - 1, ix]) if iy > 0 else z_bottom
            z_low = max(z_bottom, z_ny0)
            if zt > z_low + eps:
                triangles.extend(
                    _quad_triangles(
                        (x1, y0, z_low),
                        (x1, y0, zt),
                        (x0, y0, zt),
                        (x0, y0, z_low),
                        (0.0, -1.0, 0.0),
                    )
                )

            # +Y neighbor wall
            z_ny1 = float(z_top[iy + 1, ix]) if iy + 1 < ny else z_bottom
            z_low = max(z_bottom, z_ny1)
            if zt > z_low + eps:
                triangles.extend(
                    _quad_triangles(
                        (x0, y1, z_low),
                        (x0, y1, zt),
                        (x1, y1, zt),
                        (x1, y1, z_low),
                        (0.0, 1.0, 0.0),
                    )
                )

    return triangles


def write_binary_stl(
    path: str | Path,
    triangles: Sequence[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
    header_text: str = "GeminiRobotics-SO101 ArUco Marker STL",
) -> Path:
    """Write a sequence of triangles to a standard binary STL file.

    Args:
        path: Output file path.
        triangles: Sequence of `(normal, v0, v1, v2)` tuples.
        header_text: Up to 80-byte ASCII header description.

    Returns:
        Path to the written STL file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_bytes = header_text.encode("ascii", errors="ignore")[:80].ljust(80, b"\x00")
    with out_path.open("wb") as f:
        f.write(header_bytes)
        f.write(struct.pack("<I", len(triangles)))
        for normal, v0, v1, v2 in triangles:
            f.write(
                struct.pack(
                    "<12fH",
                    normal[0],
                    normal[1],
                    normal[2],
                    v0[0],
                    v0[1],
                    v0[2],
                    v1[0],
                    v1[1],
                    v1[2],
                    v2[0],
                    v2[1],
                    v2[2],
                    0,
                )
            )
    return out_path


JAW_TAG_TOTAL_SIDE_MM: float = 25.0
JAW_TAG_MARKER_SIZE_MM: float = 19.53125  # 25.0 * (25.0 / 32.0) -> 78.125% active black square
JAW_TAG_MARGIN_MM: float = 2.734375  # 25.0 * (3.5 / 32.0) -> 25.0 mm outer tile with border
JAW_TAG_THICKNESS_MM: float = 1.5


def generate_aruco_stl(
    marker_id: int = 10,
    marker_size_mm: float = JAW_TAG_MARKER_SIZE_MM,
    margin_mm: float = JAW_TAG_MARGIN_MM,
    thickness_mm: float = JAW_TAG_THICKNESS_MM,
    groove_depth_mm: float = 0.6,
    dictionary_name: str = "DICT_4X4_50",
    output_dir: str | Path = "assets/aruco_markers",
    name_prefix: str | None = None,
    include_multicolor_parts: bool = True,
) -> dict[str, Path]:
    """Generate 3D-printable STL files for a single ArUco marker.

    Produces:
    - `<prefix>_grooved.stl`: Single-body tile with `groove_depth_mm` recessed grooves
      for the black ArUco squares (like `imontesino/3D-aruco-generator`).
    - `<prefix>_white.stl` & `<prefix>_black.stl` (when `include_multicolor_parts=True`):
      Complementary non-overlapping bodies for Bambu Lab AMS multicolor printing.
      Import both together in Bambu Studio as one object with two parts and assign
      White filament to `_white.stl` and Black filament to `_black.stl`.

    Args:
        marker_id: ArUco marker ID (default 10 for the SO-101 gripper jaw tag).
        marker_size_mm: Side length of the black ArUco square in mm (excluding quiet-zone margin;
            default 19.53125 mm for a 25.0 mm outer jaw tile).
        margin_mm: Width of the surrounding white quiet-zone border on each side in mm
            (default 2.734375 mm -> 25.0 mm total outer tile side).
        thickness_mm: Total Z thickness of the marker tile in mm (default 1.5 mm).
        groove_depth_mm: Depth of the black groove / multicolor inlay in mm.
        dictionary_name: OpenCV ArUco dictionary name (default `DICT_4X4_50`).
        output_dir: Directory where generated STL files are written.
        name_prefix: Optional filename prefix (defaults to `aruco_jaw_id_10_25mm` for ID 10 at
            25 mm outer tile, or `aruco_id_<marker_id>_<size>mm`).
        include_multicolor_parts: Whether to also export `_white.stl` and `_black.stl` AMS parts.

    Returns:
        Dictionary mapping part role (`"grooved"`, and optionally `"white"`, `"black"`)
        to the generated `Path`.

    Raises:
        ValueError: If geometric parameters are non-positive or invalid.
    """
    if marker_size_mm <= 0:
        raise ValueError(f"marker_size_mm must be > 0, got {marker_size_mm}")
    if margin_mm < 0:
        raise ValueError(f"margin_mm must be >= 0, got {margin_mm}")
    if thickness_mm <= 0:
        raise ValueError(f"thickness_mm must be > 0, got {thickness_mm}")
    if groove_depth_mm <= 0 or groove_depth_mm >= thickness_mm:
        raise ValueError(
            f"groove_depth_mm ({groove_depth_mm}) must be in (0, thickness_mm ({thickness_mm}))."
        )

    grid = generate_aruco_occupancy_grid(dictionary_name=dictionary_name, marker_id=marker_id)
    n_cells = grid.shape[0]
    cell_mm = marker_size_mm / float(n_cells)
    half_marker = marker_size_mm / 2.0
    half_total = half_marker + margin_mm
    total_side_mm = marker_size_mm + 2.0 * margin_mm

    # Build rectilinear grid coordinates centered at (0, 0) in XY, Z in [0, thickness_mm].
    inner_coords = [-half_marker + i * cell_mm for i in range(n_cells + 1)]
    if margin_mm > 0:
        coords = [-half_total, *inner_coords, half_total]
        full_grid = np.ones((n_cells + 2, n_cells + 2), dtype=np.int32)
        # Image row 0 is top (+Y), row n-1 is bottom (-Y). Map iy=1..n_cells from bottom to top:
        for r in range(n_cells):
            iy = n_cells - r
            full_grid[iy, 1 : n_cells + 1] = grid[r, :]
    else:
        coords = inner_coords
        full_grid = np.zeros((n_cells, n_cells), dtype=np.int32)
        for r in range(n_cells):
            iy = n_cells - 1 - r
            full_grid[iy, :] = grid[r, :]

    z_base = thickness_mm - groove_depth_mm

    # 1. Grooved / single-body heightmap (white=thickness_mm, black=thickness_mm - groove_depth_mm)
    z_top_grooved = np.where(full_grid == 1, thickness_mm, z_base).astype(np.float64)
    tris_grooved = _heightmap_to_triangles(coords, coords, z_top_grooved, z_bottom=0.0)

    if name_prefix is None:
        if marker_id == 10 and (
            abs(total_side_mm - JAW_TAG_TOTAL_SIDE_MM) < 1e-2
            or abs(marker_size_mm - JAW_TAG_TOTAL_SIDE_MM) < 1e-2
        ):
            name_prefix = f"aruco_jaw_id_{marker_id}_25mm"
        else:
            name_prefix = f"aruco_id_{marker_id}_{marker_size_mm:g}mm"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    header = (
        f"ArUco {dictionary_name} ID {marker_id} "
        f"({total_side_mm:g}mm tile: {marker_size_mm:.2f}mm + {margin_mm:.2f}mm margin, "
        f"{thickness_mm:g}mm thick)"
    )

    grooved_file = write_binary_stl(
        out_path / f"{name_prefix}_grooved.stl",
        tris_grooved,
        header_text=f"{header} [grooved]",
    )
    created: dict[str, Path] = {"grooved": grooved_file}
    logger.info(f"[ArUco STL] Generated {grooved_file}")

    if include_multicolor_parts:
        # White body is identical to the grooved base (has recessed pockets for the black inlay)
        white_file = write_binary_stl(
            out_path / f"{name_prefix}_white.stl",
            tris_grooved,
            header_text=f"{header} [AMS white body]",
        )
        # Black body occupies only the black cells (full_grid == 0) from z_base to thickness_mm
        z_top_black = np.where(full_grid == 0, thickness_mm, z_base).astype(np.float64)
        tris_black = _heightmap_to_triangles(coords, coords, z_top_black, z_bottom=z_base)
        black_file = write_binary_stl(
            out_path / f"{name_prefix}_black.stl",
            tris_black,
            header_text=f"{header} [AMS black inlay]",
        )
        created["white"] = white_file
        created["black"] = black_file
        logger.info(f"[ArUco STL] Generated AMS pair: {white_file}, {black_file}")

    return created


def generate_jaw_tag_stls(
    output_dir: str | Path = "assets/aruco_markers",
    marker_id: int = 10,
    tag_size_mm: float = JAW_TAG_TOTAL_SIDE_MM,
    margin_mm: float = JAW_TAG_MARGIN_MM,
    thickness_mm: float = JAW_TAG_THICKNESS_MM,
    groove_depth_mm: float = 0.6,
    dictionary_name: str = "DICT_4X4_50",
) -> dict[str, Path]:
    """Generate the 3D-printable STL set for the SO-101 gripper jaw ArUco tag (ID 10, 25 mm tile).

    Args:
        output_dir: Directory where STL files will be saved.
        marker_id: ArUco ID for the jaw tag (default 10).
        tag_size_mm: Total outer tile side length in mm, white border included (default 25.0 mm).
        margin_mm: White quiet-zone border width on each side in mm (default 2.734375 mm ->
            19.53125 mm active black ArUco square inside the 25.0 mm tile).
        thickness_mm: Total tile thickness in mm (default 1.5 mm).
        groove_depth_mm: Black groove / inlay depth in mm (default 0.6 mm = 3 layers at 0.2 mm).
        dictionary_name: OpenCV ArUco dictionary name (default `DICT_4X4_50`).

    Returns:
        Dictionary mapping `"grooved"`, `"white"`, and `"black"` to the generated STL paths.
    """
    marker_size_mm = tag_size_mm - 2.0 * margin_mm
    return generate_aruco_stl(
        marker_id=marker_id,
        marker_size_mm=marker_size_mm,
        margin_mm=margin_mm,
        thickness_mm=thickness_mm,
        groove_depth_mm=groove_depth_mm,
        dictionary_name=dictionary_name,
        output_dir=output_dir,
        include_multicolor_parts=True,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    """Create CLI parser for standalone 3D ArUco STL generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3D-printable ArUco marker STL files (grooved single-body and "
            "Bambu Studio / AMS white+black multicolor parts)."
        )
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=10,
        help="ArUco marker ID to generate (default: 10 for the SO-101 gripper jaw tag).",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=JAW_TAG_MARKER_SIZE_MM,
        help=(
            "Active black ArUco square side length in mm "
            f"(default: {JAW_TAG_MARKER_SIZE_MM:g} for a 25 mm total tile)."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=JAW_TAG_MARGIN_MM,
        help=(
            "White quiet-zone margin width per side in mm "
            f"(default: {JAW_TAG_MARGIN_MM:g} -> 25.0 mm total side)."
        ),
    )
    parser.add_argument(
        "--box-side",
        type=float,
        default=None,
        help=(
            "Optional total outer box side length in mm (marker_size + 2 * margin), matching "
            "3D-aruco-generator's --box_side. If set, overrides --marker-size as "
            "(box_side - 2 * margin)."
        ),
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=JAW_TAG_THICKNESS_MM,
        help=f"Total tile thickness in mm (default: {JAW_TAG_THICKNESS_MM:g}).",
    )
    parser.add_argument(
        "--groove-depth",
        type=float,
        default=0.6,
        help="Groove / black inlay depth in mm (default: 0.6, i.e. 3 layers at 0.2 mm).",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_50",
        choices=sorted(ARUCO_DICT_MAP.keys()),
        help="ArUco dictionary name (default: DICT_4X4_50).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/aruco_markers",
        help="Output directory for generated STL files (default: assets/aruco_markers).",
    )
    parser.add_argument(
        "--all-workspace-markers",
        action="store_true",
        help="Also generate 40 mm tabletop workspace corner markers (IDs 0, 1, 2, 3).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for `uv run python -m tools.aruco_stl`."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    marker_size_mm = (
        args.box_side - 2.0 * args.margin if args.box_side is not None else args.marker_size
    )
    total_side_mm = marker_size_mm + 2.0 * args.margin

    grid = generate_aruco_occupancy_grid(args.dictionary, args.marker_id)
    print(f"ArUco Dictionary: {args.dictionary} | Marker ID: {args.marker_id}")
    print(
        f"Dimensions: {marker_size_mm:g} mm marker + {args.margin:g} mm margin "
        f"= {total_side_mm:g} mm total side | "
        f"thickness={args.thickness:g} mm, groove={args.groove_depth:g} mm"
    )
    print(format_aruco_ascii(grid))

    created = generate_aruco_stl(
        marker_id=args.marker_id,
        marker_size_mm=marker_size_mm,
        margin_mm=args.margin,
        thickness_mm=args.thickness,
        groove_depth_mm=args.groove_depth,
        dictionary_name=args.dictionary,
        output_dir=args.output_dir,
        include_multicolor_parts=True,
    )
    print("\nGenerated 3D printable STL files:")
    for role, path in created.items():
        print(f"  • [{role:7s}] {path}")

    if args.all_workspace_markers:
        print("\nGenerating 40 mm workspace corner markers (IDs 0, 1, 2, 3)...")
        for corner_id in (0, 1, 2, 3):
            corner_files = generate_aruco_stl(
                marker_id=corner_id,
                marker_size_mm=40.0,
                margin_mm=5.0,
                thickness_mm=args.thickness,
                groove_depth_mm=args.groove_depth,
                dictionary_name=args.dictionary,
                output_dir=args.output_dir,
                include_multicolor_parts=True,
            )
            for role, path in corner_files.items():
                print(f"  • [ID {corner_id} {role:7s}] {path}")


if __name__ == "__main__":
    main()
