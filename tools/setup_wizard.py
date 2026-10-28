"""Interactive wizard for workspace setup and ArUco marker configuration."""

import functools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from robot.cartesian_ik import CartesianIK
from robot.constants import PRESET_POSES
from robot.settings import get_settings, table_height_to_robot_z_m
from tools.calibration_sheet import (
    generate_aruco_markers,
    generate_jaw_tag_sheet,
    generate_placement_guide,
)

logger = logging.getLogger(__name__)

# Reference workspace setup (mm from the shoulder-pan rotation axis) and marker size (mm),
# used as the fallback when no calibration JSON exists yet.
REFERENCE_WORKSPACE_MM: dict[str, float] = {
    "near_x_mm": 97.0,
    "depth_mm": 317.0,
    "width_mm": 342.0,
    "lateral_offset_mm": -69.0,
    "marker_size_mm": 40.0,
}

_MARKER_IDS_ALL: list[int] = [0, 1, 2, 3, 10]


@functools.lru_cache(maxsize=4)
def get_pan_origin_offset_x_mm(xml_path: str | Path | None = None) -> float:
    """Read forward X offset (mm) between robot base origin and shoulder pan rotation axis.

    The SO-101 base frame origin is 38.8 mm behind the shoulder_pan rotation axis.
    This function extracts that geometric offset dynamically from the MJCF model.

    Args:
        xml_path: Optional path to the MJCF scene XML file. Defaults to simulation/assets/scene.xml.

    Returns:
        Offset in mm along X from base origin to pan rotation axis.
    """
    if xml_path is None:
        xml_path = Path(__file__).resolve().parent.parent / "simulation" / "assets" / "scene.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return float(model.body("shoulder").pos[0] * 1000.0)


def pan_to_base_xy_mm(
    x_pan_mm: float, y_pan_mm: float, xml_path: str | Path | None = None
) -> tuple[float, float]:
    """Convert coordinates measured from shoulder pan rotation axis to robot base frame.

    Args:
        x_pan_mm: X coordinate measured from pan axis (mm).
        y_pan_mm: Y coordinate measured from pan axis (mm).
        xml_path: Optional MJCF scene XML path.

    Returns:
        Tuple of (x_base_mm, y_base_mm) in robot base frame.
    """
    offset_x = get_pan_origin_offset_x_mm(xml_path)
    return (round(x_pan_mm + offset_x, 1), round(y_pan_mm, 1))


def base_to_pan_xy_mm(
    x_base_mm: float, y_base_mm: float, xml_path: str | Path | None = None
) -> tuple[float, float]:
    """Convert coordinates from robot base frame to shoulder-pan-relative coordinates.

    Args:
        x_base_mm: X coordinate in robot base frame (mm).
        y_base_mm: Y coordinate in robot base frame (mm).
        xml_path: Optional MJCF scene XML path.

    Returns:
        Tuple of (x_pan_mm, y_pan_mm) measured from shoulder pan rotation axis.
    """
    offset_x = get_pan_origin_offset_x_mm(xml_path)
    return (round(x_base_mm - offset_x, 1), round(y_base_mm, 1))


def _hover_reachable(
    ik: CartesianIK,
    target_xyz_m: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray | None:
    """Ask the runtime's question: would a top-down hover move to this point succeed?

    Makes the same single ``solve_ik`` call as ``agent/motion.py`` (including the solver's own
    retry from a pan-aligned neutral posture) and accepts the result when the 3-D error is at
    most 5 mm.

    Args:
        ik: CartesianIK solver instance.
        target_xyz_m: Hover target in the robot base frame (m).
        seed: 6-DOF joint seed (radians), as the runtime would start the move from.

    Returns:
        The IK solution if the 3-D positioning error is <= 5.0 mm, otherwise None.
    """
    target = np.asarray(target_xyz_m, dtype=np.float64)
    qpos = ik.solve_ik(
        target_pos=target,
        initial_qpos=seed,
        approach="top_down",
        null_weight=1e-3,
    )
    achieved = ik.forward_kinematics(qpos)
    if float(np.linalg.norm(achieved - target)) <= 0.005:
        return np.asarray(qpos, dtype=np.float64)
    return None


def evaluate_workspace_reachability(
    workspace_corners_base: dict[str, dict[str, float]],
    step_mm: float = 40.0,
    ik: CartesianIK | None = None,
) -> tuple[float, list[str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate Cartesian hover reachability across a workspace rectangle and marker corners.

    Samples the rectangle on a grid with approximately step_mm resolution at 30 mm hover height,
    each point solved once from the HOME seed like a runtime move.
    A point is reachable when the 3-D IK positioning error is <= 5.0 mm.

    Args:
        workspace_corners_base: Dictionary of 4 corners in robot base frame.
        step_mm: Grid sampling resolution in millimeters.
        ik: Optional CartesianIK instance (reused to avoid reloading the MJCF model).

    Returns:
        Tuple of (reachable_fraction, out_of_reach_marker_names, reach_grid).
    """
    if ik is None:
        ik = CartesianIK()

    target_z_m = table_height_to_robot_z_m(0.030, mode="real")

    all_x = [float(c["x"]) for c in workspace_corners_base.values()]
    all_y = [float(c["y"]) for c in workspace_corners_base.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    xs = np.arange(x_min, x_max + 1e-3, step_mm)
    ys = np.arange(y_min, y_max + 1e-3, step_mm)
    reachable_mask = np.zeros((len(xs), len(ys)), dtype=bool)

    reachable_count = 0
    total_count = len(xs) * len(ys)

    home = PRESET_POSES["HOME"].copy()
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            target = np.array([x / 1000.0, y / 1000.0, target_z_m])
            if _hover_reachable(ik, target, home) is not None:
                reachable_mask[i, j] = True
                reachable_count += 1

    fraction = reachable_count / max(1, total_count)

    # Evaluate marker reachability using the same helper
    out_of_reach_markers: list[str] = []
    for name, pt_dict in workspace_corners_base.items():
        mx, my = float(pt_dict["x"]), float(pt_dict["y"])
        if _hover_reachable(ik, np.array([mx / 1000.0, my / 1000.0, target_z_m]), home) is None:
            out_of_reach_markers.append(name)

    grid = (xs, ys, reachable_mask)
    return (fraction, out_of_reach_markers, grid)


def setup_workspace_interactive(
    config_path: str | Path | None = None,
    output_dir: str | Path = "assets/aruco_markers",
    interactive: bool = True,
    reuse: bool | None = None,
    *,
    near_x_mm: float | None = None,
    depth_mm: float | None = None,
    width_mm: float | None = None,
    lateral_offset_mm: float | None = None,
    marker_size_mm: float | None = None,
) -> dict[str, Any]:
    """Configure workspace rectangle geometry and generate placement guide and ArUco markers.

    Prompts the operator for workspace rectangle parameters measured from the shoulder-pan
    rotation axis, calculates corner coordinates in the robot base frame, verifies reachability,
    updates configuration JSON, and renders the A4 printable placement guide.

    Args:
        config_path: Target JSON config file path. Defaults to config/workspace_calibration.json.
        output_dir: Folder to save marker PNGs, placement guide, and jaw tag sheet.
        interactive: If True and running in a terminal, prompts the user for inputs.
        reuse: If True, reuse existing markers without regenerating files. If False, force
            regeneration. If None, asks the user interactively (or auto-detects).
        near_x_mm: Optional override for forward distance from pan axis to near markers (mm).
        depth_mm: Optional override for rectangle depth from near to far markers (mm).
        width_mm: Optional override for rectangle width from right to left markers (mm).
        lateral_offset_mm: Optional override for Y offset of rectangle centre line (mm).
        marker_size_mm: Optional override for ArUco marker side length (mm).

    Returns:
        Updated configuration dictionary.
    """
    cfg_file = Path(config_path) if config_path else get_settings().calibration_config_path
    out_path = Path(output_dir)
    pan_offset_x = get_pan_origin_offset_x_mm()

    cfg: dict[str, Any] = {}
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not parse existing calibration JSON: {e}")
            cfg = {}

    corners = cfg.get("workspace_corners_mm", {})
    def_marker_size = float(cfg.get("marker_size_mm", REFERENCE_WORKSPACE_MM["marker_size_mm"]))

    # Derive default rectangle parameters from existing corners or the reference setup constant
    def_near_x = REFERENCE_WORKSPACE_MM["near_x_mm"]
    def_depth = REFERENCE_WORKSPACE_MM["depth_mm"]
    def_width = REFERENCE_WORKSPACE_MM["width_mm"]
    def_lateral_offset = REFERENCE_WORKSPACE_MM["lateral_offset_mm"]

    if corners and all(
        k in corners for k in ("top_left", "top_right", "bottom_right", "bottom_left")
    ):
        tl_p = base_to_pan_xy_mm(float(corners["top_left"]["x"]), float(corners["top_left"]["y"]))
        tr_p = base_to_pan_xy_mm(float(corners["top_right"]["x"]), float(corners["top_right"]["y"]))
        br_p = base_to_pan_xy_mm(
            float(corners["bottom_right"]["x"]), float(corners["bottom_right"]["y"])
        )
        bl_p = base_to_pan_xy_mm(
            float(corners["bottom_left"]["x"]), float(corners["bottom_left"]["y"])
        )

        # Check if existing corners form an axis-aligned rectangle within 2 mm
        is_rect = (
            abs(tl_p[0] - bl_p[0]) <= 2.0
            and abs(tr_p[0] - br_p[0]) <= 2.0
            and abs(tl_p[1] - tr_p[1]) <= 2.0
            and abs(bl_p[1] - br_p[1]) <= 2.0
        )
        if not is_rect:
            warn_msg = (
                "⚠️ Warning: Existing workspace corners are not an axis-aligned rectangle within "
                "2 mm; they will be replaced by their bounding rectangle."
            )
            print(f"\n{warn_msg}\n")
            logger.warning(warn_msg)

        all_xs = [tl_p[0], tr_p[0], br_p[0], bl_p[0]]
        all_ys = [tl_p[1], tr_p[1], br_p[1], bl_p[1]]
        def_near_x = min(all_xs)
        def_depth = max(all_xs) - def_near_x
        def_right_y = min(all_ys)
        def_left_y = max(all_ys)
        def_width = def_left_y - def_right_y
        def_lateral_offset = (def_left_y + def_right_y) / 2.0

    # Print banner before any prompt (including the reuse prompt)
    print("\n" + "=" * 67)
    print("🎯 SO-101 ARUCO WORKSPACE SETUP (RECTANGLE-ONLY)")
    print("=" * 67)
    print(
        "NOTE: Rectangle only, axis-aligned with the robot (+X forward along HOME,\n"
        "+Y to robot's left). Free-form or rotated layouts are not supported.\n"
        "\n"
        "Measurement Datum & Axis Conventions:\n"
        "  • Dimension origin (0, 0): Shoulder-pan rotation axis (tape measure datum)\n"
        "  • +X: Forward along HOME pose\n"
        "  • +Y: Robot left, -Y: Robot right\n"
        f"  • Robot base offset: +{pan_offset_x:.1f} mm in X is added automatically to\n"
        "    record corner coordinates in the robot base frame."
    )
    print(
        f"  Axis diagram:  [Base Origin] ───(+{pan_offset_x:.1f} mm)───► [Pan Axis (0,0)] "
        "───► +X (Forward along HOME)\n"
        "                 ▲ +Y (Robot Left) / ▼ -Y (Robot Right)\n"
        "==================================================================="
    )

    # Check existing marker files in output directory
    marker_files = [out_path / f"marker_{i}.png" for i in _MARKER_IDS_ALL]
    guide_file = out_path / "workspace_placement_guide.png"
    existing_found = [f for f in marker_files if f.exists()]
    has_existing = len(existing_found) > 0 or guide_file.exists()

    # Determine reuse vs regenerate
    should_reuse = False
    if has_existing:
        if reuse is not None:
            should_reuse = reuse
        elif interactive and sys.stdin.isatty():
            print(f"🔍 Found existing marker files in '{output_dir}':")
            for ef in existing_found:
                print(f"   • {ef.name}")
            if guide_file.exists():
                print(f"   • {guide_file.name}")

            choice = (
                input(
                    "\n❓ Do you want to reuse existing markers or regenerate/replace them?\n"
                    "   [1] Reuse existing markers (update dimensions in JSON only) [DEFAULT]\n"
                    "   [2] Regenerate / overwrite marker images and placement guide\n"
                    "   Choice [1/2 or reuse/regenerate]: "
                )
                .strip()
                .lower()
            )
            should_reuse = choice in ("", "1", "reuse", "r", "y", "yes", "keep")
        else:
            should_reuse = True

    def _prompt_float(
        prompt_text: str,
        default_val: float,
        validator: Callable[[float], tuple[bool, str]],
    ) -> float:
        """Prompt user with validation loop; asks again on bad input without silent fallback."""
        if not interactive or not sys.stdin.isatty():
            return default_val
        while True:
            try:
                raw = input(f"  ➜ {prompt_text} [default: {default_val:g}]: ").strip()
                val = default_val if not raw else float(raw)
            except (EOFError, KeyboardInterrupt):
                return default_val
            except ValueError:
                print("     ❌ Invalid number. Please enter a valid numeric value.")
                continue

            valid, reason = validator(val)
            if valid:
                return val
            print(f"     ❌ Invalid input: {reason}. Please try again.")

    # 1. Near X
    est_marker_size = marker_size_mm if marker_size_mm is not None else def_marker_size
    min_near_x = est_marker_size / 2.0 + 60.0

    def _val_near_x(val: float) -> tuple[bool, str]:
        if val <= min_near_x:
            return (
                False,
                f"near X must be greater than marker size / 2 + 60 mm ({min_near_x:.1f} mm) "
                "to clear the robot base.",
            )
        return True, ""

    if near_x_mm is not None:
        near_x = float(near_x_mm)
    else:
        if interactive and sys.stdin.isatty():
            print("\n1. Workspace Rectangle Measurements (mm to marker centres from pan axis):")
        near_x = _prompt_float(
            "Near X (distance forward from pan axis to near markers)",
            def_near_x,
            _val_near_x,
        )

    # 2. Depth
    def _val_depth(val: float) -> tuple[bool, str]:
        if val <= 0.0:
            return False, "depth must be positive."
        return True, ""

    if depth_mm is not None:
        depth = float(depth_mm)
    else:
        depth = _prompt_float(
            "Depth (distance from near markers to far markers)",
            def_depth,
            _val_depth,
        )

    # 3. Width
    def _val_width(val: float) -> tuple[bool, str]:
        if val <= 0.0:
            return False, "width must be positive."
        return True, ""

    if width_mm is not None:
        width = float(width_mm)
    else:
        width = _prompt_float(
            "Width (distance from right markers to left markers)",
            def_width,
            _val_width,
        )

    # 4. Lateral offset
    if lateral_offset_mm is not None:
        lateral_offset = float(lateral_offset_mm)
    else:
        lateral_offset = _prompt_float(
            "Lateral offset (Y of centre line: 0 = centred, negative = robot right)",
            def_lateral_offset,
            lambda _: (True, ""),
        )

    # 5. Marker size
    def _val_marker_size(val: float) -> tuple[bool, str]:
        if val <= 0.0:
            return False, "marker size must be positive."
        if near_x <= val / 2.0 + 60.0:
            return False, f"marker size too large: near X ({near_x:.1f} mm) <= size/2 + 60 mm."
        return True, ""

    if marker_size_mm is not None:
        marker_size = float(marker_size_mm)
    else:
        if interactive and sys.stdin.isatty():
            print("\n2. ArUco Marker Parameters:")
        marker_size = _prompt_float(
            "Marker size (mm)",
            def_marker_size,
            _val_marker_size,
        )

    # Compute four centres in pan frame
    far_x = round(near_x + depth, 1)
    near_x_r = round(near_x, 1)
    right_y = round(lateral_offset - width / 2.0, 1)
    left_y = round(lateral_offset + width / 2.0, 1)

    corners_pan: dict[str, tuple[float, float]] = {
        "top_left": (far_x, right_y),
        "top_right": (near_x_r, right_y),
        "bottom_right": (near_x_r, left_y),
        "bottom_left": (far_x, left_y),
    }

    # Convert to robot base frame (+pan_offset_x mm in X)
    tl_base = pan_to_base_xy_mm(*corners_pan["top_left"])
    tr_base = pan_to_base_xy_mm(*corners_pan["top_right"])
    br_base = pan_to_base_xy_mm(*corners_pan["bottom_right"])
    bl_base = pan_to_base_xy_mm(*corners_pan["bottom_left"])

    workspace_corners = {
        "top_left": {"x": tl_base[0], "y": tl_base[1]},
        "top_right": {"x": tr_base[0], "y": tr_base[1]},
        "bottom_right": {"x": br_base[0], "y": br_base[1]},
        "bottom_left": {"x": bl_base[0], "y": bl_base[1]},
    }

    # Evaluate reachability (load CartesianIK once)
    print("\n🔍 Evaluating arm workspace reachability (30 mm hover)...")
    ik = CartesianIK()
    reach_fraction, out_of_reach, _ = evaluate_workspace_reachability(
        workspace_corners, step_mm=40.0, ik=ik
    )

    print(f"  • Reachable area: {reach_fraction * 100.0:.1f}% of workspace rectangle.")
    if out_of_reach:
        formatted_names = ", ".join(out_of_reach)
        print(
            f"  • Markers outside reach: {formatted_names}\n"
            "    (Markers out of reach are expected and fine; only the camera needs to see them.)"
        )
    else:
        print("  • All 4 workspace marker positions are within reach.")

    if reach_fraction < 0.5:
        print("  ⚠️ WARNING: Less than 50% of the workspace rectangle is reachable by the arm.")

    dictionary_name = str(cfg.get("aruco_dictionary", "DICT_4X4_50"))
    marker_ids: dict[str, int] = cfg.get(
        "marker_ids",
        {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 0},
    )

    # Build updated configuration
    updated_config: dict[str, Any] = {
        "aruco_dictionary": dictionary_name,
        "marker_ids": marker_ids,
        "marker_size_mm": marker_size,
        "workspace_corners_mm": workspace_corners,
    }

    # Save JSON configuration
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps(updated_config, indent=2) + "\n")
    print(f"\n✅ Workspace calibration profile saved to: {cfg_file}")

    # Generate placement guide diagram
    print("\n📄 Rendering workspace placement guide (3508x2480 A4 diagram)...")
    guide_path = generate_placement_guide(
        corners_pan_mm=corners_pan,
        marker_size_mm=marker_size,
        dictionary_name=dictionary_name,
        pan_offset_x_mm=pan_offset_x,
        output_dir=output_dir,
        marker_ids=marker_ids,
    )
    print(f"   ✔ Placement Guide: {guide_path}")

    # Generate marker tiles and jaw tag sheet
    if not should_reuse:
        print(f"\n📦 Generating printable ArUco markers in '{output_dir}'...")
        created_markers = generate_aruco_markers(output_dir=output_dir, marker_ids=_MARKER_IDS_ALL)
        for p in created_markers:
            print(f"   ✔ Marker tile:   {p}")

        jaw_sheet_path = generate_jaw_tag_sheet(output_dir=output_dir)
        print(f"   ✔ Jaw Tag Sheet: {jaw_sheet_path}")
    else:
        created_markers = [out_path / f"marker_{i}.png" for i in _MARKER_IDS_ALL]
        jaw_sheet_path = out_path / "jaw_tag_sheet.png"
        print(f"\n♻️  Reusing existing marker images in '{output_dir}'.")

    marker_ids_str = ",".join(str(p.stem.split("_")[1]) for p in created_markers)
    # Step 6: Final wizard output & Next steps
    print("\n" + "=" * 67)
    print("🎉 WORKSPACE CALIBRATION SETUP COMPLETE!")
    print("=" * 67)
    print("Files written / updated:")
    print(f"  • Calibration JSON:     {cfg_file}")
    print(f"  • Placement Guide:      {guide_path}")
    print(f"  • Marker PNG Tiles:     {output_dir}/marker_{{{marker_ids_str}}}.png")
    print(f"  • Jaw Tag Sheet:        {jaw_sheet_path}")
    print("\nNext Steps:")
    print("  1. Print the placement guide and the ArUco marker tiles.")
    print("  2. Place the marker centres with a tape measure from the pan axis.")
    print("  3. Run `uv run python main.py` and enable '📐 Show Calibration Overlay' to verify.")
    print("  4. Compare the `[Calibration] [CAMERA] height_mm` log with your measured lens height.")
    print("=" * 67 + "\n")

    return updated_config
