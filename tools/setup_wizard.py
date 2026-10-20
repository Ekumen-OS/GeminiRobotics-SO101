"""Interactive wizard for workspace setup and ArUco marker configuration."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from robot.calibration import ArucoWorkspaceCalibrator
from robot.settings import get_settings
from tools.calibration_sheet import (
    generate_aruco_markers,
    generate_calibration_sheet,
    generate_jaw_tag_sheet,
)

logger = logging.getLogger(__name__)


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
    import mujoco

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


def setup_workspace_interactive(
    config_path: str | Path | None = None,
    output_dir: str | Path = "assets/aruco_markers",
    interactive: bool = True,
    reuse: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Interactive / CLI wizard to configure workspace dimensions and handle ArUco markers.

    Detects existing markers in the output directory. If markers exist, asks the user whether
    to reuse existing physical markers (updating only physical dimensions in JSON) or regenerate
    and replace them. If no markers exist, generates new ones automatically.

    Args:
        config_path: Target JSON config file path. Defaults to config/workspace_calibration.json.
        output_dir: Folder to save or search for marker PNGs and sheet.
        interactive: If True and running in a terminal, prompts the user for inputs.
        reuse: If True, reuse existing markers without regenerating files. If False, force
            regeneration. If None, asks the user interactively (or auto-detects).
        **kwargs: Optional dimension overrides (e.g. x_max, x_min, y_max, y_min, marker_size).

    Returns:
        Updated configuration dictionary.
    """
    cfg_file = Path(config_path) if config_path else get_settings().calibration_config_path
    calibrator = ArucoWorkspaceCalibrator(config_path=cfg_file)
    cfg = dict(calibrator.config)
    out_path = Path(output_dir)

    pan_offset_x = get_pan_origin_offset_x_mm()
    corners = cfg.get("workspace_corners_mm", {})
    cur_tl = corners.get("top_left", {"x": round(414.0 + pan_offset_x, 1), "y": -240.0})
    cur_tr = corners.get("top_right", {"x": round(97.0 + pan_offset_x, 1), "y": -240.0})
    cur_br = corners.get("bottom_right", {"x": round(97.0 + pan_offset_x, 1), "y": 102.0})
    cur_bl = corners.get("bottom_left", {"x": round(414.0 + pan_offset_x, 1), "y": 102.0})
    cur_marker_size = float(cfg.get("marker_size_mm", 40))

    # Check for existing markers in output directory
    marker_files = [out_path / f"marker_{i}.png" for i in range(4)]
    sheet_file = out_path / "workspace_calibration_sheet.png"
    existing_found = [f for f in marker_files if f.exists()]
    has_existing = len(existing_found) > 0 or sheet_file.exists()

    def _prompt(text: str, default_val: Any, val_type: type = float) -> Any:
        """Prompt user for input with default fallback.

        Args:
            text: Prompt description string.
            default_val: Default value to return if skipped or non-interactive.
            val_type: Target type to cast user input to.

        Returns:
            User input parsed as val_type, or default_val.
        """
        if not interactive or not sys.stdin.isatty():
            return default_val
        try:
            val = input(f"  ➜ {text} [default: {default_val}]: ").strip()
            if not val:
                return default_val
            return val_type(val)
        except (EOFError, KeyboardInterrupt):
            return default_val

    print("\n" + "=" * 65)
    print("🎯 SO-101 ARUCO WORKSPACE SETUP & CALIBRATION WIZARD")
    print("=" * 65)

    # 1. Determine whether to reuse existing markers or regenerate
    should_reuse = False
    if has_existing:
        print(f"🔍 Found existing marker files in '{output_dir}':")
        for ef in existing_found:
            print(f"   • {ef.name}")
        if sheet_file.exists():
            print(f"   • {sheet_file.name}")

        if reuse is not None:
            should_reuse = reuse
        elif interactive and sys.stdin.isatty():
            choice = (
                input(
                    "\n❓ Do you want to reuse existing markers or regenerate/replace them?\n"
                    "   [1] Reuse existing markers (update dimensions in JSON only) [DEFAULT]\n"
                    "   [2] Regenerate / overwrite marker images and calibration sheet\n"
                    "   Choice [1/2 or reuse/regenerate]: "
                )
                .strip()
                .lower()
            )
            should_reuse = choice in ("", "1", "reuse", "r", "y", "yes", "keep")
        else:
            should_reuse = True
    else:
        print(f"ℹ️  No existing markers found in '{output_dir}'. New markers will be generated.")
        should_reuse = False

    if should_reuse:
        print("\n♻️  REUSING EXISTING MARKERS — Updating physical table dimensions in JSON...\n")
    else:
        print("\n📦 NEW MARKER GENERATION — Configuring dimensions and generating files...\n")

    print(
        "Measure the 4 marker centres from the shoulder pan rotation axis (mm):\n"
        "  Origin (0, 0) for tape measurements is at the shoulder pan rotation axis,\n"
        f"  +X forward along HOME, +Y to the robot's left.\n"
        f"  Note: The robot base frame origin is {pan_offset_x:.1f} mm behind the pan axis.\n"
        f"  The wizard asks for tape values from the pan axis and automatically adds "
        f"+{pan_offset_x:.1f} mm\n"
        "  in X to record corners in the robot base frame."
    )
    print(
        f"  Axis diagram:  [Base Origin] ───(+{pan_offset_x:.1f} mm)───► [Pan Axis (0,0)] "
        "───► +X (Forward along HOME)\n"
        "                 ▲ +Y (Robot Left) / ▼ -Y (Robot Right)\n"
    )

    def _parse_xy_pair(raw: str, default_xy: tuple[float, float]) -> tuple[float, float]:
        parts = [float(p.strip()) for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("Expected 'x, y'")
        return (parts[0], parts[1])

    def _prompt_xy(label: str, default_xy: tuple[float, float]) -> tuple[float, float]:
        if not interactive or not sys.stdin.isatty():
            return default_xy
        default_str = f"{default_xy[0]:g}, {default_xy[1]:g}"
        try:
            val = input(f"  ➜ {label} (x, y mm from pan axis) [default: {default_str}]: ").strip()
            if not val:
                return default_xy
            return _parse_xy_pair(val, default_xy)
        except (EOFError, KeyboardInterrupt, ValueError):
            return default_xy

    tl_pan_default = base_to_pan_xy_mm(
        float(cur_tl.get("x", 414.0 + pan_offset_x)), float(cur_tl.get("y", -240.0))
    )
    tr_pan_default = base_to_pan_xy_mm(
        float(cur_tr.get("x", 97.0 + pan_offset_x)), float(cur_tr.get("y", -240.0))
    )
    br_pan_default = base_to_pan_xy_mm(
        float(cur_br.get("x", 97.0 + pan_offset_x)), float(cur_br.get("y", 102.0))
    )
    bl_pan_default = base_to_pan_xy_mm(
        float(cur_bl.get("x", 414.0 + pan_offset_x)), float(cur_bl.get("y", 102.0))
    )

    print("1. Marker Centres measured from Shoulder Pan Rotation Axis (x, y in mm):")
    tl_pan = _prompt_xy("top_left (ID 1)", tl_pan_default)
    tr_pan = _prompt_xy("top_right (ID 2)", tr_pan_default)
    br_pan = _prompt_xy("bottom_right (ID 3)", br_pan_default)
    bl_pan = _prompt_xy("bottom_left (ID 0)", bl_pan_default)

    # Convert tape measurements (from pan axis) to robot base frame (+pan_offset_x mm in X)
    tl_xy = pan_to_base_xy_mm(tl_pan[0], tl_pan[1])
    tr_xy = pan_to_base_xy_mm(tr_pan[0], tr_pan[1])
    br_xy = pan_to_base_xy_mm(br_pan[0], br_pan[1])
    bl_xy = pan_to_base_xy_mm(bl_pan[0], bl_pan[1])

    # 2. Marker dimensions
    print("\n2. ArUco Marker Parameters:")
    marker_size = kwargs.get("marker_size", _prompt("Marker size (mm)", cur_marker_size))

    has_bbox_overrides = any(k in kwargs for k in ("x_max", "x_min", "y_max", "y_min"))
    has_direct_corner_overrides = any(
        k in kwargs for k in ("top_left", "top_right", "bottom_right", "bottom_left")
    )
    if has_direct_corner_overrides:
        workspace_corners = {
            "top_left": kwargs.get("top_left", {"x": tl_xy[0], "y": tl_xy[1]}),
            "top_right": kwargs.get("top_right", {"x": tr_xy[0], "y": tr_xy[1]}),
            "bottom_right": kwargs.get("bottom_right", {"x": br_xy[0], "y": br_xy[1]}),
            "bottom_left": kwargs.get("bottom_left", {"x": bl_xy[0], "y": bl_xy[1]}),
        }
    elif has_bbox_overrides:
        x_max = kwargs.get("x_max", max(tl_xy[0], bl_xy[0]))
        x_min = kwargs.get("x_min", min(tr_xy[0], br_xy[0]))
        y_max = kwargs.get("y_max", max(tl_xy[1], bl_xy[1]))
        y_min = kwargs.get("y_min", min(tr_xy[1], br_xy[1]))
        workspace_corners = {
            "top_left": {"x": x_max, "y": y_max},
            "top_right": {"x": x_max, "y": y_min},
            "bottom_right": {"x": x_min, "y": y_min},
            "bottom_left": {"x": x_min, "y": y_max},
        }
    elif (interactive and sys.stdin.isatty()) or not corners:
        workspace_corners = {
            "top_left": {"x": tl_xy[0], "y": tl_xy[1]},
            "top_right": {"x": tr_xy[0], "y": tr_xy[1]},
            "bottom_right": {"x": br_xy[0], "y": br_xy[1]},
            "bottom_left": {"x": bl_xy[0], "y": bl_xy[1]},
        }
    else:
        workspace_corners = corners

    # Build updated config
    updated_config = {
        "aruco_dictionary": cfg.get("aruco_dictionary", "DICT_4X4_50"),
        "marker_ids": cfg.get(
            "marker_ids",
            {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 0},
        ),
        "marker_size_mm": marker_size,
        "workspace_corners_mm": workspace_corners,
    }

    # Save JSON configuration
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps(updated_config, indent=2) + "\n")
    print(f"\n✅ Workspace calibration profile saved to: {cfg_file}")

    if not should_reuse:
        # Generate individual marker PNGs
        print(f"\n📦 Generating printable ArUco markers into '{output_dir}'...")
        created_markers = generate_aruco_markers(output_dir=output_dir, marker_ids=[0, 1, 2, 3, 10])
        for p in created_markers:
            print(f"   ✔ Created: {p}")

        # Generate combined calibration sheet
        print("\n📄 Rendering true-to-scale calibration sheets...")
        created_sheet = generate_calibration_sheet(
            config_data=updated_config,
            output_dir=output_dir,
        )
        print(f"   ✔ Workspace Calibration Sheet: {created_sheet}")

        # Generate standalone Jaw Tag Sheet
        created_jaw_sheet = generate_jaw_tag_sheet(output_dir=output_dir)
        print(f"   ✔ Jaw Tag Marker Sheet:        {created_jaw_sheet}")

        print("\n" + "=" * 65)
        print("🎉 WORKSPACE CALIBRATION SETUP COMPLETE!")
        print("=" * 65)
        print(f"  • Marker PNGs generated in: {output_dir}")
        print(f"  • Full printable sheet:     {created_sheet}")
        print(f"  • Jaw Tag Sheet:            {created_jaw_sheet}")
        print(f"  • Physical JSON profile:    {cfg_file}")
    else:
        print("\n" + "=" * 65)
        print("🎉 WORKSPACE DIMENSIONS UPDATED SUCCESSFULLY!")
        print("=" * 65)
        print("  • Physical JSON profile updated.")
        print(f"  • Reusing existing markers in: {output_dir}")

    print("\nNext Steps:")
    print("  1. Place markers on tabletop at the specified dimensions.")
    print("  2. Run 'uv run python main.py' and enable 'Show Calibration Overlay'.\n")

    return updated_config
