"""Interactive wizard for workspace setup and ArUco marker configuration."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from robot.calibration import ArucoWorkspaceCalibrator, _get_default_config_path
from tools.calibration_sheet import (
    generate_aruco_markers,
    generate_calibration_sheet,
    generate_jaw_tag_sheet,
)

logger = logging.getLogger(__name__)


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
    cfg_file = Path(config_path) if config_path else _get_default_config_path()
    calibrator = ArucoWorkspaceCalibrator(config_path=cfg_file)
    cfg = dict(calibrator.config)
    out_path = Path(output_dir)

    corners = cfg.get("workspace_corners_mm", {})
    cur_x_max = float(corners.get("top_left", {}).get("x", 380))
    cur_x_min = float(corners.get("bottom_left", {}).get("x", 120))
    cur_y_max = float(corners.get("top_left", {}).get("y", 280))
    cur_y_min = float(corners.get("top_right", {}).get("y", -200))
    cur_marker_size = float(cfg.get("marker_size_mm", 40))
    cur_hover_z = float(cfg.get("default_hover_z_mm", 30))
    cur_grasp_z = float(cfg.get("default_grasp_z_mm", 5))
    cur_dest_bin = cfg.get("default_destination_bin_xyz_mm", {"x": 180, "y": 250, "z": 30})

    # Check for existing markers in output directory
    marker_files = [out_path / f"marker_{i}.png" for i in range(4)]
    sheet_file = out_path / "workspace_calibration_sheet.png"
    existing_found = [f for f in marker_files if f.exists()]
    has_existing = len(existing_found) > 0 or sheet_file.exists()

    def _prompt(text: str, default_val: Any, val_type: type = float) -> Any:
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

    # 2. Forward range (X)
    print("1. Forward Depth Range (Robot X axis — distance from base in mm):")
    x_max = kwargs.get("x_max", _prompt("Far edge distance (X Max mm)", cur_x_max))
    x_min = kwargs.get("x_min", _prompt("Near edge distance (X Min mm)", cur_x_min))

    # 3. Lateral range (Y)
    print("\n2. Lateral Width Range (Robot Y axis — left is positive, right is negative in mm):")
    y_max = kwargs.get("y_max", _prompt("Left edge distance (Y Max mm)", cur_y_max))
    y_min = kwargs.get("y_min", _prompt("Right edge distance (Y Min mm)", cur_y_min))

    # 4. Marker dimensions
    print("\n3. ArUco Marker Parameters:")
    marker_size = kwargs.get("marker_size", _prompt("Marker size (mm)", cur_marker_size))

    # 5. Vertical clearances
    print("\n4. Operating Heights (mm):")
    hover_z = kwargs.get("hover_z", _prompt("Default hover clearance height (Z mm)", cur_hover_z))
    grasp_z = kwargs.get(
        "grasp_z", _prompt("Default grasp table surface height (Z mm)", cur_grasp_z)
    )

    has_corner_overrides = any(k in kwargs for k in ("x_max", "x_min", "y_max", "y_min")) or (
        interactive and sys.stdin.isatty()
    )
    if has_corner_overrides or not corners:
        workspace_corners = {
            "top_left": {"x": x_max, "y": y_max},
            "top_right": {"x": x_max, "y": y_min},
            "bottom_right": {"x": x_min, "y": y_min},
            "bottom_left": {"x": x_min, "y": y_max},
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
        "default_hover_z_mm": hover_z,
        "default_grasp_z_mm": grasp_z,
        "default_destination_bin_xyz_mm": cur_dest_bin,
    }

    # Save JSON configuration
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps(updated_config, indent=2))
    print(f"\n✅ Workspace calibration profile saved to: {cfg_file}")

    if not should_reuse:
        # Generate individual marker PNGs
        print(f"\n📦 Generating printable ArUco markers into '{output_dir}'...")
        created_markers = generate_aruco_markers(output_dir=output_dir, marker_ids=[0, 1, 2, 3, 10])

        # Generate all-in-one calibration mat sheet
        created_sheet = generate_calibration_sheet(
            config_data=updated_config, output_dir=output_dir
        )

        # Generate the 25 mm printable jaw tag sheet
        created_jaw_sheet = generate_jaw_tag_sheet(output_dir=output_dir)

        print("\n" + "=" * 65)
        print("🎉 WORKSPACE SETUP & MARKER GENERATION COMPLETE!")
        print("=" * 65)
        print("Generated Artifacts:")
        for mf in created_markers:
            print(f"  • Individual Marker : {mf}")
        print(f"  • Full Printable Mat: {created_sheet}")
        print(f"  • Jaw Tag Sheet     : {created_jaw_sheet}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SO-101 ArUco Workspace Calibration & Setup Utility"
    )
    parser.add_argument(
        "--generate-markers",
        action="store_true",
        help="Run interactive workspace setup and generate markers & sheet.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/aruco_markers",
        help="Directory to save generated markers.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing markers in output directory without regenerating images.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration and overwrite of all marker images.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use existing config without prompting.",
    )
    args = parser.parse_args()

    reuse_flag = True if args.reuse else (False if args.regenerate else None)

    setup_workspace_interactive(
        output_dir=args.output_dir,
        interactive=not args.non_interactive,
        reuse=reuse_flag,
    )
