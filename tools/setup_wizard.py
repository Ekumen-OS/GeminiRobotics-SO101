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

    corners = cfg.get("workspace_corners_mm", {})
    cur_x_max = float(corners.get("top_left", {}).get("x", 380))
    cur_x_min = float(corners.get("bottom_left", {}).get("x", 120))
    cur_y_max = float(corners.get("top_left", {}).get("y", 280))
    cur_y_min = float(corners.get("top_right", {}).get("y", -200))
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
