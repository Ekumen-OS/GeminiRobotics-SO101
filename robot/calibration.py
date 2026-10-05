"""ArUco workspace calibration module for SO-101 robot arm.

Detects 4-corner fiducial markers (DICT_4X4_50) defining the physical workspace mat,
computes perspective homography, and transforms Gemini normalized camera coordinates
directly to physical millimeter/meter coordinates in the robot base frame.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot.settings import get_settings

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    """Dynamically resolve configuration defaults from central Settings."""
    if name == "DEFAULT_CONFIG_PATH":
        return get_settings().calibration_config_path
    if name == "DEFAULT_SAVE_PATH":
        return get_settings().calibration_save_path
    if name == "DEFAULT_SIM_SAVE_PATH":
        return get_settings().calibration_sim_save_path
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _get_default_config_path() -> Path:
    return getattr(sys.modules[__name__], "DEFAULT_CONFIG_PATH")


def _get_default_save_path() -> Path:
    return getattr(sys.modules[__name__], "DEFAULT_SAVE_PATH")


def _get_default_sim_save_path() -> Path:
    return getattr(sys.modules[__name__], "DEFAULT_SIM_SAVE_PATH")


class ArucoWorkspaceCalibrator:
    """Calibrates and maps camera pixel coordinates to physical robot base frame coordinates.

    Uses 4-corner ArUco fiducials to calculate a perspective homography matrix.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        save_path: str | Path | None = None,
        mode: str = "real",
    ) -> None:
        """Initialize ArUco workspace calibrator.

        Args:
            config_path: Path to workspace calibration JSON configuration.
            save_path: Optional path to saved homography (.npz) file.
            mode: Operating mode ('real', 'sim', 'twin').
        """
        self.config_path = Path(config_path) if config_path else _get_default_config_path()
        self.mode = mode
        self.config: dict[str, Any] = {}
        self.homography_matrix: np.ndarray | None = None
        self.inverse_homography: np.ndarray | None = None
        self.reprojection_error: float | None = None
        self.last_detected_markers: dict[int, np.ndarray] = {}

        self._load_config()
        self._init_aruco()

        # Determine target calibration file path based on mode
        target_save_path = (
            Path(save_path)
            if save_path
            else (_get_default_sim_save_path() if mode == "sim" else _get_default_save_path())
        )

        # Try to load existing saved homography if available
        if target_save_path.exists():
            self.load_calibration(target_save_path)
        elif mode != "sim":
            warnings.warn(
                f"[ArucoCalibrator] ⚠️ WARNING: No saved homography found at '{target_save_path}'. "
                "Workspace must be calibrated before executing real-world motions.",
                UserWarning,
                stacklevel=2,
            )

    def _load_config(self) -> None:
        """Load workspace configuration from JSON file."""
        if self.config_path.exists():
            try:
                self.config = json.loads(self.config_path.read_text())
                return
            except Exception as e:
                warnings.warn(
                    f"[ArucoCalibrator] ⚠️ WARNING: Error parsing config '{self.config_path}': {e}",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"[ArucoCalibrator] ⚠️ WARNING: Calibration config not found at "
                f"'{self.config_path}'. No defaults assumed. Calibration required.",
                UserWarning,
                stacklevel=2,
            )

        # Minimal fallback configuration without assumed physical dimensions
        self.config = {
            "aruco_dictionary": "DICT_4X4_50",
            "marker_ids": {
                "top_left": 1,
                "top_right": 2,
                "bottom_right": 3,
                "bottom_left": 0,
            },
        }

    def _init_aruco(self) -> None:
        """Initialize OpenCV ArUco dictionary and detector parameters."""
        dict_name = self.config.get("aruco_dictionary", "DICT_4X4_50")
        dict_attr = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
        self.detector_params = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        else:
            self.detector = None

    @property
    def is_calibrated(self) -> bool:
        """Return True if a valid perspective homography matrix is available."""
        return self.homography_matrix is not None

    def detect_markers(self, frame_rgb: np.ndarray) -> dict[int, np.ndarray]:
        """Detect ArUco fiducial markers in an RGB frame.

        Args:
            frame_rgb: Input RGB image array of shape (H, W, 3).

        Returns:
            Dictionary mapping integer marker ID to (4, 2) array of corner pixel coordinates.
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )

        detected: dict[int, np.ndarray] = {}
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                detected[int(marker_id)] = corners[i].reshape(4, 2)

        self.last_detected_markers = detected
        return detected

    def update_homography(
        self,
        detected_markers: dict[int, np.ndarray] | None = None,
        frame_rgb: np.ndarray | None = None,
    ) -> bool:
        """Calculate or update homography from detected ArUco markers to physical coordinates.

        Args:
            detected_markers: Optional pre-detected markers dictionary.
            frame_rgb: Optional RGB frame to run detection on if markers not provided.

        Returns:
            True if homography was successfully computed and updated, False otherwise.
        """
        if detected_markers is None:
            if frame_rgb is None:
                return False
            detected_markers = self.detect_markers(frame_rgb)

        marker_ids = self.config.get("marker_ids", {})
        corners_cfg = self.config.get("workspace_corners_mm", {})

        expected_pairs: list[tuple[str, int]] = [
            ("top_left", int(marker_ids.get("top_left", 1))),
            ("top_right", int(marker_ids.get("top_right", 2))),
            ("bottom_right", int(marker_ids.get("bottom_right", 3))),
            ("bottom_left", int(marker_ids.get("bottom_left", 0))),
        ]

        src_pixels: list[list[float]] = []
        dst_robot_mm: list[list[float]] = []

        for corner_name, marker_id in expected_pairs:
            if marker_id in detected_markers and corner_name in corners_cfg:
                corner_px = detected_markers[marker_id]
                center_px = np.mean(corner_px, axis=0)
                src_pixels.append([float(center_px[0]), float(center_px[1])])

                target_pt = corners_cfg[corner_name]
                dst_robot_mm.append([float(target_pt["x"]), float(target_pt["y"])])

        if len(src_pixels) < 4:
            return False

        src_arr = np.array(src_pixels, dtype=np.float32)
        dst_arr = np.array(dst_robot_mm, dtype=np.float32)

        try:
            h_mat, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
            if h_mat is None:
                h_mat = cv2.getPerspectiveTransform(src_arr[:4], dst_arr[:4])

            if h_mat is not None:
                self.homography_matrix = h_mat
                self.inverse_homography = np.linalg.inv(h_mat)

                # Compute reprojection error
                src_homo = np.hstack([src_arr, np.ones((len(src_arr), 1), dtype=np.float32)])
                proj = (h_mat @ src_homo.T).T
                proj_xy = proj[:, :2] / proj[:, 2:3]
                err = float(np.mean(np.linalg.norm(proj_xy - dst_arr, axis=1)))
                self.reprojection_error = err
                return True
        except Exception as e:
            logger.error(f"[ArucoCalibrator] Homography computation error: {e}")

        return False

    def gemini_to_pixel(
        self,
        point_1000: list[float] | np.ndarray,
        img_shape: tuple[int, int] = (480, 640),
    ) -> tuple[float, float]:
        """Convert Gemini normalized [y, x] (0-1000) coordinates to pixel (px, py).

        Args:
            point_1000: Normalized [y, x] coordinate from Gemini (range 0 to 1000).
            img_shape: Frame dimensions as (height, width).

        Returns:
            Tuple of (px, py) float pixel coordinates.
        """
        h, w = img_shape
        y_norm = float(point_1000[0]) / 1000.0
        x_norm = float(point_1000[1]) / 1000.0
        px = x_norm * w
        py = y_norm * h
        return px, py

    def pixel_to_robot_mm(self, px: float, py: float) -> np.ndarray:
        """Project camera pixel coordinates to physical robot base frame (mm).

        Args:
            px: Horizontal pixel coordinate.
            py: Vertical pixel coordinate.

        Returns:
            NumPy array of shape (2,) containing [X_mm, Y_mm] in robot frame.
        """
        if self.homography_matrix is None:
            raise RuntimeError("Workspace is not calibrated. Homography matrix is unavailable.")

        pt_homo = np.array([px, py, 1.0], dtype=np.float64)
        projected = self.homography_matrix @ pt_homo
        if abs(projected[2]) < 1e-8:
            return np.array([0.0, 0.0], dtype=np.float64)

        x_mm = projected[0] / projected[2]
        y_mm = projected[1] / projected[2]
        return np.array([x_mm, y_mm], dtype=np.float64)

    def pixel_to_robot_xyz(
        self,
        px: float,
        py: float,
        z_m: float = 0.03,
    ) -> np.ndarray:
        """Project pixel coordinates to physical 3D robot base frame coordinates (in meters).

        Args:
            px: Horizontal pixel coordinate.
            py: Vertical pixel coordinate.
            z_m: Target table height coordinate in meters.

        Returns:
            NumPy array of shape (3,) containing [X, Y, Z] in meters for robot IK.
        """
        xy_mm = self.pixel_to_robot_mm(px, py)
        return np.array([xy_mm[0] / 1000.0, xy_mm[1] / 1000.0, z_m], dtype=np.float64)

    def gemini_to_robot_xyz(
        self,
        point_1000: list[float] | np.ndarray,
        img_shape: tuple[int, int] = (480, 640),
        z_m: float = 0.03,
    ) -> np.ndarray:
        """Convert Gemini normalized [y, x] (0-1000) point to physical 3D robot XYZ (in meters).

        Requires an active calibrated homography matrix.

        Args:
            point_1000: Normalized [y, x] in range 0-1000.
            img_shape: Image dimensions as (height, width).
            z_m: Target height coordinate in meters.

        Returns:
            NumPy array of shape (3,) containing [X, Y, Z] in meters.

        Raises:
            RuntimeError: If the workspace is not calibrated.
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Workspace is not calibrated. ArUco calibration or a saved homography (.npz) "
                "file is required before transforming coordinates to robot space."
            )

        px, py = self.gemini_to_pixel(point_1000, img_shape=img_shape)
        return self.pixel_to_robot_xyz(px, py, z_m=z_m)

    def robot_mm_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float] | None:
        """Project physical robot frame (X, Y) mm to camera pixel coordinates (u, v).

        Args:
            x_mm: Forward coordinate in robot base frame (mm).
            y_mm: Lateral coordinate in robot base frame (mm).

        Returns:
            Tuple of (u, v) pixel coordinates if calibrated, or None.
        """
        if self.inverse_homography is None:
            return None
        pt_robot = np.array([float(x_mm), float(y_mm), 1.0], dtype=np.float64)
        proj = self.inverse_homography @ pt_robot
        if abs(proj[2]) < 1e-8:
            return None
        return float(proj[0] / proj[2]), float(proj[1] / proj[2])

    def draw_workspace_overlay(
        self,
        frame_rgb: np.ndarray,
        detected_markers: dict[int, np.ndarray] | None = None,
        show_axes: bool = True,
        show_markers: bool = True,
        active_target_px: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Draw detected ArUco corners, calibrated workspace boundary, and coordinate axes.

        Args:
            frame_rgb: Input RGB frame image.
            detected_markers: Optional pre-detected markers dict.
            show_axes: Whether to render coordinate frame axes.
            show_markers: Whether to outline detected marker polygons.
            active_target_px: Optional (px, py) pixel coordinate to render target crosshair.

        Returns:
            Annotated RGB frame array.
        """
        annotated = frame_rgb.copy()
        if detected_markers is None:
            detected_markers = self.last_detected_markers

        # 1. Draw detected marker boundaries and IDs
        if show_markers and detected_markers:
            for marker_id, corners in detected_markers.items():
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                center = np.mean(corners, axis=0).astype(np.int32)
                cv2.putText(
                    annotated,
                    f"ID:{marker_id}",
                    (center[0] - 15, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw workspace bounding polygon
        marker_ids = self.config.get("marker_ids", {})
        corners_cfg = self.config.get("workspace_corners_mm", {})
        id_order = [
            int(marker_ids.get("top_left", 1)),
            int(marker_ids.get("top_right", 2)),
            int(marker_ids.get("bottom_right", 3)),
            int(marker_ids.get("bottom_left", 0)),
        ]

        # Draw polygon: use detected corners if all 4 visible, or projected corners if calibrated
        poly_pts = None
        if all(mid in detected_markers for mid in id_order):
            poly_pts = np.array(
                [np.mean(detected_markers[mid], axis=0) for mid in id_order],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
        elif self.is_calibrated:
            pt_tl = self.robot_mm_to_pixel(
                float(corners_cfg.get("top_left", {}).get("x", 380)),
                float(corners_cfg.get("top_left", {}).get("y", -190)),
            )
            pt_tr = self.robot_mm_to_pixel(
                float(corners_cfg.get("top_right", {}).get("x", 100)),
                float(corners_cfg.get("top_right", {}).get("y", -190)),
            )
            pt_br = self.robot_mm_to_pixel(
                float(corners_cfg.get("bottom_right", {}).get("x", 100)),
                float(corners_cfg.get("bottom_right", {}).get("y", 190)),
            )
            pt_bl = self.robot_mm_to_pixel(
                float(corners_cfg.get("bottom_left", {}).get("x", 380)),
                float(corners_cfg.get("bottom_left", {}).get("y", 190)),
            )
            if all(p is not None for p in [pt_tl, pt_tr, pt_br, pt_bl]):
                poly_pts = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype=np.int32).reshape(
                    (-1, 1, 2)
                )

        if poly_pts is not None:
            cv2.polylines(annotated, [poly_pts], isClosed=True, color=(0, 230, 255), thickness=2)

            bl_pt = poly_pts[3][0]
            cv2.putText(
                annotated,
                "CALIBRATED WORKSPACE",
                (bl_pt[0] + 5, bl_pt[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Robot coordinate axes using true homography projection
        # Prefer origin near robot base (top_right, near ruler and robot base)
        # so +X steps into the workspace (Left) and +Y steps towards front of table (Down)
        axis_corner_key = "top_right" if "top_right" in corners_cfg else "bottom_left"
        axis_id = int(marker_ids.get(axis_corner_key, 2))
        axis_phys = corners_cfg.get(axis_corner_key, {"x": 100.0, "y": -190.0})
        ox_mm = float(axis_phys["x"])
        oy_mm = float(axis_phys["y"])

        origin: tuple[float, float] | None = None
        if axis_id in detected_markers:
            mean_pt = np.mean(detected_markers[axis_id], axis=0)
            origin = (float(mean_pt[0]), float(mean_pt[1]))
        elif self.is_calibrated:
            origin = self.robot_mm_to_pixel(ox_mm, oy_mm)

        if show_axes and origin is not None:
            axis_len_px = 65.0
            tip_x: tuple[int, int] | None = None
            tip_y: tuple[int, int] | None = None

            if self.is_calibrated and self.inverse_homography is not None:
                # Step +80mm in physical robot space
                px_x_step = self.robot_mm_to_pixel(ox_mm + 80.0, oy_mm)
                px_y_step = self.robot_mm_to_pixel(ox_mm, oy_mm + 80.0)

                if px_x_step is not None:
                    dx = px_x_step[0] - origin[0]
                    dy = px_x_step[1] - origin[1]
                    norm_x = (dx * dx + dy * dy) ** 0.5
                    if norm_x > 1e-3:
                        tip_x = (
                            int(origin[0] + axis_len_px * dx / norm_x),
                            int(origin[1] + axis_len_px * dy / norm_x),
                        )

                if px_y_step is not None:
                    dx = px_y_step[0] - origin[0]
                    dy = px_y_step[1] - origin[1]
                    norm_y = (dx * dx + dy * dy) ** 0.5
                    if norm_y > 1e-3:
                        tip_y = (
                            int(origin[0] + axis_len_px * dx / norm_y),
                            int(origin[1] + axis_len_px * dy / norm_y),
                        )

            # Fallback if homography unavailable
            if tip_x is None:
                tip_x = (int(origin[0]), int(origin[1] - axis_len_px))
            if tip_y is None:
                tip_y = (int(origin[0] - axis_len_px), int(origin[1]))

            origin_pt = (int(origin[0]), int(origin[1]))

            # +X axis points Forward in robot space - Red
            cv2.arrowedLine(annotated, origin_pt, tip_x, (255, 50, 50), 2, tipLength=0.25)
            cv2.putText(
                annotated,
                "+X (Robot Fwd)",
                (tip_x[0] + 5, tip_x[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 50, 50),
                1,
                cv2.LINE_AA,
            )

            # +Y axis points Left in robot space - Green
            cv2.arrowedLine(annotated, origin_pt, tip_y, (50, 255, 50), 2, tipLength=0.25)
            cv2.putText(
                annotated,
                "+Y (Robot Left)",
                (tip_y[0] + 5, tip_y[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (50, 255, 50),
                1,
                cv2.LINE_AA,
            )

        # 4. Draw active target crosshair if provided
        if active_target_px is not None:
            t_x, t_y = int(active_target_px[0]), int(active_target_px[1])
            cv2.drawMarker(
                annotated,
                (t_x, t_y),
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

        return annotated

    def save_calibration(self, path: str | Path | None = None) -> str:
        """Persist current homography matrix and reprojection error to an .npz file.

        Args:
            path: Target save path. Defaults to config/workspace_homography.npz (or sim variant).

        Returns:
            Status message string.
        """
        if self.homography_matrix is None:
            return "Cannot save: calibrator has not computed a valid homography."

        default_path = (
            _get_default_sim_save_path() if self.mode == "sim" else _get_default_save_path()
        )
        save_p = Path(path) if path else default_path
        save_p.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            save_p,
            homography=self.homography_matrix,
            inverse_homography=self.inverse_homography,
            reprojection_error=self.reprojection_error or 0.0,
        )
        return f"Calibration successfully saved to {save_p}"

    def load_calibration(self, path: str | Path | None = None) -> bool:
        """Load homography matrix from a saved .npz file.

        Args:
            path: Path to the .npz file. Defaults to config/workspace_homography.npz.

        Returns:
            True if successfully loaded, False otherwise.
        """
        load_p = Path(path) if path else _get_default_save_path()
        if not load_p.exists():
            return False

        try:
            data = np.load(load_p)
            self.homography_matrix = data["homography"]
            self.inverse_homography = data.get(
                "inverse_homography", np.linalg.inv(data["homography"])
            )
            self.reprojection_error = float(data.get("reprojection_error", 0.0))
            return True
        except Exception as e:
            logger.error(f"[ArucoCalibrator] Error loading calibration from {load_p}: {e}")
            return False

    def get_calibration_status(self) -> dict[str, Any]:
        """Return diagnostic status summary dictionary.

        Returns:
            Dictionary containing calibration flags and metrics.
        """
        return {
            "calibrated": self.is_calibrated,
            "markers_detected": len(self.last_detected_markers),
            "reprojection_error": self.reprojection_error,
        }


def generate_aruco_markers(
    output_dir: str | Path = "assets/aruco_markers",
    marker_size_px: int = 400,
    dictionary_name: str = "DICT_4X4_50",
    marker_ids: list[int] | None = None,
) -> list[Path]:
    """Generate printable ArUco marker PNG images for workspace calibration.

    Args:
        output_dir: Output directory path where marker PNGs are saved.
        marker_size_px: Marker image resolution in pixels.
        dictionary_name: ArUco dictionary identifier string.
        marker_ids: List of marker IDs to generate. Defaults to [0, 1, 2, 3].

    Returns:
        List of Path objects for the generated PNG files.
    """
    if marker_ids is None:
        marker_ids = [0, 1, 2, 3]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dict_attr = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)

    created_files: list[Path] = []
    for mid in marker_ids:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size_px)
        # Add white border for printing
        bordered = cv2.copyMakeBorder(marker_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        # Annotate marker ID text
        cv2.putText(
            bordered,
            f"ID {mid} (DICT_4X4_50)",
            (35, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            0,
            1,
            cv2.LINE_AA,
        )

        file_path = out_path / f"marker_{mid}.png"
        cv2.imwrite(str(file_path), bordered)
        created_files.append(file_path)
        logger.info(f"[ArUco] Generated {file_path}")

    return created_files


def generate_calibration_sheet(
    config_data: dict[str, Any] | None = None,
    output_dir: str | Path = "assets/aruco_markers",
    sheet_width_px: int = 2400,
    sheet_height_px: int = 1900,
) -> Path:
    """Generate a single printable high-resolution workspace calibration mat sheet.

    Creates an all-in-one true-to-scale poster/sheet containing all 4 corner markers,
    dimension guides, robot base alignment indicator, and coordinate axes strictly matching
    the physical aspect ratio and base offset from configuration.

    Args:
        config_data: Optional dictionary containing calibration settings.
        output_dir: Directory where the calibration sheet PNG will be saved.
        sheet_width_px: Canvas width in pixels.
        sheet_height_px: Canvas height in pixels.

    Returns:
        Path to the generated calibration sheet PNG file.
    """
    if config_data is None:
        cfg_file = _get_default_config_path()
        if cfg_file.exists():
            try:
                config_data = json.loads(cfg_file.read_text())
            except Exception:
                config_data = {}
        else:
            config_data = {}

    corners = config_data.get("workspace_corners_mm", {})
    tl_cfg = corners.get("top_left", {"x": 380, "y": 280})
    tr_cfg = corners.get("top_right", {"x": 380, "y": -200})
    br_cfg = corners.get("bottom_right", {"x": 120, "y": -200})
    bl_cfg = corners.get("bottom_left", {"x": 120, "y": 280})

    all_x = [float(c["x"]) for c in [tl_cfg, tr_cfg, br_cfg, bl_cfg]]
    all_y = [float(c["y"]) for c in [tl_cfg, tr_cfg, br_cfg, bl_cfg]]
    x_max, x_min = max(all_x), min(all_x)
    y_max, y_min = max(all_y), min(all_y)

    delta_x = x_max - x_min
    delta_y = y_max - y_min
    y_mid = (y_max + y_min) / 2.0

    dict_name = config_data.get("aruco_dictionary", "DICT_4X4_50")
    dict_attr = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)

    marker_ids = config_data.get("marker_ids", {})
    id_tl = int(marker_ids.get("top_left", 1))
    id_tr = int(marker_ids.get("top_right", 2))
    id_br = int(marker_ids.get("bottom_right", 3))
    id_bl = int(marker_ids.get("bottom_left", 0))

    # Create crisp white canvas
    canvas = np.full((sheet_height_px, sheet_width_px, 3), 255, dtype=np.uint8)

    # Compute true-to-scale uniform scaling (pixels per mm)
    mat_top_v = 170
    margin_bottom = 200
    avail_w = sheet_width_px - 380
    avail_h = sheet_height_px - mat_top_v - margin_bottom
    scale = min(avail_w / max(1.0, delta_y), avail_h / max(1.0, (x_max + 10)))

    def pt(x: float, y: float) -> tuple[int, int]:
        """Map physical (X_mm, Y_mm) to canvas pixel (u, v)."""
        u = int(sheet_width_px / 2.0 - (y - y_mid) * scale)
        v = int(mat_top_v + (x_max - x) * scale)
        return (u, v)

    # Physical corner coordinates on canvas
    tl = pt(x_max, y_max)
    tr = pt(x_max, y_min)
    br = pt(x_min, y_min)
    bl = pt(x_min, y_max)
    base = pt(0.0, 0.0)

    # Draw grid background inside mat area every 50mm
    step_mm = 50.0
    # X grid lines (horizontal on canvas)
    cur_x = x_min
    while cur_x <= x_max + 1e-4:
        v_line = pt(cur_x, 0.0)[1]
        cv2.line(canvas, (tl[0], v_line), (tr[0], v_line), (242, 245, 248), 1)
        cur_x += step_mm

    # Y grid lines (vertical on canvas)
    cur_y = y_min
    while cur_y <= y_max + 1e-4:
        u_line = pt(x_min, cur_y)[0]
        cv2.line(canvas, (u_line, tl[1]), (u_line, bl[1]), (242, 245, 248), 1)
        cur_y += step_mm

    # Outer mat border with true physical aspect ratio
    cv2.rectangle(canvas, tl, br, (51, 65, 85), 3)

    # Generate and place the 4 ArUco markers
    marker_px = max(160, min(240, int(scale * 45)))
    m_tl = cv2.aruco.generateImageMarker(aruco_dict, id_tl, marker_px)
    m_tr = cv2.aruco.generateImageMarker(aruco_dict, id_tr, marker_px)
    m_br = cv2.aruco.generateImageMarker(aruco_dict, id_br, marker_px)
    m_bl = cv2.aruco.generateImageMarker(aruco_dict, id_bl, marker_px)

    def _paste_marker(marker_img: np.ndarray, x: int, y: int, label: str, id_num: int) -> None:
        rgb_m = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[y : y + marker_px, x : x + marker_px] = rgb_m
        cv2.rectangle(canvas, (x - 2, y - 2), (x + marker_px + 2, y + marker_px + 2), (0, 0, 0), 2)
        cv2.putText(
            canvas,
            f"ARUCO ID {id_num}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x, y + marker_px + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 80, 95),
            1,
            cv2.LINE_AA,
        )

    # Paste markers strictly inside each corner
    p_pad = 20
    _paste_marker(
        m_tl,
        tl[0] + p_pad,
        tl[1] + p_pad,
        f"Top-Left (X:{tl_cfg['x']}mm, Y:+{tl_cfg['y']}mm)",
        id_tl,
    )
    _paste_marker(
        m_tr,
        tr[0] - p_pad - marker_px,
        tr[1] + p_pad,
        f"Top-Right (X:{tr_cfg['x']}mm, Y:{tr_cfg['y']}mm)",
        id_tr,
    )
    _paste_marker(
        m_br,
        br[0] - p_pad - marker_px,
        br[1] - p_pad - marker_px,
        f"Bottom-Right (X:{br_cfg['x']}mm, Y:{br_cfg['y']}mm)",
        id_br,
    )
    _paste_marker(
        m_bl,
        bl[0] + p_pad,
        bl[1] - p_pad - marker_px,
        f"Bottom-Left (X:{bl_cfg['x']}mm, Y:+{bl_cfg['y']}mm)",
        id_bl,
    )

    # Header title banner
    cv2.putText(
        canvas,
        "SO-101 ROBOTIC ARM - WORKSPACE CALIBRATION SHEET",
        (tl[0], 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (15, 23, 42),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"True-to-Scale Reference | Width (Y): {delta_y:.0f} mm | Depth (X): {delta_x:.0f} mm",
        (tl[0], 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )

    # Draw Center Crosshair
    cx = (tl[0] + tr[0]) // 2
    cy = (tl[1] + bl[1]) // 2
    cv2.drawMarker(canvas, (cx, cy), (160, 175, 190), cv2.MARKER_CROSS, 40, 2)
    cv2.putText(
        canvas,
        f"WORKSPACE CENTER (X:{(x_max + x_min) / 2:.0f}, Y:{y_mid:.0f})",
        (cx - 120, cy - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )

    # Draw Coordinate Axes indicator in center
    axis_ox, axis_oy = cx - 180, cy + 80
    cv2.arrowedLine(
        canvas,
        (axis_ox, axis_oy),
        (axis_ox, axis_oy - 100),
        (220, 38, 38),
        3,
        tipLength=0.25,
    )
    cv2.putText(
        canvas,
        "+X (Forward)",
        (axis_ox + 10, axis_oy - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 38, 38),
        2,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        canvas,
        (axis_ox, axis_oy),
        (axis_ox - 100, axis_oy),
        (22, 163, 74),
        3,
        tipLength=0.25,
    )
    cv2.putText(
        canvas,
        "+Y (Left)",
        (axis_ox - 110, axis_oy - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (22, 163, 74),
        2,
        cv2.LINE_AA,
    )

    # Robot Base Alignment Guide precisely positioned at true physical (X=0, Y=0)
    base_box_w, base_box_h = 340, 80
    base_bx1 = base[0] - (base_box_w // 2)
    base_by1 = base[1] - 30

    # Draw dashed/dotted connection line from base to near edge of mat
    cv2.line(canvas, (base[0], bl[1]), (base[0], base_by1), (220, 38, 38), 2)

    cv2.rectangle(
        canvas,
        (base_bx1, base_by1),
        (base_bx1 + base_box_w, base_by1 + base_box_h),
        (30, 41, 59),
        2,
    )
    cv2.drawMarker(
        canvas,
        (base[0], base_by1 + (base_box_h // 2)),
        (220, 38, 38),
        cv2.MARKER_TILTED_CROSS,
        25,
        2,
    )
    cv2.putText(
        canvas,
        "ROBOT BASE ALIGNMENT (X=0, Y=0)",
        (base_bx1 + 15, base_by1 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Align arm base center here ({x_min:.0f}mm from mat edge)",
        (base_bx1 + 10, base_by1 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )

    # Save output sheet
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    sheet_file = out_p / "workspace_calibration_sheet.png"
    cv2.imwrite(str(sheet_file), canvas)
    logger.info(f"[ArUco] Generated true-to-scale calibration sheet: {sheet_file}")
    return sheet_file


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

    # Build updated config
    updated_config = {
        "aruco_dictionary": cfg.get("aruco_dictionary", "DICT_4X4_50"),
        "marker_ids": cfg.get(
            "marker_ids",
            {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 0},
        ),
        "marker_size_mm": marker_size,
        "workspace_corners_mm": {
            "top_left": {"x": x_max, "y": y_max},
            "top_right": {"x": x_max, "y": y_min},
            "bottom_right": {"x": x_min, "y": y_min},
            "bottom_left": {"x": x_min, "y": y_max},
        },
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
        created_markers = generate_aruco_markers(output_dir=output_dir)

        # Generate all-in-one calibration mat sheet
        created_sheet = generate_calibration_sheet(
            config_data=updated_config, output_dir=output_dir
        )

        print("\n" + "=" * 65)
        print("🎉 WORKSPACE SETUP & MARKER GENERATION COMPLETE!")
        print("=" * 65)
        print("Generated Artifacts:")
        for mf in created_markers:
            print(f"  • Individual Marker : {mf}")
        print(f"  • Full Printable Mat: {created_sheet}")
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
