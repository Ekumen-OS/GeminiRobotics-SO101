"""ArUco workspace calibration module for SO-101 robot arm.

Detects 4-corner fiducial markers (DICT_4X4_50) defining the physical workspace mat,
computes perspective homography, and transforms Gemini normalized camera coordinates
directly to physical millimeter/meter coordinates in the robot base frame.
"""

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
            mode: Operating mode ('real' or 'sim').
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
