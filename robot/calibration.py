"""ArUco workspace calibration module for SO-101 robot arm.

Detects 4-corner fiducial markers (DICT_4X4_50) defining the physical workspace mat,
computes perspective homography, and transforms Gemini normalized camera coordinates
directly to physical millimeter/meter coordinates in the robot base frame.
"""

import json
import logging
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot.camera_model import PinholeCamera
from robot.errors import CalibrationUnavailable
from robot.logger import log_with_tag
from robot.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeometrySnapshot:
    """Immutable bundle of everything derived from one workspace homography.

    The shared calibrator's homography is replaced on every frame that shows all four markers
    (real mode), so consumers take one snapshot per tool call and read H, its inverse and the
    camera model from it: they can never mix two homographies.

    Attributes:
        homography: 3x3 pixel -> robot-mm matrix (read-only).
        inverse_homography: 3x3 robot-mm -> pixel matrix (read-only).
        reprojection_error: Mean marker reprojection error in mm, if known.
        image_size: (width, height) of the homography's pixel frame.
        hfov_deg: Horizontal field of view the camera model was derived with.
        camera: Pinhole camera recovered from the homography, or None if the derivation failed.
    """

    homography: np.ndarray
    inverse_homography: np.ndarray
    reprojection_error: float | None
    image_size: tuple[int, int]
    hfov_deg: float
    camera: PinholeCamera | None

    def pixel_to_table_mm(self, u: float, v: float) -> tuple[float, float]:
        """Map a pixel to the table plane (z = 0) through the homography.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.

        Returns:
            (x_mm, y_mm) robot-frame table position.
        """
        vec = self.homography @ np.array([u, v, 1.0], dtype=np.float64)
        return float(vec[0] / vec[2]), float(vec[1] / vec[2])

    def pixel_to_plane_mm(self, u: float, v: float, height_mm: float) -> tuple[float, float]:
        """Map the pixel of a point known to be ``height_mm`` above the table to its true XY.

        The table hit of the pixel comes from the homography H (so a point on the table agrees
        exactly with every other table measurement); the parallax of a raised point is removed
        with the camera centre C:
        ``xy = C_xy + (H(u, v) - C_xy) * (C_z - h) / C_z``.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            height_mm: Height of the observed point above the table.

        Returns:
            (x_mm, y_mm) of the point's table-plane footprint.

        Raises:
            CalibrationUnavailable: If a raised point is observed but there is no camera model.
        """
        base = np.array(self.pixel_to_table_mm(u, v))
        if abs(height_mm) < 1e-9:
            return float(base[0]), float(base[1])
        if self.camera is None:
            raise CalibrationUnavailable("Camera model unavailable: cannot remove parallax.")
        centre = self.camera.centre_mm
        scale = (centre[2] - height_mm) / centre[2]
        xy = centre[:2] + (base - centre[:2]) * scale
        return float(xy[0]), float(xy[1])

    def table_mm_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Map a table-plane point to its pixel through the inverse homography."""
        vec = self.inverse_homography @ np.array([x_mm, y_mm, 1.0], dtype=np.float64)
        return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def _readonly(matrix: np.ndarray) -> np.ndarray:
    """Return a read-only float64 copy of a matrix."""
    out = np.array(matrix, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


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
        self.config_path = (
            Path(config_path) if config_path else get_settings().calibration_config_path
        )
        self.mode = mode
        self.config: dict[str, Any] = {}
        # Everything derived from one homography lives in one immutable snapshot that is replaced
        # in a single assignment (see ``set_geometry``); never mutate pieces of it.
        self._geometry: GeometrySnapshot | None = None
        self._camera_logged = False
        self._camera_log_lock = threading.Lock()
        self.last_detected_markers: dict[int, np.ndarray] = {}

        self._load_config()
        self._init_aruco()

        # Determine target calibration file path based on mode
        target_save_path = (
            Path(save_path)
            if save_path
            else (
                get_settings().calibration_sim_save_path
                if mode == "sim"
                else get_settings().calibration_save_path
            )
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
            except (json.JSONDecodeError, OSError) as e:
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

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    @property
    def is_calibrated(self) -> bool:
        """Return True if a valid perspective homography matrix is available."""
        return self._geometry is not None

    @property
    def homography_matrix(self) -> np.ndarray | None:
        """Pixel -> robot-mm homography of the current geometry, or None."""
        geometry = self._geometry
        return geometry.homography if geometry is not None else None

    @homography_matrix.setter
    def homography_matrix(self, value: np.ndarray | None) -> None:
        if value is None:
            self._geometry = None
            return
        geometry = self._geometry
        try:
            inverse = np.linalg.inv(value)
        except np.linalg.LinAlgError:
            inverse = geometry.inverse_homography if geometry is not None else value
        self.set_geometry(
            value, inverse, geometry.reprojection_error if geometry is not None else None
        )

    @property
    def inverse_homography(self) -> np.ndarray | None:
        """Robot-mm -> pixel homography of the current geometry, or None."""
        geometry = self._geometry
        return geometry.inverse_homography if geometry is not None else None

    @inverse_homography.setter
    def inverse_homography(self, value: np.ndarray | None) -> None:
        geometry = self._geometry
        if value is None:
            self._geometry = None
            return
        if geometry is None:
            self.set_geometry(np.linalg.inv(value), value, None)
        else:
            self.set_geometry(geometry.homography, value, geometry.reprojection_error)

    @property
    def reprojection_error(self) -> float | None:
        """Mean marker reprojection error (mm) of the current geometry, or None."""
        geometry = self._geometry
        return geometry.reprojection_error if geometry is not None else None

    @reprojection_error.setter
    def reprojection_error(self, value: float | None) -> None:
        geometry = self._geometry
        if geometry is not None:
            self.set_geometry(geometry.homography, geometry.inverse_homography, value)

    def geometry_snapshot(self) -> GeometrySnapshot | None:
        """Return the current immutable geometry (H, H^-1, error, camera), or None.

        Consumers take one snapshot per tool call so a homography update that lands mid-call
        (real mode refreshes it on every frame showing four markers) cannot mix two geometries.
        The first snapshot after start-up also logs the derived camera centre.

        Returns:
            The current GeometrySnapshot, or None if the workspace is uncalibrated.
        """
        geometry = self._geometry
        if geometry is not None:
            self._log_camera_once(geometry)
        return geometry

    def set_geometry(
        self,
        homography: np.ndarray,
        inverse_homography: np.ndarray,
        reprojection_error: float | None,
        hfov_deg: float | None = None,
        image_size: tuple[int, int] | None = None,
        log_camera: bool = False,
    ) -> GeometrySnapshot:
        """Commit a new geometry in one assignment, deriving the camera model from it.

        Args:
            homography: 3x3 pixel -> robot-mm matrix.
            inverse_homography: 3x3 robot-mm -> pixel matrix.
            reprojection_error: Mean marker reprojection error in mm, if known.
            hfov_deg: Horizontal field of view for the camera model. Defaults to the previous
                geometry's value, then to ``settings.camera_hfov_deg``.
            image_size: (width, height) of the homography's pixel frame. Defaults to the
                previous geometry's value, then to the configured capture size.
            log_camera: Log the derived camera centre at INFO.

        Returns:
            The committed snapshot.
        """
        previous = self._geometry
        settings = get_settings()
        hfov = float(
            hfov_deg
            if hfov_deg is not None
            else (previous.hfov_deg if previous is not None else settings.camera_hfov_deg)
        )
        size = (
            (int(image_size[0]), int(image_size[1]))
            if image_size is not None
            else (
                previous.image_size
                if previous is not None
                else (int(settings.stream_width), int(settings.stream_height))
            )
        )
        camera: PinholeCamera | None
        try:
            camera = PinholeCamera.from_homography(inverse_homography, size, hfov)
        except (ValueError, cv2.error, np.linalg.LinAlgError, ZeroDivisionError) as e:
            logger.debug(f"[ArucoCalibrator] Camera model not derived: {e}")
            camera = None
        snapshot = GeometrySnapshot(
            homography=_readonly(homography),
            inverse_homography=_readonly(inverse_homography),
            reprojection_error=None if reprojection_error is None else float(reprojection_error),
            image_size=size,
            hfov_deg=hfov,
            camera=camera,
        )
        self._geometry = snapshot  # the single commit
        if log_camera:
            self._log_camera(snapshot)
        return snapshot

    def _log_camera_once(self, snapshot: GeometrySnapshot) -> None:
        """Log the derived camera once per process (first snapshot after start-up)."""
        with self._camera_log_lock:
            if self._camera_logged:
                return
            self._log_camera(snapshot)

    def _log_camera(self, snapshot: GeometrySnapshot) -> None:
        """Log the derived camera centre (robot mm) so it can be checked with a tape measure."""
        self._camera_logged = True
        if snapshot.camera is None:
            log_with_tag(
                logger,
                "CAMERA",
                "camera model unavailable for this homography",
                module="Calibration",
            )
            return
        centre = snapshot.camera.centre_mm
        log_with_tag(
            logger,
            "CAMERA",
            f"centre_mm=[{centre[0]:.0f}, {centre[1]:.0f}, {centre[2]:.0f}] "
            f"hfov={snapshot.hfov_deg:.1f}",
            module="Calibration",
        )

    def require_calibrated(self) -> None:
        """Verify that a valid homography transformation matrix is set.

        Raises:
            CalibrationUnavailable: If workspace is not calibrated.
        """
        if not self.is_calibrated:
            raise CalibrationUnavailable(
                "Workspace is not calibrated. Please ensure 4 ArUco markers are in view and "
                "save calibration under the ArUco Workspace Calibration panel before executing "
                "spatial commands."
            )

    def detect_markers(self, frame_rgb: np.ndarray) -> dict[int, np.ndarray]:
        """Detect ArUco fiducial markers in an RGB frame.

        Args:
            frame_rgb: Input RGB image array of shape (H, W, 3).

        Returns:
            Dictionary mapping integer marker ID to (4, 2) array of corner pixel coordinates.
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

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
                inv_h = np.linalg.inv(h_mat)

                # Compute reprojection error
                src_homo = np.hstack([src_arr, np.ones((len(src_arr), 1), dtype=np.float32)])
                proj = (h_mat @ src_homo.T).T
                proj_xy = proj[:, :2] / proj[:, 2:3]
                err = float(np.mean(np.linalg.norm(proj_xy - dst_arr, axis=1)))

                # Commit atomically (one snapshot) only after all calculations succeed
                self.set_geometry(h_mat, inv_h, err)
                return True
        except (cv2.error, np.linalg.LinAlgError, ValueError, TypeError) as e:
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
            get_settings().calibration_sim_save_path
            if self.mode == "sim"
            else get_settings().calibration_save_path
        )
        save_p = Path(path) if path else default_path
        save_p.parent.mkdir(parents=True, exist_ok=True)

        geometry = self._geometry
        assert geometry is not None
        np.savez(
            save_p,
            homography=geometry.homography,
            inverse_homography=geometry.inverse_homography,
            reprojection_error=geometry.reprojection_error or 0.0,
            hfov_deg=geometry.hfov_deg,
            image_size=np.array(geometry.image_size),
        )
        self._log_camera(geometry)
        return f"Calibration successfully saved to {save_p}"

    def load_calibration(self, path: str | Path | None = None) -> bool:
        """Load homography matrix from a saved .npz file.

        Args:
            path: Path to the .npz file. Defaults to config/workspace_homography.npz.

        Returns:
            True if successfully loaded, False otherwise.
        """
        load_p = Path(path) if path else get_settings().calibration_save_path
        if not load_p.exists():
            return False

        try:
            data = np.load(load_p)
            h_mat = data["homography"]
            if "inverse_homography" in data:
                inv_h = data["inverse_homography"]
            else:
                inv_h = np.linalg.inv(h_mat)
            reproj_err = float(data.get("reprojection_error", 0.0))
            # Optional keys (absent in files written before the camera model): the field of view
            # and pixel frame the homography was recorded with. Old files load unchanged.
            settings = get_settings()
            hfov_deg = (
                float(data["hfov_deg"]) if "hfov_deg" in data else float(settings.camera_hfov_deg)
            )
            image_size = (
                (int(data["image_size"][0]), int(data["image_size"][1]))
                if "image_size" in data
                else (int(settings.stream_width), int(settings.stream_height))
            )

            # Commit atomically only after all computations succeed
            self.set_geometry(
                h_mat,
                inv_h,
                reproj_err,
                hfov_deg=hfov_deg,
                image_size=image_size,
                log_camera=True,
            )
            return True
        except (OSError, KeyError, ValueError, TypeError, np.linalg.LinAlgError) as e:
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
