"""Pinhole camera model recovered from the workspace homography plus nominal intrinsics.

The workspace homography maps image pixels to table-plane robot millimetres, but anything above
the table (the gripper, a lifted object) is displaced in the image by parallax. Knowing where the
camera centre is lets the jaw tag, whose height comes from forward kinematics, be projected back
to its true table-plane XY.

The intrinsics are *nominal*: principal point at the image centre, square pixels, and a focal
length from the horizontal field of view. Estimating the focal length from a single-plane
homography is exact on a noise-free matrix but ill-conditioned (at 0.5 px noise the camera centre
is off by up to a metre), so it is deliberately not done. However, the nominal field of view must
match the true sensor width: for this low, oblique camera (~345 mm above the table), assuming 70°
instead of the true 62.2° (Arducam IMX219 3.04 mm lens) shifts the recovered camera height from
356 mm to 297 mm and introduces 35–120 mm of jaw-locator parallax error at the tag's hover height
(~110 mm above the table).
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """A pinhole camera in the robot frame (millimetres, +Z up).

    A robot-frame point X maps to the camera frame as ``R @ X + t`` (OpenCV convention: x right,
    y down, z forward) and to the image as ``K @ (R @ X + t)``.

    Attributes:
        K: 3x3 intrinsics matrix (pixels).
        R: 3x3 world-to-camera rotation.
        t: (3,) world-to-camera translation (mm).
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def centre_mm(self) -> np.ndarray:
        """Camera centre in the robot frame (mm)."""
        return -self.R.T @ self.t

    def project(
        self, xyz_mm: np.ndarray | list[float] | tuple[float, float, float]
    ) -> tuple[float, float]:
        """Project a robot-frame point to image pixels.

        Args:
            xyz_mm: Point (x, y, z) in robot mm.

        Returns:
            (u, v) pixel coordinates.

        Raises:
            ValueError: If the point is behind the camera.
        """
        cam = self.R @ np.asarray(xyz_mm, dtype=np.float64) + self.t
        if cam[2] <= 1e-9:
            raise ValueError("Point is behind the camera")
        uvw = self.K @ cam
        return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])

    def ray_to_plane(self, u: float, v: float, z_mm: float) -> tuple[float, float]:
        """Intersect the viewing ray of a pixel with the horizontal plane Z = ``z_mm``.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            z_mm: Height of the plane in robot mm.

        Returns:
            (x_mm, y_mm) where the ray meets that plane.

        Raises:
            ValueError: If the ray does not head down to the plane.
        """
        direction = self.R.T @ np.linalg.solve(self.K, np.array([u, v, 1.0]))
        centre = self.centre_mm
        if direction[2] >= -1e-12:
            raise ValueError("Viewing ray does not reach the plane")
        scale = (z_mm - centre[2]) / direction[2]
        hit = centre + scale * direction
        return float(hit[0]), float(hit[1])

    @classmethod
    def from_homography(
        cls,
        robot_mm_to_pixel: np.ndarray,
        image_size: tuple[int, int],
        hfov_deg: float,
    ) -> "PinholeCamera":
        """Recover the camera pose from the table homography and nominal intrinsics.

        The pose comes from ``cv2.solvePnP`` (IPPE, planar target) on a grid of table points
        mapped through the homography; of the two IPPE solutions the one with the camera above
        the table and the smaller reprojection error is kept.

        Args:
            robot_mm_to_pixel: 3x3 homography G mapping robot (x, y) mm on the table to pixels.
            image_size: (width, height) of the homography's own pixel frame.
            hfov_deg: Nominal horizontal field of view in degrees.

        Returns:
            The recovered camera.

        Raises:
            ValueError: If the homography is singular or no pose puts the camera above the table.
        """
        width, height = image_size
        if not 5.0 <= hfov_deg <= 175.0:
            raise ValueError(f"hfov_deg {hfov_deg} outside (5, 175)")
        focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

        pixel_to_robot = np.linalg.inv(robot_mm_to_pixel)
        us = np.linspace(0.1 * width, 0.9 * width, 6)
        vs = np.linspace(0.1 * height, 0.9 * height, 5)
        pixels = np.array([[u, v] for v in vs for u in us], dtype=np.float64)
        homogeneous = np.hstack([pixels, np.ones((len(pixels), 1))])
        table = (pixel_to_robot @ homogeneous.T).T
        table_xy = table[:, :2] / table[:, 2:3]
        object_points = np.hstack([table_xy, np.zeros((len(table_xy), 1))]).astype(np.float64)

        _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points,
            pixels.reshape(-1, 1, 2),
            K,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        best: PinholeCamera | None = None
        best_error = float("inf")
        for rvec, tvec, err in zip(rvecs, tvecs, np.asarray(errors).ravel(), strict=False):
            rotation, _ = cv2.Rodrigues(rvec)
            candidate = cls(K=K, R=rotation, t=np.asarray(tvec, dtype=np.float64).ravel())
            if candidate.centre_mm[2] > 0.0 and float(err) < best_error:
                best, best_error = candidate, float(err)
        if best is None:
            raise ValueError("No camera pose places the camera above the table")
        return best
