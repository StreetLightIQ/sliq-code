"""Camera geometry. World frame (metres): origin on the ground below the camera, X right, Y down, Z forward; the ground is the plane
Y = camera height. Camera rays map to world rays with R = Rx(-pitch) @ Rz(roll)."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    hfov_deg: float = 70.0
    pitch_deg: float = 5.0           # positive = looking down
    roll_deg: float = 0.0
    height_m: float = 1.3

    @property
    def f(self):
        return (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)

    @property
    def R(self):
        p, r = np.radians(self.pitch_deg), np.radians(self.roll_deg)
        rx = np.array([[1, 0, 0], [0, np.cos(-p), -np.sin(-p)], [0, np.sin(-p), np.cos(-p)]])
        rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        return rx @ rz

    def rays(self, u, v):
        """World unit rays through pixels u, v -> (..., 3)."""
        c = np.stack([(np.asarray(u, float) - self.width / 2) / self.f, (np.asarray(v, float) - self.height / 2) / self.f,
                      np.ones(np.shape(u))], -1)
        w = c @ self.R.T
        return w / np.linalg.norm(w, axis=-1, keepdims=True)

    def ground_points(self, u, v, max_range=80.0):
        """Ground point of each pixel ray, and a mask (False above the horizon or beyond max_range)."""
        r = self.rays(u, v)
        down = r[..., 1]
        ok = down > 1e-6
        t = np.where(ok, self.height_m / np.where(ok, down, 1.0), np.inf)
        p = r * t[..., None]
        ok &= np.hypot(p[..., 0], p[..., 2]) <= max_range
        return p, ok

    def project(self, points):
        """World points (..., 3) -> pixel u, v and a mask of points in front of the camera."""
        c = np.asarray(points, float) @ self.R
        z = c[..., 2]
        front = z > 1e-6
        zs = np.where(front, z, 1.0)
        return c[..., 0] / zs * self.f + self.width / 2, c[..., 1] / zs * self.f + self.height / 2, front


def camera_from_gravity(width, height, gravity_xyz, hfov_deg=70.0, height_m=1.3, axes='landscape_left'):
    """Pitch and roll from the gravity vector (device axes, pointing away from the ground at rest), phone in landscape left with the
    rear camera along -z."""
    gx, gy, gz = (float(v) for v in gravity_xyz)
    g = np.array([gx, gy, gz]) / max(1e-9, np.linalg.norm([gx, gy, gz]))
    if axes != 'landscape_left':
        raise ValueError(axes)
    up_img, right_img, forward = g[0], -g[1], -g[2]
    pitch = np.degrees(np.arcsin(np.clip(-forward, -1, 1)))
    roll = np.degrees(np.arctan2(right_img, up_img))
    return Camera(width, height, hfov_deg, pitch, roll, height_m)
