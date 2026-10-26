from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from photosim._checks import as_flat_points, check_rotation


@dataclass(frozen=True, eq=False)
class Camera:
    """Внутренние параметры камеры. После создания не меняются.

    dist - коэффициенты дисторсии в порядке OpenCV: (k1, k2, p1, p2[, k3]).
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray | None = None
    r_max: float = field(init=False)
    has_distortion: bool = field(init=False)
    _ray_cache: OrderedDict = field(init=False, repr=False, default_factory=OrderedDict)

    RAY_CACHE_SIZE = 8      # сколько разных (roi, ss) держать в памяти
    RAY_CACHE_BYTES = 256 * 2**20   # и не больше байт: сетка при ss = 64 — сотни МБ

    def __post_init__(self):
        set_ = object.__setattr__      # обход frozen, только здесь
        for name in ("fx", "fy", "cx", "cy"):
            set_(self, name, float(getattr(self, name)))
        set_(self, "width", int(self.width))
        set_(self, "height", int(self.height))

        if not (self.fx > 0 and self.fy > 0):
            raise ValueError(f"fx и fy должны быть > 0, а пришли {self.fx}, {self.fy}")
        if not (self.width > 0 and self.height > 0):
            raise ValueError(f"width и height должны быть > 0, "
                             f"а пришли {self.width}, {self.height}")

        dist = self._check_dist(self.dist)
        set_(self, "dist", dist)
        set_(self, "has_distortion", dist is not None and bool(np.any(dist != 0)))
        set_(self, "r_max", self._compute_r_max(dist) if self.has_distortion else np.inf)

    @staticmethod
    def _check_dist(dist):
        """Порядок как в OpenCV: (k1, k2, p1, p2, k3). k3 можно не указывать."""
        if dist is None:
            return None
        d = np.array(dist, dtype=float).ravel()     # копия, чужой массив не трогаем
        if d.size == 4:
            d = np.append(d, 0.0)
        if d.size != 5:
            raise ValueError(f"dist: нужно 4 или 5 чисел (k1, k2, p1, p2[, k3]), "
                             f"а пришло {d.size}")
        d.flags.writeable = False                   # cam.dist[0] = ... тоже запрещено
        return d

    @staticmethod
    def _compute_r_max(dist):
        """Радиус (в нормализованных координатах), до которого дисторсия обратима.
        """
        k1, k2, _, _, k3 = dist
        roots = np.roots([7 * k3, 5 * k2, 3 * k1, 1])   # корни по s = r^2
        s = roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)].real
        return float(np.sqrt(s.min())) if s.size else np.inf

    @classmethod
    def from_physical(cls, focal_mm, pixel_um, width, height):
        """Камера по фокусу (мм), размеру пикселя (мкм) и разрешению."""
        f = focal_mm * 1000 / pixel_um
        return cls(f, f, (width - 1) / 2, (height - 1) / 2, width, height)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1.0]])

    def _distort(self, xy):
        """Брауна-Конради в нормализованных координатах: (M, 2) -> (M, 2)."""
        if not self.has_distortion:
            return xy
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[:, 0], xy[:, 1]

        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))   # 1 + k1 r^2 + k2 r^4 + k3 r^6
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.column_stack([xd, yd])

    def _distort_jacobian(self, xy):
        """Производная _distort по (x, y): (M, 2) -> (M, 2, 2)."""
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[:, 0], xy[:, 1]
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        d_radial = k1 + r2 * (2 * k2 + r2 * 3 * k3)    # d(radial)/d(r^2)

        J = np.empty((len(xy), 2, 2))
        J[:, 0, 0] = radial + 2 * x * x * d_radial + 2 * p1 * y + 6 * p2 * x
        J[:, 0, 1] = 2 * x * y * d_radial + 2 * p1 * x + 2 * p2 * y
        J[:, 1, 0] = J[:, 0, 1]
        J[:, 1, 1] = radial + 2 * y * y * d_radial + 6 * p1 * y + 2 * p2 * x
        return J

    def _undistort(self, xy_d, tol=1e-14, max_iter=20):
        """Обратная дисторсия методом Ньютона: (M, 2) -> (M, 2) и маска (M,).
        """
        if not self.has_distortion:
            return xy_d, np.all(np.isfinite(xy_d), axis=1)

        xy = xy_d.copy()                           # стартуем с искаженной точки
        active = np.all(np.isfinite(xy_d), axis=1)
        done = np.zeros(len(xy), dtype=bool)

        for _ in range(max_iter):
            idx = np.flatnonzero(active & ~done)
            if idx.size == 0:
                break
            p = xy[idx]
            res = self._distort(p) - xy_d[idx]
            ok = np.linalg.norm(res, axis=1) < tol
            done[idx[ok]] = True

            idx, p, res = idx[~ok], p[~ok], res[~ok]
            J = self._distort_jacobian(p)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            bad = ~(det > 0)                       # складка или nan: дальше не идем
            active[idx[bad]] = False
            idx, p, res, J, det = idx[~bad], p[~bad], res[~bad], J[~bad], det[~bad]

            # шаг Ньютона: p -= J^-1 res, для 2x2 обращаем вручную
            dx = ( J[:, 1, 1] * res[:, 0] - J[:, 0, 1] * res[:, 1]) / det
            dy = (-J[:, 1, 0] * res[:, 0] + J[:, 0, 0] * res[:, 1]) / det
            xy[idx] = p - np.column_stack([dx, dy])

        # последняя проверка для тех, кто сделал шаг на последней итерации
        idx = np.flatnonzero(active & ~done)
        if idx.size:
            res = self._distort(xy[idx]) - xy_d[idx]
            done[idx[np.linalg.norm(res, axis=1) < tol]] = True

        valid = done & (np.sum(xy * xy, axis=1) < self.r_max ** 2)
        xy[~valid] = np.nan
        return xy, valid

    def project(self, X_cam, return_mask=False):
        """Точки камеры (..., 3) -> пиксели (..., 2).

        nan получают точки с Z <= 0 и точки дальше r_max (там дисторсия
        не взаимно однозначна).
        """
        X, lead = as_flat_points(X_cam, 3, "X_cam")
        Z = X[:, 2]
        valid = Z > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            xy = X[:, :2] / Z[:, None]
        valid &= np.sum(xy * xy, axis=1) < self.r_max ** 2
        xy[~valid] = np.nan

        xy = self._distort(xy)
        uv = xy * [self.fx, self.fy] + [self.cx, self.cy]

        uv = uv.reshape(lead + (2,))
        valid = valid.reshape(lead)
        return (uv, valid) if return_mask else uv

    def backproject(self, uv, return_mask=False):
        """Пиксели (..., 2) -> единичные лучи (..., 3).

        nan получают пиксели, для которых обратная дисторсия не нашлась.
        """
        p, lead = as_flat_points(uv, 2, "uv")
        xy = (p - [self.cx, self.cy]) / [self.fx, self.fy]
        xy, valid = self._undistort(xy)

        rays = np.column_stack([xy, np.ones(len(xy))])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

        rays = rays.reshape(lead + (3,))
        valid = valid.reshape(lead)
        return (rays, valid) if return_mask else rays

    def jacobian(self, X_cam):
        """Производная d(u, v)/d(X, Y, Z): (..., 3) -> (..., 2, 3).
        """
        X, lead = as_flat_points(X_cam, 3, "X_cam")
        x, y, z = X.T
        with np.errstate(divide="ignore", invalid="ignore"):
            iz = 1 / z
            xn, yn = x * iz, y * iz
            N = np.zeros((len(X), 2, 3))
            N[:, 0, 0] = iz
            N[:, 0, 2] = -xn * iz
            N[:, 1, 1] = iz
            N[:, 1, 2] = -yn * iz

        if self.has_distortion:
            D = self._distort_jacobian(np.column_stack([xn, yn]))
            N = D @ N                              # (M, 2, 2) @ (M, 2, 3)

        J = N * np.array([self.fx, self.fy])[:, None]
        bad = (z <= 0) | ~(xn * xn + yn * yn < self.r_max ** 2)
        J[bad] = np.nan                            # та же политика, что в project
        return J.reshape(lead + (2, 3))


    def _check_roi(self, roi, ss):
        """Проверить roi = (u0, v0, w, h) и ss. Дробные значения не допускаются."""
        try:
            vals = [float(a) for a in roi]
        except (TypeError, ValueError):
            raise ValueError(f"roi должна быть (u0, v0, w, h), а пришло {roi!r}") from None
        if len(vals) != 4:
            raise ValueError(f"roi должна быть (u0, v0, w, h), а пришло {roi!r}")
        if not all(np.isfinite(a) and a == int(a) for a in vals):
            raise ValueError(
                f"roi должна состоять из целых чисел, а пришло {tuple(roi)}; "
                "округлите границы наружу (floor/ceil) перед вызовом"
            )
        u0, v0, w, h = (int(a) for a in vals)

        if not (np.isfinite(ss) and int(ss) == ss and ss >= 1):
            raise ValueError(f"ss должно быть целым >= 1, а пришло {ss!r}")
        if w < 1 or h < 1:
            raise ValueError(f"размер roi должен быть >= 1, а пришло {w}x{h}")
        if u0 < 0 or v0 < 0 or u0 + w > self.width or v0 + h > self.height:
            raise ValueError(
                f"roi ({u0}, {v0}, {w}, {h}) выходит за кадр "
                f"{self.width}x{self.height}"
            )
        return (u0, v0, w, h), int(ss)

    def subpixel_centers(self, roi, ss):
        """Центры субпикселей в пикселях: (h, w, ss, ss, 2).
        """
        (u0, v0, w, h), ss = self._check_roi(roi, ss)
        off = (np.arange(ss) + 0.5) / ss - 0.5
        U = (u0 + np.arange(w))[None, :, None, None] + off[None, None, None, :]
        V = (v0 + np.arange(h))[:, None, None, None] + off[None, None, :, None]
        return np.stack(np.broadcast_arrays(U, V), axis=-1)

    def ray_grid(self, roi, ss, dtype=np.float64):
        """Единичные лучи для субпикселей ROI: (h, w, ss, ss, 3) и маска (h, w, ss, ss).

        Яркость пикселя: values.mean(axis=(2, 3)).
        Лучи за r_max - nan и False в маске.
        Результат кешируется и доступен только для чтения.
        ROI за пределами кадра запрещена.
        """
        roi, ss = self._check_roi(roi, ss)
        key = (roi, ss, np.dtype(dtype).str)
        cache = self._ray_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        rays, valid = self.backproject(self.subpixel_centers(roi, ss), return_mask=True)
        rays = rays.astype(dtype, copy=False)
        rays.flags.writeable = False
        valid.flags.writeable = False

        result = cache[key] = (rays, valid)
        # выкинуть самые давние; текущая сетка остаётся, даже если она больше бюджета
        while len(cache) > 1 and (
            len(cache) > self.RAY_CACHE_SIZE
            or sum(r.nbytes + m.nbytes for r, m in cache.values())
            > self.RAY_CACHE_BYTES
        ):
            cache.popitem(last=False)
        return result


class Pose:
    """Поза камеры: X_cam = R @ X_world + T."""

    def __init__(self, R, T, atol=1e-6):
        T = np.asarray(T, dtype=float)
        if T.size != 3:
            raise ValueError(f"T должен содержать 3 числа, а пришло {T.shape}")
        R = check_rotation(R, "R", atol=atol)   # копия, только для чтения

        self.R = R
        self.T = T.reshape(3)

    @classmethod
    def from_euler(cls, angles_deg, T, seq="xyz"):
        """Поза из углов Эйлера (в градусах).
        """
        R = Rotation.from_euler(seq, angles_deg, degrees=True).as_matrix()
        return cls(R, T)

    @classmethod
    def from_rvec(cls, rvec, T):
        """Поза из rvec и tvec в формате OpenCV."""
        return cls(Rotation.from_rotvec(np.ravel(rvec)).as_matrix(), T)

    @classmethod
    def look_at(cls, center, target, up=(0, 0, 1)):
        """Камера в точке center смотрит на target, up - верх в мире."""
        C = np.asarray(center, dtype=float)
        z = np.asarray(target, dtype=float) - C
        if np.linalg.norm(z) == 0:
            raise ValueError("center и target совпадают")
        z /= np.linalg.norm(z)

        up = np.asarray(up, dtype=float)
        if np.linalg.norm(up) == 0:
            raise ValueError("up не должен быть нулевым")
        up = up / np.linalg.norm(up)

        x = np.cross(z, up)                  # |x| = sin угла между z и up
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("камера смотрит параллельно up, выберите другой up")
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        R = np.vstack([x, y, z])
        return cls(R, -R @ C)

    @property
    def center(self):
        """Центр камеры в мире."""
        return -self.R.T @ self.T

    def to_camera(self, X_world):
        return np.asarray(X_world, dtype=float) @ self.R.T + self.T

    def to_world(self, X_cam):
        return (np.asarray(X_cam, dtype=float) - self.T) @ self.R
