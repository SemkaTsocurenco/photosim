"""Эксперименты: описание конфигурации -> таблица метрик всех методов.
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as _SciRotation

import photosim
from photosim import presets
from photosim.camera import Camera, Pose
from photosim.estimators import ESTIMATORS, CenterEstimate, estimate
from photosim.oracle import conic_center, exposure_conic_center
from photosim.render import render
from photosim.scene import (
    Accel,
    Arc,
    AxisRotation,
    FixedOrientation,
    Linear,
    Motion,
    Static,
    frame_from_normal,
    reference_time,
)
from photosim.sensor import SensorModel
from photosim.target import Target

__all__ = [
    "EFFECTS",
    "REFERENCES",
    "RUNS_DIR",
    "TILT_REFERENCES",
    "T_K",
    "Config",
    "Effects",
    "Run",
    "Setup",
    "ablation",
    "additivity",
    "build",
    "load_run",
    "references",
    "run_point",
    "save_run",
    "sweep",
]

#: Начало экспозиции кадра, с.
T_K = 0.0

#: Эталоны центра, см. ``references``.
REFERENCES = ("proj_start", "proj_mid", "conic_mid", "exposure_conic")

#: От чего отсчитывается наклон марки, см. ``Config``.
TILT_REFERENCES = ("optical_axis", "line_of_sight")

#: Папка запусков по умолчанию.
RUNS_DIR = Path("runs")

_MOTION_KEYS = {
    "static": set(),
    "linear": {"velocity"},
    "accel": {"velocity", "acceleration"},
    "arc": {"center", "axis", "omega"},
}
_SENSOR_EFFECTS = ("psf", "shot_noise", "read_noise", "quantization")
_CAMERA_AXIS = {"x": 0, "y": 1}
_JSON_FIELDS = ("camera_pose", "motion", "rotation", "sensor_overrides", "methods")


# Конфигурация


@dataclass(frozen=True)
class Effects:
    """Какие эффекты включены. Выключение делает ``build``, а не рендер.

    Выключенный эффект:

    - ``perspective`` — плоскость марки параллельна сенсору: образ — круг с
      центром в проекции центра марки, эксцентриситета нет (наклон и
      вращение не действуют);
    - ``exposure`` — ``nt = 1``, один подшаг в середине экспозиции (сигнал
      по-прежнему от ``tau``);
    - ``motion`` — центр неподвижен в положении на момент ``t_ref``;
    - ``rotation`` — ориентация постоянна, как в момент ``t_ref``;
    - ``rolling_shutter`` — глобальный затвор;
    - ``psf``, ``shot_noise``, ``read_noise``, ``quantization`` — одноимённые
      флаги сенсора.
    """

    perspective: bool = True
    exposure: bool = True
    motion: bool = True
    rotation: bool = True
    rolling_shutter: bool = True
    psf: bool = True
    shot_noise: bool = True
    read_noise: bool = True
    quantization: bool = True

    @classmethod
    def none(cls) -> Effects:
        """Все эффекты выключены."""
        return cls(**dict.fromkeys(EFFECTS, False))


#: Имена эффектов в порядке полей ``Effects``.
EFFECTS = tuple(f.name for f in fields(Effects))


def _freeze(value: Any) -> Any:
    """Списки и массивы -> кортежи, numpy-числа -> числа Python, рекурсивно."""
    if isinstance(value, Mapping):
        return {str(k): _freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _choice(value: str, options: Sequence[str] | Mapping[str, Any], name: str) -> None:
    if value not in options:
        raise ValueError(
            f"{name} = {value!r}, допустимо: {', '.join(sorted(options))}"
        )


@dataclass(frozen=True)
class Config:
    """Описание одной точки эксперимента.

    Параметры
    ---------
    camera
        Пресет камеры из ``presets.CAMERAS``.
    camera_pose
        ``Pose`` или ``(R, T)``; ``None`` — камера в начале мира, смотрит по +z.
    radius
        Радиус марки, м.
    position
        Центр марки в момент ``t_ref``, м, в мире.
    tilt_deg, tilt_axis, tilt_reference
        Наклон нормали, градусы, вокруг оси ``"x"`` или ``"y"`` камеры (её
        части, перпендикулярной направлению отсчёта). Отсчёт
        ``tilt_reference``: ``"optical_axis"`` (по умолчанию) — от минус оси
        камеры, то есть наклон к плоскости изображения, как в формуле
        эксцентриситета; ``"line_of_sight"`` — от направления на камеру.
        На оптической оси оба совпадают. Положительный наклон вокруг ``"y"``
        поворачивает нормаль к -x камеры.
    motion
        Закон движения центра: ``{"type": "static"}``,
        ``{"type": "linear", "velocity": V}``,
        ``{"type": "accel", "velocity": V, "acceleration": A}`` (скорость в
        момент ``t_ref``), ``{"type": "arc", "center": c, "axis": a,
        "omega": w}`` (``position - center`` перпендикулярно ``axis``).
    rotation
        ``None`` или ``{"axis": a, "omega": w}`` — вращение марки вокруг
        мировой оси.
    sensor, sensor_overrides
        Пресет сенсора из ``presets.SENSORS`` и замены его полей. Флаги
        этапов сенсора задаются через ``effects``, не здесь.
    tau, fill, tau_ref
        Экспозиция, с; поток подобран так, что при экспозиции ``tau_ref``
        (``None`` — ``tau``) марка заполняет ``fill`` ёмкости. Для перебора
        по ``tau`` с постоянной яркостью сцены задайте ``tau_ref``.
    t_ref
        Момент привязки: ``"mid"`` или ``"start"`` экспозиции.
    ss, nt, shutter, t_row
        Параметры рендера.
    n_mc, seed
        Реализаций шума и seed.
    methods
        Методы оценки; по умолчанию все из ``ESTIMATORS``.
    effects
        Включённые эффекты.
    reference
        Эталон для метрик, один из ``REFERENCES``.
    """

    camera: str = "review_16mm"
    camera_pose: Any = None
    radius: float = 0.005
    position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tilt_deg: float = 0.0
    tilt_axis: str = "y"
    tilt_reference: str = "optical_axis"
    motion: dict[str, Any] = field(default_factory=lambda: {"type": "static"})
    rotation: dict[str, Any] | None = None
    sensor: str = "review_16mm"
    sensor_overrides: dict[str, Any] = field(default_factory=dict)
    tau: float = 1e-3
    fill: float = 0.7
    tau_ref: float | None = None
    t_ref: str = "mid"
    ss: int = 16
    nt: int = 64
    shutter: str = "global"
    t_row: float = 0.0
    n_mc: int = 500
    seed: int = 2026
    methods: tuple[str, ...] = field(default_factory=lambda: tuple(ESTIMATORS))
    effects: Effects = field(default_factory=Effects)
    reference: str = "exposure_conic"

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # обход frozen, только здесь
        pose = self.camera_pose
        if isinstance(pose, Pose):
            pose = (pose.R, pose.T)
        set_(self, "camera_pose", None if pose is None else _freeze(pose))
        for name in ("position", "motion", "rotation", "sensor_overrides", "methods"):
            set_(self, name, _freeze(getattr(self, name)))
        if isinstance(self.effects, Mapping):
            set_(self, "effects", Effects(**self.effects))

        if len(self.position) != 3:
            raise ValueError(f"position: нужно 3 числа, получено {self.position}")
        _choice(self.camera, presets.CAMERAS, "camera")
        _choice(self.sensor, presets.SENSORS, "sensor")
        _choice(self.tilt_axis, _CAMERA_AXIS, "tilt_axis")
        _choice(self.tilt_reference, TILT_REFERENCES, "tilt_reference")
        _choice(self.motion.get("type"), _MOTION_KEYS, "motion['type']")
        _choice(self.t_ref, ("start", "mid"), "t_ref")
        _choice(self.shutter, ("global", "rolling"), "shutter")
        _choice(self.reference, REFERENCES, "reference")
        for m in self.methods:
            _choice(m, ESTIMATORS, "methods")
        clash = set(_SENSOR_EFFECTS) & set(self.sensor_overrides)
        if clash:
            raise ValueError(
                f"флаги сенсора {sorted(clash)} задаются через effects, "
                "а не sensor_overrides"
            )


@dataclass(frozen=True, eq=False)
class Setup:
    """Объекты одной точки, собранные ``build`` с учётом ``effects``."""

    camera: Camera
    pose: Pose
    target: Target
    motion: Motion
    sensor: SensorModel
    tau: float
    t_ref: float
    ss: int
    nt: int
    shutter: str
    t_row: float


def _rotate(axis: NDArray, angle: float) -> NDArray:
    return _SciRotation.from_rotvec(axis * angle).as_matrix()


def _unit(v: Any, name: str) -> NDArray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if a.shape != (3,) or not n > 0:
        raise ValueError(f"{name}: нужен ненулевой вектор из 3 чисел, получено {v}")
    return a / n


def _translation(motion: Mapping[str, Any], X_ref: NDArray, t_ref: float):
    kind = motion["type"]
    params = {k: v for k, v in motion.items() if k != "type"}
    if set(params) != _MOTION_KEYS[kind]:
        raise ValueError(
            f"motion {kind!r}: нужны ключи {sorted(_MOTION_KEYS[kind])}, "
            f"получены {sorted(params)}"
        )
    if kind == "static":
        return Static(X_ref)
    if kind == "linear":
        V = np.asarray(params["velocity"], dtype=np.float64)
        return Linear(X_ref - V * t_ref, V)
    if kind == "accel":
        V = np.asarray(params["velocity"], dtype=np.float64)
        A = np.asarray(params["acceleration"], dtype=np.float64)
        V0 = V - A * t_ref
        return Accel(X_ref - V0 * t_ref - 0.5 * A * t_ref**2, V0, A)
    center = np.asarray(params["center"], dtype=np.float64)
    axis = _unit(params["axis"], "motion['axis']")
    omega = float(params["omega"])
    start = center + _rotate(axis, -omega * t_ref) @ (X_ref - center)
    return Arc(center, axis, start, omega)


def _tilted_normal(
    pose: Pose, X: NDArray, tilt_deg: float, tilt_axis: str, reference: str
) -> NDArray:
    """Нормаль, наклонённая на ``tilt_deg`` от направления отсчёта ``d``.

    ``d`` — минус ось камеры в мире (``"optical_axis"``) или направление на
    камеру (``"line_of_sight"``); ось наклона — ось камеры ``tilt_axis`` без
    составляющей вдоль ``d``.
    """
    if reference == "optical_axis":
        d = -pose.R[2]
    else:
        d = _unit(pose.center - X, "направление на камеру")
    a = pose.R[_CAMERA_AXIS[tilt_axis]]  # ось камеры в мире
    a = a - (a @ d) * d
    if np.linalg.norm(a) < 1e-9:
        raise ValueError(f"направление отсчёта параллельно оси наклона {tilt_axis!r}")
    a /= np.linalg.norm(a)
    t = math.radians(tilt_deg)
    return d * math.cos(t) + np.cross(a, d) * math.sin(t)


def build(config: Config) -> Setup:
    """Собрать объекты точки; выключенные эффекты заменяются здесь."""
    cfg, eff = config, config.effects
    camera = presets.camera(cfg.camera)
    pose = (
        Pose(np.eye(3), np.zeros(3))
        if cfg.camera_pose is None
        else Pose(*cfg.camera_pose)
    )
    t_ref = reference_time(T_K, cfg.tau, cfg.t_ref)
    X_ref = np.array(cfg.position, dtype=np.float64)

    translation = _translation(cfg.motion, X_ref, t_ref)
    normal = _tilted_normal(
        pose, X_ref, cfg.tilt_deg, cfg.tilt_axis, cfg.tilt_reference
    )
    R_ref = frame_from_normal(normal)
    if cfg.rotation is None:
        orientation = FixedOrientation(R_ref)
    else:
        axis = _unit(cfg.rotation["axis"], "rotation['axis']")
        omega = float(cfg.rotation["omega"])
        orientation = AxisRotation(_rotate(axis, -omega * t_ref) @ R_ref, axis, omega)

    if not eff.motion:
        translation = Static(X_ref)
    if not eff.rotation:
        orientation = FixedOrientation(R_ref)
    if not eff.perspective:
        orientation = FixedOrientation.from_normal(-pose.R[2])  # || сенсору

    tau_ref = cfg.tau if cfg.tau_ref is None else cfg.tau_ref
    flags = {name: getattr(eff, name) for name in _SENSOR_EFFECTS}
    sensor = presets.sensor(
        cfg.sensor, tau_ref, cfg.fill, **cfg.sensor_overrides, **flags
    )
    rolling = cfg.shutter == "rolling" and eff.rolling_shutter
    return Setup(
        camera=camera,
        pose=pose,
        target=Target(cfg.radius),
        motion=Motion(translation, orientation),
        sensor=sensor,
        tau=cfg.tau,
        t_ref=t_ref,
        ss=cfg.ss,
        nt=cfg.nt if eff.exposure else 1,
        shutter="rolling" if rolling else "global",
        t_row=cfg.t_row if rolling else 0.0,
    )


# Одна точка


def references(setup: Setup) -> tuple[dict[str, NDArray], float]:
    """Эталоны центра и сдвиг времени строки, с.
    """
    s = setup
    radius = s.target.radius

    def project(t: float) -> NDArray:
        return s.camera.project(s.pose.to_camera(s.motion.position(t)))

    shift = 0.0
    if s.shutter == "rolling":
        for _ in range(2):  # строка зависит от сдвига; t_row мал, хватает двух
            shift = float(project(T_K + shift + s.tau / 2)[1]) * s.t_row
    t0 = T_K + shift
    mid = t0 + s.tau / 2
    refs = {
        "proj_start": project(t0),
        "proj_mid": project(mid),
        "conic_mid": conic_center(
            s.camera, s.pose, s.motion.position(mid), s.motion.orientation(mid), radius
        ),
        "exposure_conic": exposure_conic_center(
            s.camera, s.pose, s.motion, radius, t0, s.tau, s.nt
        ),
    }
    return refs, shift


def _metrics(est: CenterEstimate, est0: CenterEstimate, ref: NDArray) -> dict:
    """Метрики в пикселях относительно ``ref``."""
    d = est.uv[est.ok] - ref
    k = len(d)
    nan2 = np.full(2, np.nan)
    bias = d.mean(axis=0) if k else nan2
    std = d.std(axis=0, ddof=1) if k > 1 else nan2
    se = std / math.sqrt(k) if k > 1 else nan2
    rmse = math.sqrt(float(np.mean(np.sum(d**2, axis=1)))) if k else math.nan
    err0 = est0.uv - ref  # nan, если без шума метод не отработал
    return {
        "bias_u": bias[0],
        "bias_v": bias[1],
        "bias": math.hypot(*bias),
        "std_u": std[0],
        "std_v": std[1],
        "rmse": rmse,
        "se_bias_u": se[0],
        "se_bias_v": se[1],
        "err0_u": err0[0],
        "err0_v": err0[1],
        "err0": math.hypot(*err0),
    }


def config_row(config: Config) -> dict[str, Any]:
    """Параметры конфигурации плоским словарём для строки таблицы.
    """
    row: dict[str, Any] = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if f.name == "effects":
            row.update({f"effects.{k}": getattr(value, k) for k in EFFECTS})
        elif f.name == "position":
            row.update(zip(("position.x", "position.y", "position.z"), value))
        elif f.name in _JSON_FIELDS:
            row[f.name] = json.dumps(value, ensure_ascii=False)
        elif f.name == "tau_ref":
            row[f.name] = config.tau if value is None else value
        else:
            row[f.name] = value
    return row


def run_point(
    config: Config, seed: np.random.SeedSequence | None = None
) -> list[dict[str, Any]]:
    """Метрики всех методов в одной точке; одна строка (словарь) на метод
    """
    s = build(config)
    rng = np.random.default_rng(seed if seed is not None else config.seed)
    base = config_row(config)
    res = render(
        s.camera,
        s.pose,
        s.target,
        s.motion,
        T_K,
        s.tau,
        ss=s.ss,
        nt=s.nt,
        shutter=s.shutter,
        t_row=s.t_row,
        margin_px=s.sensor.required_margin_px,
    )
    status = "empty" if res.empty else "clipped" if res.clipped else "ok"
    if status != "ok":
        return [{**base, "method": m, "status": status} for m in config.methods]

    refs, shift = references(s)
    ref = refs[config.reference]
    z = float(s.pose.to_camera(s.motion.position(s.t_ref))[2])
    um_per_px = z / s.camera.fx * 1e6
    clean = replace(s.sensor, shot_noise=False, read_noise=False, quantization=False)
    image0 = clean.expose(res.irradiance, s.tau, rng)  # шум не генерируется
    frames = s.sensor.expose_many(res.irradiance, s.tau, rng, config.n_mc)
    point = {
        **base,
        "status": status,
        "z_m": z,
        "f_px": s.camera.fx,
        "um_per_px": um_per_px,
        "radius_px": s.camera.fx * s.target.radius / z,
        "signal_e": float(res.irradiance.sum() * s.sensor.flux_e_per_s * s.tau),
        "roi.u0": res.roi[0],
        "roi.v0": res.roi[1],
        "roi.w": res.roi[2],
        "roi.h": res.roi[3],
        "ref_time_shift": shift,
    }
    for name, uv in refs.items():
        point[f"ref.{name}.u"], point[f"ref.{name}.v"] = (float(x) for x in uv)

    black = s.sensor.black_level_dn
    rows = []
    for method in config.methods:
        est0 = estimate(method, image0, res.roi, black_level_dn=black)
        est = estimate(method, frames, res.roi, black_level_dn=black)
        row = {**point, "method": method}
        for key, value in _metrics(est, est0, ref).items():
            row[f"{key}_px"] = float(value)
            row[f"{key}_um"] = float(value) * um_per_px
        row["ok_frac"] = float(est.ok.mean())
        row["ok0_frac"] = float(est0.ok)
        rows.append(row)
    return rows


# Перебор


def _set_path(obj: Any, path: str, value: Any) -> Any:
    """Копия ``obj`` с заменой поля по пути ``"a.b"`` (датаклассы и словари)."""
    head, _, rest = path.partition(".")
    if is_dataclass(obj):
        if head not in {f.name for f in fields(obj)}:
            raise ValueError(f"нет поля {head!r} в {type(obj).__name__}")
        new = _set_path(getattr(obj, head), rest, value) if rest else value
        return replace(obj, **{head: new})
    if isinstance(obj, Mapping):
        new = _set_path(obj[head], rest, value) if rest else value
        return {**obj, head: new}
    raise ValueError(f"путь {path!r}: {type(obj).__name__} не датакласс и не словарь")


def _n_workers(n_jobs: int) -> int:
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise TypeError(f"n_jobs должно быть целым, получено {n_jobs!r}")
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError(f"n_jobs должно быть >= 1 или -1, получено {n_jobs}")
    return n_jobs


def _run_all(
    configs: Sequence[Config],
    seed: int,
    labels: Sequence[Mapping[str, Any]],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Точки с независимыми потомками ``SeedSequence(seed)``; колонка ``point``.

    Seed каждой точки свой, поэтому результат не зависит от ``n_jobs``.
    """
    seeds = np.random.SeedSequence(seed).spawn(len(configs))
    workers = min(_n_workers(n_jobs), len(configs))
    if workers <= 1:
        results = [run_point(cfg, ss) for cfg, ss in zip(configs, seeds, strict=True)]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            results = list(pool.map(run_point, configs, seeds))
    rows = []
    for i, (point_rows, label) in enumerate(zip(results, labels, strict=True)):
        rows += [{"point": i, **label, **row} for row in point_rows]
    return pd.DataFrame(rows)


def sweep(
    base: Config, grid: Mapping[str, Sequence[Any]], *, n_jobs: int = 1
) -> pd.DataFrame:
    """Декартово произведение значений ``grid`` вокруг ``base``.

    Ключи — поля ``Config``; вложенные через точку (``"effects.psf"``,
    ``"motion.velocity"``). Строк: произведение длин списков на число методов.

    ``n_jobs`` — число процессов (``-1`` — все ядра); результат от него не
    зависит. Процессы запускаются методом ``spawn``: скрипт, который
    вызывает перебор, должен быть под ``if __name__ == "__main__":``, а
    методы оценки — зарегистрированы при импорте модуля, а не во время
    выполнения.
    """
    keys = list(grid)
    configs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = base
        for key, value in zip(keys, combo, strict=True):
            cfg = _set_path(cfg, key, value)
        configs.append(cfg)
    return _run_all(configs, base.seed, [{}] * len(configs), n_jobs)


def ablation(
    base: Config, effects: Sequence[str] | None = None, *, n_jobs: int = 1
) -> pd.DataFrame:
    """Режимы включения эффектов: колонки ``mode`` и ``effect``.

    ``n_jobs`` — как в ``sweep``.
    """
    names = list(EFFECTS if effects is None else effects)
    for name in names:
        _choice(name, EFFECTS, "effect")

    def with_on(on: set[str]) -> Config:
        flags = {**asdict(base.effects), **{n: n in on for n in names}}
        return replace(base, effects=Effects(**flags))

    plan = [("none", None, with_on(set())), ("all", None, with_on(set(names)))]
    plan += [("single", n, with_on({n})) for n in names]
    plan += [("leave_one_out", n, with_on(set(names) - {n})) for n in names]
    labels = [{"mode": mode, "effect": eff} for mode, eff, _ in plan]
    return _run_all([cfg for _, _, cfg in plan], base.seed, labels, n_jobs)


#: Метрики ``additivity`` по умолчанию: смещения и дисперсии складываются.
ADDITIVE_METRICS = (
    "err0_u_px",
    "err0_v_px",
    "bias_u_px",
    "bias_v_px",
    "var_u_px",
    "var_v_px",
)


def additivity(
    df: pd.DataFrame, metrics: Sequence[str] = ADDITIVE_METRICS
) -> pd.DataFrame:
    """Проверка аддитивности эффектов по таблице ``ablation``..
    """
    d = df[df["status"] == "ok"].copy()
    for m in metrics:
        if m.startswith("var_") and m not in d:
            d[m] = d["std_" + m.removeprefix("var_")] ** 2
    cols = list(metrics)

    def mode(name: str) -> pd.DataFrame:
        return d[d["mode"] == name].set_index("method")[cols]

    none = mode("none")
    single = mode("single")
    contrib = single - none.reindex(single.index).to_numpy()
    sum_single = contrib.groupby(level=0).sum()
    total = mode("all") - none
    out = {}
    for m in cols:
        out[f"{m}.sum_single"] = sum_single[m]
        out[f"{m}.all"] = total[m]
        out[f"{m}.residual"] = sum_single[m] - total[m]
    return pd.DataFrame(out)


# Сохранение


class Run(NamedTuple):

    df: pd.DataFrame
    configs: dict[str, Any]
    meta: dict[str, Any]
    path: Path


def _to_json(obj: Any) -> Any:
    if isinstance(obj, Config):
        return {"__config__": _to_json(asdict(obj))}
    if is_dataclass(obj):
        return _to_json(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _from_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__config__" in obj:
            return Config(**obj["__config__"])
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [
            _from_json(v) if isinstance(v, dict) and "__config__" in v else _freeze(v)
            for v in obj
        ]
    return obj


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _versions() -> dict[str, str]:
    import pyarrow
    import scipy
    import skimage

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit-image": skimage.__version__,
    }


def save_run(
    df: pd.DataFrame,
    configs: Config | Mapping[str, Any],
    name: str,
    root: str | Path = RUNS_DIR,
) -> Path:
    """Сохранить запуск в ``root/<YYYYMMDD_HHMMSS>_<name>/`` и вернуть путь.
    """
    if isinstance(configs, Config):
        configs = {"base": configs}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{stamp}_{name}"
    k = 1
    while path.exists():  # два запуска в одну секунду
        k += 1
        path = Path(root) / f"{stamp}_{name}_{k}"
    path.mkdir(parents=True)

    df.to_parquet(path / "results.parquet")
    _write_json(path / "config.json", _to_json(configs))
    commit = _git("rev-parse", "HEAD")
    base = configs.get("base")
    meta = {
        "name": name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "photosim": photosim.__version__,
        "git_commit": commit,
        "git_dirty": None if commit is None else bool(_git("status", "--porcelain")),
        "seed": base.seed if isinstance(base, Config) else None,
        "rows": len(df),
        "versions": _versions(),
    }
    _write_json(path / "meta.json", meta)
    return path


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_run(path: str | Path) -> Run:
    """Прочитать запуск, сохранённый ``save_run``."""
    path = Path(path)
    return Run(
        df=pd.read_parquet(path / "results.parquet"),
        configs=_from_json(json.loads((path / "config.json").read_text())),
        meta=json.loads((path / "meta.json").read_text()),
        path=path,
    )
