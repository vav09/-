"""
pkps_hull.py — схематичная 3D-геометрия корпуса для визуализации.

Корпус строится по главным размерениям L, B, H, осадкам T и Tн и коэффициенту
полноты Cb. Это иллюстрация, а не теоретический чертёж: форма обводов
типовая, но размеры (длина, ширина, высота борта, осадка, дифферент) —
точные, а полнота подгоняется так, чтобы коэффициент полноты модели
совпал с введённым Cb (если это возможно для такой формы).

Система координат: x — вдоль судна (корма 0, нос L), y — поперёк (левый борт −),
z — вверх от основной плоскости. Вода лежит горизонтально на уровне z = T,
дифферент показан наклоном самого корпуса.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x и 2.x

WAVE_NONE, WAVE_HOG, WAVE_SAG = "none", "hog", "sag"


# --------------------------------------------------------------------------
# Обводы
# --------------------------------------------------------------------------
def _cb_norm(cb: float) -> float:
    """Cb → 0…1 (0 при 0,55 и меньше; 1 при 1,0)."""
    return float(np.clip((cb - 0.55) / 0.45, 0.0, 1.0))


class _Shape:
    """Плановые и поперечные обводы. pm — доля параллельной вставки."""

    def __init__(self, L: float, B: float, H: float, cb: float, pm: float):
        self.L, self.B, self.H = L, B, H
        cbn = _cb_norm(cb)
        self.pm = pm
        self.run = 0.45 * (1.0 - pm) * L            # длина кормовой скулы
        self.bow = 0.55 * (1.0 - pm) * L            # длина носового заострения
        self.x0 = self.run
        self.x1 = L - self.bow
        self.n_bow = 1.0 + 1.6 * cbn                # полнота носовой ватерлинии
        self.transom = 0.30 + 0.30 * cbn            # ширина транца, доля от B
        self.e_stern = 0.9 + 1.0 * (1.0 - cbn)      # заострённость кормовой скулы
        # радиус скулы: у полных судов меньше
        self.rb = min((0.10 + 0.20 * (1.0 - cbn)) * B, 0.6 * H, 0.45 * B)

    def wl(self, x):
        """Относительная полуширина ватерлинии по длине (0…1)."""
        x = np.asarray(x, dtype=float)
        if self.x0 > 1e-9:
            s = np.clip(x / self.x0, 0.0, 1.0)
            stern = self.transom + (1.0 - self.transom) * np.sin(0.5 * math.pi * s) ** self.e_stern
        else:
            stern = np.ones_like(x)
        if self.bow > 1e-9:
            u = np.clip((x - self.x1) / self.bow, 0.0, 1.0)
            bow = 1.0 - u ** self.n_bow
        else:
            bow = np.ones_like(x)
        return np.where(x < self.x0, stern, np.where(x > self.x1, bow, 1.0))

    def keel(self, x):
        """Подъём киля в носу (z нижней точки корпуса в сечении x)."""
        x = np.asarray(x, dtype=float)
        start = self.x1 + 0.35 * self.bow
        v = np.clip((x - start) / max(self.L - start, 1e-6), 0.0, 1.0)
        return 0.22 * self.H * v ** 2

    def half_width_section(self, zl):
        """Полуширина сечения на высоте zl над килем (без плановой скуловой коррекции)."""
        zl = np.asarray(zl, dtype=float)
        rb = self.rb
        core = self.B / 2.0 - rb
        arc = core + np.sqrt(np.maximum(rb ** 2 - (rb - np.minimum(zl, rb)) ** 2, 0.0))
        return np.where(zl < rb, arc, self.B / 2.0)


def _draft_line(L: float, T: float, Tn: float, x):
    """Осадка по длине при линейном дифференте: корма 2T − Tн, нос Tн."""
    Tk = 2.0 * T - Tn
    return Tk + (Tn - Tk) * np.asarray(x, dtype=float) / L


def displaced_cb(shape: _Shape, T: float, Tn: float, nx: int = 161, nz: int = 48) -> float:
    """Коэффициент полноты модели: V / (L·B·T)."""
    L, H = shape.L, shape.H
    t = np.linspace(0.0, 1.0, nx)
    x = L * (0.5 - 0.5 * np.cos(math.pi * t))
    zb = shape.keel(x)
    d = np.clip(_draft_line(L, T, Tn, x), zb, H)
    frac = np.linspace(0.0, 1.0, nz)
    zl = (d - zb)[:, None] * frac[None, :]
    hw = shape.wl(x)[:, None] * shape.half_width_section(zl)
    area = 2.0 * _trapz(hw, zl, axis=1)
    vol = _trapz(area, x)
    return float(vol / (L * shape.B * T))


def fit_parallel_body(L, B, H, T, Tn, cb) -> tuple[float, float]:
    """Подбирает долю параллельной вставки pm так, чтобы Cb модели ≈ Cb.
    Возвращает (pm, cb_модели). Если цель недостижима — ближайшая граница."""
    lo, hi = 0.0, 0.97
    f = lambda pm: displaced_cb(_Shape(L, B, H, cb, pm), T, Tn)
    f_lo, f_hi = f(lo), f(hi)
    if cb <= f_lo:
        return lo, f_lo
    if cb >= f_hi:
        return hi, f_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) < cb:
            lo = mid
        else:
            hi = mid
    pm = 0.5 * (lo + hi)
    return pm, f(pm)


def _box(x0, x1, y0, y1, z0, z1) -> tuple[np.ndarray, np.ndarray]:
    """Параллелепипед: 8 вершин, 12 треугольников."""
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    t = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 5, 1], [0, 4, 5],
                  [1, 6, 2], [1, 5, 6], [2, 7, 3], [2, 6, 7], [3, 4, 0], [3, 7, 4]], dtype=np.int32)
    return v, t


# --------------------------------------------------------------------------
# Сетка корпуса
# --------------------------------------------------------------------------
@dataclass
class Hull3D:
    verts: np.ndarray                      # (N, 3)
    groups: dict[str, np.ndarray]          # имя → (M, 3) индексы вершин
    waterline: np.ndarray                  # (K, 3) замкнутая ломаная ватерлинии
    L: float
    B: float
    H: float
    T: float
    Tn: float
    Tk: float
    trim_deg: float
    cb_target: float
    cb_model: float
    pm: float
    notes: list[str] = field(default_factory=list)

    def wl_edge(self) -> tuple[np.ndarray, np.ndarray]:
        """Полуширина ватерлинии по длине: (x, hw). Нужна, чтобы вода не заходила внутрь корпуса."""
        n = (len(self.waterline) - 1) // 2
        port = self.waterline[:n]
        return port[:, 0].copy(), -port[:, 1]

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.verts.min(axis=0), self.verts.max(axis=0)


def build_hull(L: float, B: float, H: float, T: float, Tn: float, Cb: float,
               nx: int = 44, jl: int = 7, ju: int = 4) -> Hull3D:
    """Строит сетку корпуса. Группы треугольников:
    under (подводная часть), above (надводный борт), deck, transom, house (надстройка)."""
    pm, cb_model = fit_parallel_body(L, B, H, T, Tn, Cb)
    sh = _Shape(L, B, H, Cb, pm)
    Tk = 2.0 * T - Tn

    t = np.linspace(0.0, 1.0, nx)
    xs = L * (0.5 - 0.5 * np.cos(math.pi * t))          # сгущение к оконечностям
    zb = sh.keel(xs)
    lo_lim = zb + 0.03 * H
    hi_lim = np.maximum(H - 0.03 * H, lo_lim)
    d = np.clip(_draft_line(L, T, Tn, xs), lo_lim, hi_lim)

    R = jl + ju + 1                                      # рядов по высоте на борт
    z = np.empty((nx, R))
    t_low = (np.arange(0, jl + 1) / jl) ** 1.35
    z[:, : jl + 1] = zb[:, None] + (d - zb)[:, None] * t_low[None, :]
    t_up = np.arange(1, ju + 1) / ju
    z[:, jl + 1:] = d[:, None] + (H - d)[:, None] * t_up[None, :]

    hw = sh.wl(xs)[:, None] * sh.half_width_section(z - zb[:, None])

    ring = 2 * R
    verts = np.empty((nx, ring, 3))
    for k in range(ring):
        if k < R:
            row, sign = R - 1 - k, -1.0
        else:
            row, sign = k - R, 1.0
        verts[:, k, 0] = xs
        verts[:, k, 1] = sign * hw[:, row]
        verts[:, k, 2] = z[:, row]

    def vid(i, k):
        return i * ring + k

    under, above, deck, transom = [], [], [], []
    for i in range(nx - 1):
        for k in range(ring - 1):
            a, b, c, e = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            if k < R - 1:
                is_under = (R - 1 - k) <= jl             # верхний ряд квадрата ниже ватерлинии
            elif k == R - 1:
                is_under = True                           # плоское днище
            else:
                is_under = (k - R + 1) <= jl
            (under if is_under else above).extend([(a, b, c), (a, c, e)])
        p0, s0, p1, s1 = vid(i, 0), vid(i, ring - 1), vid(i + 1, 0), vid(i + 1, ring - 1)
        deck.extend([(p0, s0, s1), (p0, s1, p1)])

    flat = verts.reshape(-1, 3)
    centre = flat[:ring].mean(axis=0, keepdims=True)
    cid = flat.shape[0]
    for k in range(ring):
        transom.append((cid, vid(0, k), vid(0, (k + 1) % ring)))
    flat = np.vstack([flat, centre])

    # надстройка в корме (две ступени) — чтобы модель читалась как судно
    xa, xb = 0.05 * L, 0.05 * L + 0.10 * L
    hwh = min(0.30 * B, 0.8 * float(min(sh.wl(xa), sh.wl(xb))) * B / 2.0)
    h1, h2 = 0.55 * H, 0.35 * H
    v1, t1 = _box(xa, xb, -hwh, hwh, H, H + h1)
    v2, t2 = _box(xa + 0.15 * (xb - xa), xb - 0.05 * (xb - xa), -0.7 * hwh, 0.7 * hwh,
                  H + h1, H + h1 + h2)
    base = flat.shape[0]
    flat = np.vstack([flat, v1, v2])
    house = np.vstack([t1 + base, t2 + base + 8])

    # ватерлиния: борт левый (корма→нос), правый (нос→корма), замыкание по транцу
    kp, ks = R - 1 - jl, R + jl
    wl_pts = [flat[vid(i, kp)] for i in range(nx)] + \
             [flat[vid(i, ks)] for i in range(nx - 1, -1, -1)]
    wl_pts.append(wl_pts[0])
    waterline = np.array(wl_pts)

    # дифферент: поворачиваем корпус так, чтобы ватерлиния легла горизонтально на z = T
    theta = math.atan2(Tn - Tk, L)
    c, s = math.cos(theta), math.sin(theta)

    def rotate(p: np.ndarray) -> np.ndarray:
        rx, rz = p[..., 0] - L / 2.0, p[..., 2] - T
        out = p.copy()
        out[..., 0] = L / 2.0 + rx * c + rz * s
        out[..., 2] = T + rz * c - rx * s
        return out

    flat = rotate(flat)
    waterline = rotate(waterline)

    notes: list[str] = []
    if abs(cb_model - Cb) > 0.02:
        notes.append(f"Cb модели ≈ {cb_model:.2f}: для такой формы обводов введённое Cb "
                     f"{Cb:.2f} недостижимо, показана ближайшая полнота.")
    if Tk > H:
        notes.append("Осадка кормой больше высоты борта: корма показана затопленной.")

    return Hull3D(
        verts=flat,
        groups={"under": np.array(under, dtype=np.int32),
                "above": np.array(above, dtype=np.int32),
                "deck": np.array(deck, dtype=np.int32),
                "transom": np.array(transom, dtype=np.int32),
                "house": house.astype(np.int32)},
        waterline=waterline, L=L, B=B, H=H, T=T, Tn=Tn, Tk=Tk,
        trim_deg=math.degrees(theta), cb_target=Cb, cb_model=cb_model, pm=pm, notes=notes)


# --------------------------------------------------------------------------
# Вода и волна
# --------------------------------------------------------------------------
def build_water(L: float, T: float, h: float, mode: str, ext_x: float, ext_y: float,
                cell: float, edge: tuple[np.ndarray, np.ndarray] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Сетка водной поверхности в виде двух полос (левый и правый борт), внутренняя
    кромка которых идёт по ватерлинии корпуса — внутри корпуса воды нет.

    Волна длиной L и высотой h (от подошвы до гребня):
    «hog» — гребень на миделе (перегиб), «sag» — подошва на миделе (прогиб).
    Возвращает (вершины (N,3), треугольники (M,3))."""
    nx = max(int(round(2 * ext_x / cell)), 2)
    nyb = max(int(round(ext_y / cell)), 2)
    xs = np.linspace(L / 2.0 - ext_x, L / 2.0 + ext_x, nx + 1)
    if edge is None:
        hw = np.zeros_like(xs)
    else:
        hw = np.interp(xs, edge[0], edge[1], left=0.0, right=0.0)
    sgn = {WAVE_HOG: 1.0, WAVE_SAG: -1.0}.get(mode, 0.0)
    zline = T + sgn * 0.5 * h * np.cos(2.0 * math.pi * (xs - L / 2.0) / L)

    u = np.linspace(0.0, 1.0, nyb + 1)
    y_port = -ext_y + (ext_y - hw)[:, None] * u[None, :]      # от -ext_y до -hw
    y_star = hw[:, None] + (ext_y - hw)[:, None] * u[None, :]  # от +hw до +ext_y
    Y = np.concatenate([y_port, y_star], axis=1)
    cols = Y.shape[1]
    X = np.repeat(xs[:, None], cols, axis=1)
    Z = np.repeat(zline[:, None], cols, axis=1)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    tris = []
    for i in range(nx):
        for j in range(cols - 1):
            if j == nyb:                      # щель между полосами — место корпуса
                continue
            a = i * cols + j
            b = (i + 1) * cols + j
            c = (i + 1) * cols + j + 1
            e = i * cols + j + 1
            tris.extend([(a, b, c), (a, c, e)])
    return verts, np.array(tris, dtype=np.int32)
