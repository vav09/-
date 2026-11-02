"""
pkps_core.py — расчётное ядро.

Дополнительный волновой изгибающий момент на миделе судов классов «М», «О»,
«Р», «Л» по ПКПС РКО, часть I, п. 2.2.10 (формулы 2.2.10-1 … 2.2.10-21).

Модуль не зависит от интерфейса: его можно использовать и из Streamlit,
и из скрипта, и из тестов.

Единицы (как в Правилах):
    L, B, T, Tн, h  — м
    V0 (v_tv)       — км/ч   (!)
    I               — м^4
    D               — кН
    моменты         — кН·м
    силы            — кН
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


def _c(x: float, nd: int = 2) -> str:
    """Число с десятичной запятой для сообщений: 3.5 → '3,50'."""
    return f"{x:.{nd}f}".replace(".", ",")


# --------------------------------------------------------------------------
# Исключения
# --------------------------------------------------------------------------
class InputError(ValueError):
    """Недопустимые исходные данные."""


class RulesRangeError(ArithmeticError):
    """Расчёт вышел за область, в которой Правила дают формулу."""


# --------------------------------------------------------------------------
# Справочные данные
# --------------------------------------------------------------------------
class BasinClass(str, Enum):
    """Разряд бассейна / класс судна (п. 2.2.10)."""
    M = "М"
    O = "О"
    R = "Р"
    L = "Л"


class ShipType(str, Enum):
    """Тип судна — нужен для коэффициента ks (формула 2.2.10-10)."""
    CARGO = "грузовое"
    PASSENGER = "пассажирское"
    TUG = "буксир/толкач"


@dataclass(frozen=True)
class ClassParams:
    eps: float        # ε
    eta: float        # η
    omega_sr: float   # ωср, с^-1
    v1: float         # v1, м/с
    h: float          # расчётная высота волны, м (табл. 2.1.2)


# Табл. 2.2.10-1 и табл. 2.1.2
CLASS_PARAMS: dict[BasinClass, ClassParams] = {
    BasinClass.M: ClassParams(0.920, 1.000, 1.18, 5.42, 3.0),
    BasinClass.O: ClassParams(0.805, 0.874, 1.64, 4.14, 2.0),
    BasinClass.R: ClassParams(0.848, 0.874, 2.22, 3.21, 1.2),
    BasinClass.L: ClassParams(0.874, 0.874, 3.28, 2.26, 0.6),
}

# Коэффициент ks (к формуле 2.2.10-10)
KS: dict[ShipType, float] = {
    ShipType.CARGO: 123e4,
    ShipType.PASSENGER: 117e4,
    ShipType.TUG: 104e4,
}

# Предельные соотношения главных размерений, табл. 2.1.3-1: (L/H, B/H).
# «грузовое» — сухогрузные трюмные суда (у наливных пределы шире).
MAX_RATIOS: dict[ShipType, dict[BasinClass, tuple[float, float]]] = {
    ShipType.CARGO: {BasinClass.M: (25.0, 4.0), BasinClass.O: (27.0, 5.0),
                     BasinClass.R: (28.0, 5.0), BasinClass.L: (28.0, 5.0)},
    ShipType.PASSENGER: {BasinClass.M: (25.0, 4.0), BasinClass.O: (27.0, 5.0),
                         BasinClass.R: (28.0, 5.0), BasinClass.L: (28.0, 5.0)},
    ShipType.TUG: {BasinClass.M: (18.0, 3.5), BasinClass.O: (18.0, 3.5),
                   BasinClass.R: (20.0, 4.0), BasinClass.L: (20.0, 4.0)},
}

# Таблицы Правил для сверки с формулами (колонки: «М», «О», «Р», «Л»)
CLASS_COLUMN = {BasinClass.M: 0, BasinClass.O: 1, BasinClass.R: 2, BasinClass.L: 3}
TABLE_KD = {0.55: 0.487, 0.60: 0.527, 0.65: 0.571, 0.70: 0.619, 0.75: 0.670,
            0.80: 0.726, 0.85: 0.787, 0.90: 0.852, 0.95: 0.923, 1.00: 1.000}      # 2.2.10-2
TABLE_KT = {                                                                         # 2.2.10-3
    0.0: (1.000, 1.000, 1.000, 1.000), 0.5: (0.939, 0.897, 0.834, 0.696),
    1.0: (0.881, 0.805, 0.696, 0.485), 1.5: (0.827, 0.722, 0.581, 0.337),
    2.0: (0.776, 0.647, 0.485, 0.235), 2.5: (0.729, 0.581, 0.404, 0.163),
    3.0: (0.684, 0.521, 0.337, 0.114), 3.5: (0.642, 0.467, 0.281, 0.079),
    4.0: (0.603, 0.419, 0.235, 0.055), 4.5: (0.566, 0.376, 0.196, 0.038)}
TABLE_KB = {                                                                         # 2.2.10-4
    3.0: (0.911, 0.853, 0.771, 0.610), 6.0: (0.832, 0.735, 0.610, 0.408),
    9.0: (0.762, 0.638, 0.493, 0.295), 12.0: (0.700, 0.559, 0.408, 0.227),
    15.0: (0.646, 0.493, 0.344, 0.183), 18.0: (0.597, 0.439, 0.295, 0.153),
    21.0: (0.553, 0.393, 0.257, 0.131), 24.0: (0.514, 0.355, 0.227, 0.115),
    27.0: (0.479, 0.323, 0.203, 0.102), 30.0: (0.448, 0.295, 0.183, 0.092)}


def _interp_table(xs: list[float], ys: list[float], x: float) -> Optional[float]:
    """Линейная интерполяция по таблице (п. 2.1.11); вне таблицы — None."""
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def table_k_delta(cb: float) -> Optional[float]:
    return _interp_table(sorted(TABLE_KD), [TABLE_KD[k] for k in sorted(TABLE_KD)], cb)


def table_k_t(basin: BasinClass, a: float) -> Optional[float]:
    c = CLASS_COLUMN[basin]
    xs = sorted(TABLE_KT)
    return _interp_table(xs, [TABLE_KT[x][c] for x in xs], a)


def table_k_b(basin: BasinClass, b: float) -> Optional[float]:
    c = CLASS_COLUMN[basin]
    xs = sorted(TABLE_KB)
    return _interp_table(xs, [TABLE_KB[x][c] for x in xs], b)


SPEED_UNITS = {"км/ч": 1.0, "м/с": 3.6, "уз": 1.852}   # множитель → км/ч


def speed_to_kmh(value: float, unit: str) -> float:
    """Перевод скорости в км/ч (в таких единицах её требуют формулы Правил)."""
    return value * SPEED_UNITS[unit]


def displacement_from_cb(cb: float, L: float, B: float, T: float, rho: float = 1.0) -> float:
    """Водоизмещение D, кН, по коэффициенту полноты (приближённо).
    rho — плотность воды, т/м³ (1,0 — пресная; 1,025 — морская)."""
    return 9.81 * rho * cb * L * B * T


# --------------------------------------------------------------------------
# Входные данные
# --------------------------------------------------------------------------
@dataclass
class ShipInput:
    basin: BasinClass
    ship_type: ShipType
    L: float          # длина, м
    B: float          # ширина, м
    T: float          # осадка, м
    Tn: float         # осадка носом, м
    Cb: float         # коэффициент полноты водоизмещения δ
    V0_kmh: float     # скорость хода на тихой воде vтв, км/ч
    I: float          # момент инерции миделевого сечения, м^4
    D: Optional[float] = None   # водоизмещение, кН; None → оценка по Cb
    rho: float = 1.0            # плотность воды для оценки D, т/м³
    H: Optional[float] = None   # высота борта, м (для 3D-модели и проверки пропорций)

    @property
    def T_stern(self) -> float:
        """Осадка кормой при средней осадке T и осадке носом Tн."""
        return 2.0 * self.T - self.Tn

    def resolved_D(self) -> float:
        if self.D is not None:
            return self.D
        return displacement_from_cb(self.Cb, self.L, self.B, self.T, self.rho)

    def validate(self) -> list[str]:
        """Бросает InputError при недопустимых данных, возвращает предупреждения."""
        for name, val in (("L", self.L), ("B", self.B), ("T", self.T),
                          ("Tн", self.Tn), ("Cb", self.Cb), ("I", self.I)):
            if not (val > 0):
                raise InputError(f"{name} должно быть больше нуля.")
        if self.V0_kmh < 0:
            raise InputError("Скорость V0 не может быть отрицательной.")
        if self.D is not None and not (self.D > 0):
            raise InputError("Водоизмещение D должно быть больше нуля.")
        if self.Cb > 1.0:
            raise InputError("Коэффициент полноты Cb не может быть больше 1.")
        if self.Tn > 2.0 * self.T:
            raise InputError("Осадка носом Tн сильно отличается от T — проверьте ввод.")

        if self.H is not None:
            if not (self.H > 0):
                raise InputError("Высота борта H должна быть больше нуля.")
            if self.T > self.H:
                raise InputError("Осадка T не может быть больше высоты борта H.")
            if self.Tn > self.H:
                raise InputError("Осадка носом Tн не может быть больше высоты борта H.")

        warns: list[str] = []
        if self.H is not None:
            lim_l, lim_b = MAX_RATIOS[self.ship_type][self.basin]
            if self.L / self.H > lim_l:
                warns.append(f"L/H = {_c(self.L / self.H, 1)} больше предельного {lim_l:g} "
                             f"(табл. 2.1.3-1): раздел 2 ПКПС на такое судно не распространяется (п. 2.1.5).")
            if self.B / self.H > lim_b:
                warns.append(f"B/H = {_c(self.B / self.H, 2)} больше предельного {lim_b:g} "
                             f"(табл. 2.1.3-1): раздел 2 ПКПС на такое судно не распространяется (п. 2.1.5).")
            if self.T_stern > self.H:
                warns.append(f"Осадка кормой Tк = 2T − Tн = {_c(self.T_stern)} м больше высоты борта H.")
        if self.Cb < 0.55:
            warns.append("Cb < 0,55: табл. 2.2.10-2 начинается с 0,55; расчёт идёт по формуле 2.2.10-3.")
        if self.L > 140.0:
            warns.append("L > 140 м: раздел 2 ПКПС распространяется на суда длиной до 140 м (п. 2.1.4).")
        if self.D is None:
            warns.append("D оценено по Cb (приближённо). Для точного расчёта введите D "
                         "из расчётного случая нагрузки.")
        return warns


# --------------------------------------------------------------------------
# Формулы Правил (каждая — отдельная функция, чтобы их можно было тестировать)
# --------------------------------------------------------------------------
def k_delta(cb: float) -> float:
    """(2.2.10-3)"""
    return math.exp(-1.6 * (1.0 - cb))


def k_t(T: float, cb: float, eta: float, h: float) -> float:
    """(2.2.10-4)"""
    return math.exp(-1.14 * T * cb / (eta * h * (2.0 * cb + 1.0)))


def k_b(B: float, cb: float, eta: float, h: float) -> float:
    """(2.2.10-5)  kв = (1 − e^(−x)) / x,  x = 0,19·δ·B/(η·h)."""
    x = 0.19 * cb * B / (eta * h)
    return (1.0 - math.exp(-x)) / x


def wave_moment_Mv(eps: float, kd: float, kt: float, kb: float,
                   B: float, L: float, h: float) -> float:
    """(2.2.10-2)  Волновой изгибающий момент Mв, кН·м."""
    return 0.255 * eps * kd * kt * kb * B * L ** 2 * h


def k_v(h: float, L: float) -> float:
    """(2.2.10-12)"""
    return 1.0 - 19.9 * h / L + 143.0 * (h / L) ** 2


def omega_k(omega_sr: float, kv: float, v_tv: float, L: float) -> float:
    """(2.2.10-9)  Частота волны, наблюдаемая с движущегося судна, с^-1.
    Здесь v_tv в км/ч, делится на L (без корня) — по тексту Правил."""
    return omega_sr + 1.92 * kv * v_tv / L


def sigma_natural(ks: float, I: float, B: float, T: float, D: float, L: float) -> float:
    """(2.2.10-10)  Частота собственных колебаний корпуса σ, с^-1."""
    return ks * math.sqrt(I / ((1.2 + B / (3.0 * T)) * D * L ** 3))


def k_mu(sig: float) -> float:
    """(2.2.10-11)  Не меньше нуля."""
    return max(0.0, 0.0612 * (1.0 - 0.047 * sig - 0.0077 * sig ** 2))


def psi_coefficient(ratio: float) -> float:
    """(2.2.10-13), (2.2.10-14).  ratio = ωк/σ.  При ratio > 1,4 Правила
    формулы не дают — бросаем RulesRangeError."""
    if ratio < 0:
        raise InputError("ωк/σ не может быть отрицательным.")
    if ratio <= 1.0:
        return math.cos(1.3 * ratio ** 2)
    if ratio <= 1.4:
        return 2.167 * ratio - 1.9
    raise RulesRangeError(
        f"ωк/σ = {_c(ratio, 3)} > 1,4: Правила не определяют коэффициент ψ "
        f"(формулы 2.2.10-13, 2.2.10-14).")


def k_p(psi: float, ratio: float, kmu: float) -> float:
    """(2.2.10-8)  Коэффициент влияния волновой вибрации."""
    den = math.sqrt((1.0 - ratio ** 2) ** 2 + (2.0 * kmu * ratio) ** 2)
    if den < 1e-9:
        raise RulesRangeError(
            "Резонанс: ωк = σ при kμ = 0, коэффициент kр обращается в бесконечность.")
    return psi * (1.0 + ratio ** 2 / den)


def Tn_threshold(kv: float, v_tv: float, L: float, eta: float, h: float) -> float:
    """(2.2.10-17)  «Пороговая» осадка носом Tн0, м."""
    return (0.68 + 0.21 * kv * v_tv / math.sqrt(L)) * eta * h


def phi_1(Tn: float, Tn0: float) -> float:
    """Коэффициент φ1 (после 2.2.10-16)."""
    if Tn <= Tn0:
        return 1.0
    if Tn < 1.5 * Tn0:
        return 3.0 - 2.0 * Tn / Tn0
    return 0.0


def b0_coef(cb: float, B: float, T: float, L: float) -> float:
    """(2.2.10-19)"""
    return 4.32 * math.sqrt(cb * (B / L) * (T / L))


def phi_0(b0: float) -> float:
    """(2.2.10-18)"""
    return 1.0 - 1.03 * b0 + b0 ** 2 - 0.417 * b0 ** 3


def v0_shock(kv: float, v_tv: float, L: float, v1: float) -> float:
    """(2.2.10-20)  Скорость v0 (в смысле Правил, не путать с V0 — скоростью хода)."""
    return (0.336 + 0.104 * kv * v_tv / math.sqrt(L)) * v1 + 0.024 * kv * v_tv


def k_y(phi0: float, sig: float, v0: float) -> float:
    """(2.2.10-16)"""
    return 5.3e-4 * phi0 * sig * v0


# --------------------------------------------------------------------------
# Результат
# --------------------------------------------------------------------------
@dataclass
class Step:
    """Одна строка протокола расчёта."""
    symbol: str
    name: str
    ref: str
    value: Optional[float]
    unit: str = ""


@dataclass
class WaveMomentResult:
    inp: ShipInput
    params: ClassParams
    ks: float
    D: float
    a: float                # 3Tδ/(2δ+1), для табл. 2.2.10-3
    b: float                # δB, для табл. 2.2.10-4
    k_delta: float
    k_t: float
    k_b: float
    Mv: float
    kv: float
    omega_k: float
    sigma: float
    k_mu: float
    ratio: float            # ωк/σ
    psi: Optional[float]
    kp: Optional[float]
    Tn0: float
    phi1: float
    b0: float
    phi0: float
    v0: float
    ky: float
    My: float
    Mdv: Optional[float]    # None, если kр определить нельзя
    Ndv: Optional[float]
    warnings: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)


def calculate(inp: ShipInput) -> WaveMomentResult:
    """Полный расчёт по п. 2.2.10."""
    warnings = inp.validate()
    p = CLASS_PARAMS[inp.basin]
    ks = KS[inp.ship_type]
    L, B, T, Tn, cb = inp.L, inp.B, inp.T, inp.Tn, inp.Cb
    v = inp.V0_kmh
    D = inp.resolved_D()

    steps: list[Step] = []

    def add(symbol, name, ref, value, unit=""):
        steps.append(Step(symbol, name, ref, value, unit))

    add("h", "Расчётная высота волны", "табл. 2.1.2", p.h, "м")
    add("ε", "Коэффициент ε", "табл. 2.2.10-1", p.eps)
    add("η", "Коэффициент η", "табл. 2.2.10-1", p.eta)
    add("ωср", "Частота волны", "табл. 2.2.10-1", p.omega_sr, "с⁻¹")
    add("v1", "Параметр v1", "табл. 2.2.10-1", p.v1, "м/с")

    # --- волновой момент Mв ------------------------------------------------
    a = 3.0 * T * cb / (2.0 * cb + 1.0)
    b = cb * B
    kd = k_delta(cb)
    kt = k_t(T, cb, p.eta, p.h)
    kb = k_b(B, cb, p.eta, p.h)
    Mv = wave_moment_Mv(p.eps, kd, kt, kb, B, L, p.h)
    add("a", "Параметр для табл. 2.2.10-3", "2.2.10-6", a, "м")
    add("b", "Параметр для табл. 2.2.10-4", "2.2.10-7", b, "м")
    add("kδ", "Коэффициент kδ", "2.2.10-3", kd)
    add("kт", "Коэффициент kт", "2.2.10-4", kt)
    add("kв", "Коэффициент kв", "2.2.10-5", kb)
    add("Mв", "Волновой изгибающий момент", "2.2.10-2", Mv, "кН·м")

    # --- вибрация: kр ----------------------------------------------------
    kv = k_v(p.h, L)
    wk = omega_k(p.omega_sr, kv, v, L)
    sig = sigma_natural(ks, inp.I, B, T, D, L)
    kmu = k_mu(sig)
    ratio = wk / sig
    add("ks", "Коэффициент ks (тип судна)", "к 2.2.10-10", ks)
    add("D", "Водоизмещение", "—", D, "кН")
    add("kv", "Коэффициент kv", "2.2.10-12", kv)
    add("ωк", "Частота волны с движущегося судна", "2.2.10-9", wk, "с⁻¹")
    add("σ", "Частота собственных колебаний корпуса", "2.2.10-10", sig, "с⁻¹")
    add("kμ", "Коэффициент демпфирования kμ", "2.2.10-11", kmu)
    add("ωк/σ", "Отношение частот", "—", ratio)

    psi: Optional[float] = None
    kp: Optional[float] = None
    try:
        psi = psi_coefficient(ratio)
        add("ψ", "Коэффициент ψ", "2.2.10-13 / 2.2.10-14", psi)
        kp = k_p(psi, ratio, kmu)
        add("kр", "Коэффициент влияния вибрации", "2.2.10-8", kp)
    except RulesRangeError as exc:
        warnings.append(f"{exc} Значения kр, Mдв и Nдв не рассчитаны.")
        add("ψ", "Коэффициент ψ", "2.2.10-13 / 2.2.10-14", None)
        add("kр", "Коэффициент влияния вибрации", "2.2.10-8", None)

    # --- ударный момент Mу ---------------------------------------------------
    Tn0 = Tn_threshold(kv, v, L, p.eta, p.h)
    ph1 = phi_1(Tn, Tn0)
    bz = b0_coef(cb, B, T, L)
    ph0 = phi_0(bz)
    v0 = v0_shock(kv, v, L, p.v1)
    ky = k_y(ph0, sig, v0)
    My = ky * ph1 * D * L
    add("Tн0", "«Пороговая» осадка носом", "2.2.10-17", Tn0, "м")
    add("φ1", "Коэффициент φ1", "к 2.2.10-16", ph1)
    add("b0", "Параметр b0", "2.2.10-19", bz)
    add("φ0", "Коэффициент φ0", "2.2.10-18", ph0)
    add("v0", "Скорость v0 (по Правилам)", "2.2.10-20", v0, "м/с")
    add("kу", "Коэффициент kу", "2.2.10-16", ky)
    add("Mу", "Ударный изгибающий момент", "2.2.10-15", My, "кН·м")
    if ph0 < 0:
        warnings.append("φ0 < 0: проверьте отношения B/L и T/L.")
    if ph1 == 0.0:
        warnings.append(f"Tн = {_c(Tn)} м ≥ 1,5·Tн0 = {_c(1.5 * Tn0)} м: ударный момент Mу = 0 "
                        f"(носовая оконечность достаточно погружена).")

    # --- итог ------------------------------------------------------------------
    Mdv = Ndv = None
    if kp is not None:
        Mdv = kp * Mv + My
        Ndv = 4.0 * Mdv / L
        add("Mдв", "Дополнительный волновой момент (±)", "2.2.10-1", Mdv, "кН·м")
        add("Nдв", "Доп. волновая перерезывающая сила (±)", "2.2.10-21", Ndv, "кН")
    else:
        add("Mдв", "Дополнительный волновой момент (±)", "2.2.10-1", None, "кН·м")
        add("Nдв", "Доп. волновая перерезывающая сила (±)", "2.2.10-21", None, "кН")

    return WaveMomentResult(
        inp=inp, params=p, ks=ks, D=D, a=a, b=b,
        k_delta=kd, k_t=kt, k_b=kb, Mv=Mv, kv=kv, omega_k=wk, sigma=sig,
        k_mu=kmu, ratio=ratio, psi=psi, kp=kp, Tn0=Tn0, phi1=ph1, b0=bz,
        phi0=ph0, v0=v0, ky=ky, My=My, Mdv=Mdv, Ndv=Ndv,
        warnings=warnings, steps=steps)


# --------------------------------------------------------------------------
# Эпюры (рис. 2.2.10)
# --------------------------------------------------------------------------
def default_plateau(basin: BasinClass) -> float:
    """Длина постоянного участка эпюры Mдв в долях L.
    «Р», «Л» — 0,5L (п. 2.2.10). «М», «О» — не менее 0,3L (по 0,15L от миделя);
    по умолчанию принято 0,5L, как на рис. 2.2.10."""
    return 0.5


def min_plateau(basin: BasinClass) -> float:
    return 0.5 if basin in (BasinClass.R, BasinClass.L) else 0.3


def _piecewise(xp: list[float], yp: list[float], n: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """Кусочно-линейная кривая; узлы включены в сетку, углы эпюры точные."""
    x = np.union1d(np.linspace(0.0, 1.0, n), np.array(xp))
    return x, np.interp(x, xp, yp)


def moment_profile(Mdv: float, plateau: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Эпюра |Mдв| по длине: x/L и значения, кН·м.
    Постоянна на длине plateau·L в средней части, к оконечностям — линейно до нуля."""
    if not (0.0 < plateau < 1.0):
        raise InputError("Длина постоянного участка должна быть в пределах (0; 1)·L.")
    edge = (1.0 - plateau) / 2.0
    x, y = _piecewise([0.0, edge, 1.0 - edge, 1.0], [0.0, 1.0, 1.0, 0.0])
    return x, Mdv * y


# Узлы эпюры Nдв по рис. 2.2.10 (20 интервалов): 0 → 0,15L → 0,30L → 0,40L …
_N_X = [0.0, 0.15, 0.30, 0.40, 0.60, 0.70, 0.85, 1.0]
_N_Y = [0.0, 1.0, 1.0, 0.7, 0.7, 1.0, 1.0, 0.0]


def shear_profile(Ndv: float) -> tuple[np.ndarray, np.ndarray]:
    """Эпюра |Nдв| по рис. 2.2.10: x/L и значения, кН."""
    x, y = _piecewise(_N_X, _N_Y)
    return x, Ndv * y


def combine_with_still_water(x_tv: list[float], M_tv: list[float],
                             x: np.ndarray, Mdv_x: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mр = Mтв ± Mдв (п. 2.2.12-1) по длине.
    x_tv, M_tv — эпюра момента на тихой воде (x/L, кН·м), не менее 2 точек.
    Возвращает (Mтв(x), Mтв + Mдв(x), Mтв − Mдв(x)).
    Знак Mтв — по принятой пользователем конвенции."""
    if len(x_tv) < 2 or len(x_tv) != len(M_tv):
        raise InputError("Для Mтв нужно не менее двух точек (x/L и значение).")
    order = np.argsort(x_tv)
    xs = np.asarray(x_tv, dtype=float)[order]
    ms = np.asarray(M_tv, dtype=float)[order]
    mtv = np.interp(x, xs, ms)
    return mtv, mtv + Mdv_x, mtv - Mdv_x


# --------------------------------------------------------------------------
# Форматирование
# --------------------------------------------------------------------------
def format_value(v: Optional[float]) -> str:
    """Читаемый вывод числа для протокола."""
    if v is None:
        return "—"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}".replace(",", "\u00a0")
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"
