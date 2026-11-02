"""
pkps_beam.py — эквивалентный брус миделевого сечения по ПКПС РКО, часть I.

Что считается
-------------
* Нулевое приближение (все φ = 1): площадь, нейтральная ось z0, момент инерции I0,
  моменты сопротивления палубы и днища. Именно I0 нужен в п. 2.2.10 («при проектных
  толщинах и редукционных коэффициентах, равных единице»).
* Последовательные приближения с редуцированием гибких пластин (п. 2.2.36–2.2.39)
  отдельно для прогиба (сжата палуба) и перегиба (сжато днище), до разницы
  напряжений в крайних кромках ≤ 5 %.
* Нормальные напряжения σ = Mр·z·10⁻³/I (2.2.40-1) и проверка по допускаемым
  напряжениям (2.2.67, табл. 2.2.68).

Единицы ввода: размеры элементов — мм, отстояния z и y — м, A — см², i — см⁴,
моменты — кН·м, напряжения — МПа. Отстояние z — от основной плоскости (ОП).

Что не входит (пока): касательные напряжения (2.2.40-2), предельные моменты
(2.2.76–2.2.80), редуцирование «жёстких» связей при потере устойчивости (2.2.79),
вырезы и обрывающиеся связи (2.2.32–2.2.34) — их учитывает сам расчётчик,
включая в таблицу только связи, которые идут в брус.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Типы элементов и столбцы таблицы
# ----------------------------------------------------------------------------
T_HPLATE = "Лист горизонтальный"
T_VPLATE = "Лист вертикальный"
T_IPLATE = "Лист наклонный"
T_FLAT = "Полоса (ребро)"
T_TEE = "Тавр"
T_ANGLE = "Уголок"
T_ANY = "Произвольный (A, i)"
TYPES = [T_HPLATE, T_VPLATE, T_IPLATE, T_FLAT, T_TEE, T_ANGLE, T_ANY]
PLATES = (T_HPLATE, T_VPLATE, T_IPLATE)
PROFILES = (T_FLAT, T_TEE, T_ANGLE)

SYS_LONG = "прод."
SYS_TRANS = "попер."
SYSTEMS = [SYS_LONG, SYS_TRANS]

# Порядок столбцов в таблице (и в файле проекта)
COLUMNS = ["Связь", "Тип", "n", "b, мм", "t, мм", "bп, мм", "tп, мм", "θ, °", "A, см²", "i, см⁴",
           "z, м", "y, м", "Вниз", "ReH, МПа", "Гибкая", "Система", "a, мм", "b пан., мм",
           "Местная нагрузка"]

DEFAULT_ROW = {"Связь": "", "Тип": T_HPLATE, "n": 1, "b, мм": 0.0, "t, мм": 0.0, "bп, мм": 0.0,
               "tп, мм": 0.0, "θ, °": 0.0, "A, см²": 0.0, "i, см⁴": 0.0, "z, м": 0.0, "y, м": 0.0,
               "Вниз": False, "ReH, МПа": 235.0, "Гибкая": False, "Система": SYS_LONG,
               "a, мм": 0.0, "b пан., мм": 0.0, "Местная нагрузка": False}

HELP = {
    "Тип": "Лист — b ширина (для вертикального и наклонного — длина по обводу), t толщина. "
           "Полоса/Тавр/Уголок — b высота стенки, t толщина стенки, bп × tп — поясок; "
           "z — уровень обшивки, к которой приварен профиль. Произвольный — A и i (собственный) "
           "из сортамента, z — центр тяжести, b — высота (для крайних волокон).",
    "n": "Количество одинаковых элементов в таблице (например, 6 продольных балок днища).",
    "θ, °": "Только для наклонного листа: угол к горизонтали.",
    "z, м": "Отстояние от ОП: для листов — центр тяжести, для профилей — уровень обшивки.",
    "y, м": "Отстояние от ДП — нужно только для эскиза сечения.",
    "Вниз": "Профиль направлен вниз от обшивки (палубные балки, подпалубные стрингеры).",
    "Гибкая": "Пластина, подлежащая редуцированию (п. 2.2.36). Не отмечайте скуловой лист (2.2.37.2).",
    "Система": "Система набора пластины: продольная (2.2.38) или поперечная (2.2.39).",
    "a, мм": "Короткая сторона опорного контура пластины (шпация).",
    "b пан., мм": "Длинная сторона опорного контура пластины.",
    "Местная нагрузка": "Связь несёт местную нагрузку (днище, нагруженные палубы) — "
                        "допускаемые напряжения 0,60σ0 (табл. 2.2.68, п. 2).",
}

# Табл. 2.2.39: редукционные коэффициенты поперечных пластин (толщина в метрах)
TABLE_2239_T = [0.004, 0.006, 0.008, 0.012]
TABLE_2239_TENSION = [0.07, 0.18, 0.33, 0.56]
TABLE_2239_COMPRESSION = [0.03, 0.07, 0.12, 0.28]

CONVERGENCE = 0.05     # 2.2.36: разница напряжений в крайних кромках ≤ 5 %
MAX_ITER = 50

SAG = "прогиб"        # палуба сжата
HOG = "перегиб"       # днище сжато


class BeamError(ValueError):
    """Ошибка в таблице связей."""


# ----------------------------------------------------------------------------
# Геометрия элемента
# ----------------------------------------------------------------------------
@dataclass
class Element:
    name: str
    kind: str
    n: int = 1
    b: float = 0.0          # мм
    t: float = 0.0          # мм
    bf: float = 0.0         # мм
    tf: float = 0.0         # мм
    theta: float = 0.0      # град
    A_cm2: float = 0.0
    i_cm4: float = 0.0
    z: float = 0.0          # м от ОП
    y: float = 0.0          # м от ДП
    down: bool = False
    ReH: float = 235.0
    flexible: bool = False
    system: str = SYS_LONG
    a: float = 0.0          # мм
    b_panel: float = 0.0    # мм
    local_load: bool = False

    # --- вычисляемые (м, м², м⁴) ---
    @property
    def is_plate(self) -> bool:
        return self.kind in PLATES

    def parts(self) -> list[tuple[float, float, float, float, float]]:
        """Прямоугольники элемента: (A, zc, i0, z_low, z_high) для ОДНОГО элемента."""
        b, t, bf, tf = self.b / 1000, self.t / 1000, self.bf / 1000, self.tf / 1000
        k = self.kind
        if k == T_HPLATE:
            return [(b * t, self.z, b * t ** 3 / 12, self.z - t / 2, self.z + t / 2)]
        if k == T_VPLATE:
            return [(b * t, self.z, t * b ** 3 / 12, self.z - b / 2, self.z + b / 2)]
        if k == T_IPLATE:
            s = abs(math.sin(math.radians(self.theta)))
            h = b * s + t * math.cos(math.radians(self.theta))
            return [(b * t, self.z, t * b ** 3 * s ** 2 / 12 + b * t ** 3 * (1 - s ** 2) / 12,
                     self.z - h / 2, self.z + h / 2)]
        if k in PROFILES:
            sg = -1.0 if self.down else 1.0
            out = []
            if b > 0 and t > 0:
                zc = self.z + sg * b / 2
                out.append((b * t, zc, t * b ** 3 / 12, zc - b / 2, zc + b / 2))
            if k != T_FLAT and bf > 0 and tf > 0:
                zc = self.z + sg * (b + tf / 2)
                out.append((bf * tf, zc, bf * tf ** 3 / 12, zc - tf / 2, zc + tf / 2))
            return out
        if k == T_ANY:
            A, i0 = self.A_cm2 * 1e-4, self.i_cm4 * 1e-8
            h = b if b > 0 else 0.0
            return [(A, self.z, i0, self.z - h / 2, self.z + h / 2)]
        raise BeamError(f"Неизвестный тип элемента «{k}»")

    def area(self) -> float:
        return self.n * sum(p[0] for p in self.parts())

    def centroid(self) -> float:
        ps = self.parts()
        A = sum(p[0] for p in ps)
        return sum(p[0] * p[1] for p in ps) / A if A > 0 else self.z

    def own_inertia(self) -> float:
        """Собственный момент инерции n элементов относительно их общего ЦТ, м⁴."""
        ps = self.parts()
        zc = self.centroid()
        return self.n * sum(p[2] + p[0] * (p[1] - zc) ** 2 for p in ps)

    def z_range(self) -> tuple[float, float]:
        ps = self.parts()
        return min(p[3] for p in ps), max(p[4] for p in ps)

    def validate(self, row: int) -> list[str]:
        e = []
        where = f"Строка {row} «{self.name or self.kind}»"
        if self.n < 1:
            e.append(f"{where}: n должно быть не меньше 1.")
        if self.kind == T_ANY:
            if self.A_cm2 <= 0:
                e.append(f"{where}: задайте площадь A.")
        elif self.b <= 0 or self.t <= 0:
            e.append(f"{where}: задайте b и t больше нуля.")
        if self.kind in (T_TEE, T_ANGLE) and (self.bf <= 0 or self.tf <= 0):
            e.append(f"{where}: у тавра и уголка задайте поясок bп × tп.")
        if self.flexible:
            if not self.is_plate:
                e.append(f"{where}: редуцируются только пластины — снимите отметку «Гибкая».")
            elif self.a <= 0:
                e.append(f"{where}: для гибкой пластины задайте a (короткую сторону опорного контура).")
            elif self.system == SYS_TRANS and self.b_panel <= 0:
                e.append(f"{where}: при поперечной системе задайте b пан. (длинную сторону).")
            elif self.b_panel and self.b_panel < self.a:
                e.append(f"{where}: b пан. не может быть меньше a.")
        if self.ReH <= 0:
            e.append(f"{where}: задайте предел текучести ReH.")
        return e


def _num(v, default=0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "да", "yes", "+")
    try:
        return bool(v) and not (isinstance(v, float) and math.isnan(v))
    except Exception:
        return False


def elements_from_records(records: list[dict]) -> list[Element]:
    """Строки таблицы (dict по COLUMNS) → элементы. Пустые строки пропускаются."""
    out = []
    for r in records:
        kind = r.get("Тип") or T_HPLATE
        if isinstance(kind, float):
            kind = T_HPLATE
        el = Element(
            name=str(r.get("Связь") or "").strip() if not isinstance(r.get("Связь"), float) else "",
            kind=str(kind),
            n=int(round(_num(r.get("n"), 1))) or 1,
            b=_num(r.get("b, мм")), t=_num(r.get("t, мм")),
            bf=_num(r.get("bп, мм")), tf=_num(r.get("tп, мм")),
            theta=_num(r.get("θ, °")),
            A_cm2=_num(r.get("A, см²")), i_cm4=_num(r.get("i, см⁴")),
            z=_num(r.get("z, м")), y=_num(r.get("y, м")),
            down=_bool(r.get("Вниз")),
            ReH=_num(r.get("ReH, МПа"), 235.0) or 235.0,
            flexible=_bool(r.get("Гибкая")),
            system=str(r.get("Система") or SYS_LONG),
            a=_num(r.get("a, мм")), b_panel=_num(r.get("b пан., мм")),
            local_load=_bool(r.get("Местная нагрузка")),
        )
        empty = (el.b == 0 and el.t == 0 and el.A_cm2 == 0 and not el.name)
        if not empty:
            out.append(el)
    return out


# ----------------------------------------------------------------------------
# Правила: напряжения и редукционные коэффициенты
# ----------------------------------------------------------------------------
def k_n(ReH: float) -> float:
    """2.2.67-3; для ReH ≤ 235 МПа kн = 1, выше 395 МПа берётся значение при 395."""
    r = min(max(ReH, 235.0), 395.0)
    q = r / 235.0 - 1.0
    return 1.0 - 0.089 * q - 0.129 * q ** 2


def sigma_0(ReH: float) -> float:
    """Опасное напряжение σ0 = kн·ReH (2.2.67-1), МПа."""
    return k_n(ReH) * ReH


def allowable_factor(basin_value: str, local_load: bool) -> float:
    """Табл. 2.2.68, пп. 1–2: доля от опасного напряжения для жёстких связей при общем изгибе."""
    if local_load:
        return 0.60
    return 0.70 if basin_value in ("М-СП", "М-ПР", "М") else 0.75


def sigma_euler_plate(t_mm: float, a_mm: float) -> float:
    """2.2.72-2: σэ = 78,5·(100t/a)², t и a в одних единицах (здесь — мм)."""
    return 78.5 * (100.0 * t_mm / a_mm) ** 2


def sigma_cr_plate(t_mm: float, a_mm: float, ReH: float) -> float:
    """2.2.72-1: критическое напряжение пластины, сжатой вдоль длинной кромки, МПа."""
    se = sigma_euler_plate(t_mm, a_mm)
    if se <= 0.6 * ReH:
        return se
    if se < 1.6 * ReH:
        return (1.63 - 0.8 * math.sqrt(ReH / se)) * ReH
    return ReH


def phi_table_2239(t_mm: float, compressed: bool) -> float:
    """Табл. 2.2.39 с линейной интерполяцией по толщине; вне таблицы — крайние значения."""
    ys = TABLE_2239_COMPRESSION if compressed else TABLE_2239_TENSION
    return float(np.interp(t_mm / 1000.0, TABLE_2239_T, ys))


def phi_limit_2239(t_mm: float, a_mm: float, b_mm: float, sigma_abs: float) -> float:
    """2.2.39-2: верхняя граница φ сжатой поперечной пластины."""
    if sigma_abs <= 1e-9:
        return 1.0
    return (19.0 / sigma_abs) * (100.0 * t_mm / a_mm) ** 2 * (1.0 + a_mm ** 2 / b_mm ** 2) ** 2


def rigid_fraction(el: Element) -> float:
    """Доля ширины пластины, не подлежащая редуцированию (2.2.37.1): по 0,25a, но не более 25t,
    с каждой стороны у продольных связей. Ширина панели поперёк: a при продольной системе,
    b пан. при поперечной."""
    strip = min(0.25 * el.a, 25.0 * el.t)
    across = el.a if el.system == SYS_LONG else (el.b_panel or el.a)
    if across <= 0:
        return 1.0
    return min(1.0, 2.0 * strip / across)


@dataclass
class PlateReduction:
    phi: float          # редукционный коэффициент пластины
    phi_eff: float      # с учётом нередуцируемых полос
    sigma: float        # напряжение в жёстких связях на уровне ЦТ пластины, МПа (+ растяжение)
    sigma_cr: Optional[float]
    rule: str


def reduce_plate(el: Element, sigma: float) -> PlateReduction:
    """Редукционный коэффициент гибкой пластины при напряжении в жёстких связях sigma
    (знак: + растяжение, − сжатие)."""
    compressed = sigma < 0
    fr = rigid_fraction(el)
    if el.system == SYS_LONG:
        if not compressed:
            return PlateReduction(1.0, 1.0, sigma, None, "растянута, продольная система — не редуцируется (2.2.37.3)")
        scr = sigma_cr_plate(el.t, el.a, el.ReH)
        phi = min(1.0, scr / abs(sigma)) if abs(sigma) > 1e-9 else 1.0
        rule = "φ = σкр/|σж| (2.2.38)"
    else:
        scr = None
        phi = phi_table_2239(el.t, compressed)
        rule = "табл. 2.2.39, " + ("сжатие" if compressed else "растяжение")
        if compressed:
            lim = phi_limit_2239(el.t, el.a, el.b_panel or el.a, abs(sigma))
            if lim < phi:
                phi, rule = lim, "ограничение 2.2.39-2"
        phi = min(phi, 1.0)
    return PlateReduction(phi, fr + (1.0 - fr) * phi, sigma, scr, rule)


# ----------------------------------------------------------------------------
# Расчёт сечения
# ----------------------------------------------------------------------------
@dataclass
class SectionProps:
    A: float            # м² (полное сечение)
    S: float            # м³ относительно ОП
    z0: float           # м
    I: float            # м⁴ относительно нейтральной оси
    z_top: float
    z_bot: float

    @property
    def W_deck(self) -> float:
        return self.I / (self.z_top - self.z0)

    @property
    def W_bottom(self) -> float:
        return self.I / (self.z0 - self.z_bot)


def section_props(els: list[Element], phis: Optional[list[float]] = None, half: bool = True) -> SectionProps:
    """Характеристики сечения при редукционных коэффициентах phis (по умолчанию все 1)."""
    if not els:
        raise BeamError("Таблица связей пуста.")
    phis = phis or [1.0] * len(els)
    k = 2.0 if half else 1.0
    A = S = J = 0.0
    for el, ph in zip(els, phis):
        a = el.area() * ph
        zc = el.centroid()
        A += a
        S += a * zc
        J += a * zc ** 2 + el.own_inertia() * ph
    if A <= 0:
        raise BeamError("Площадь сечения равна нулю — проверьте таблицу.")
    z0 = S / A
    I = J - A * z0 ** 2
    lows, highs = zip(*(el.z_range() for el in els))
    return SectionProps(A * k, S * k, z0, I * k, max(highs), min(lows))


def stress(M: float, z: float, sp: SectionProps) -> float:
    """2.2.40-1 со знаком: M > 0 — перегиб (палуба растянута). МПа."""
    return M * (z - sp.z0) * 1e-3 / sp.I


@dataclass
class Approximation:
    no: int
    props: SectionProps
    phis: list[float]
    sigma_deck: float
    sigma_bottom: float
    diff: Optional[float]           # макс. относительная разница с предыдущим, доли


@dataclass
class CaseResult:
    case: str                       # SAG / HOG
    M: float                        # кН·м со знаком (+ перегиб)
    approximations: list[Approximation]
    reductions: dict[int, PlateReduction]
    converged: bool

    @property
    def final(self) -> Approximation:
        return self.approximations[-1]


@dataclass
class ElementCheck:
    index: int
    name: str
    case: str
    z: float                # волокно с наибольшим |σ|
    sigma: float            # МПа, + растяжение
    allowable: float        # МПа
    factor: float
    sigma0: float

    @property
    def usage(self) -> float:
        return abs(self.sigma) / self.allowable if self.allowable > 0 else float("inf")

    @property
    def ok(self) -> bool:
        return self.usage <= 1.0 + 1e-9


@dataclass
class BeamResult:
    elements: list[Element]
    half: bool
    zero: SectionProps
    cases: list[CaseResult] = field(default_factory=list)
    checks: list[ElementCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def I0(self) -> float:
        return self.zero.I

    @property
    def worst(self) -> Optional[ElementCheck]:
        return max(self.checks, key=lambda c: c.usage) if self.checks else None


def run_case(els: list[Element], M: float, case: str, half: bool) -> CaseResult:
    """Последовательные приближения для одного случая изгиба (2.2.36)."""
    phis = [1.0] * len(els)
    sp = section_props(els, phis, half)
    zt, zb = sp.z_top, sp.z_bot
    apps = [Approximation(0, sp, phis[:], stress(M, zt, sp), stress(M, zb, sp), None)]
    reductions: dict[int, PlateReduction] = {}
    flexible = [i for i, e in enumerate(els) if e.flexible and e.is_plate]
    converged = not flexible
    for it in range(1, MAX_ITER + 1):
        if not flexible:
            break
        prev = apps[-1]
        new = phis[:]
        for i in flexible:
            red = reduce_plate(els[i], stress(M, els[i].centroid(), prev.props))
            reductions[i] = red
            new[i] = red.phi_eff
        phis = new
        sp = section_props(els, phis, half)
        sd, sb = stress(M, zt, sp), stress(M, zb, sp)
        diff = max(abs(sd - prev.sigma_deck) / max(abs(sd), 1e-9),
                   abs(sb - prev.sigma_bottom) / max(abs(sb), 1e-9))
        apps.append(Approximation(it, sp, phis[:], sd, sb, diff))
        if diff <= CONVERGENCE:
            converged = True
            break
    return CaseResult(case, M, apps, reductions, converged)


def check_elements(els: list[Element], cr: CaseResult, basin_value: str) -> list[ElementCheck]:
    """Напряжения в крайних волокнах каждой связи и сравнение с допускаемыми (2.2.67–2.2.68)."""
    sp = cr.final.props
    out = []
    for i, el in enumerate(els):
        lo, hi = el.z_range()
        s_lo, s_hi = stress(cr.M, lo, sp), stress(cr.M, hi, sp)
        z, s = (lo, s_lo) if abs(s_lo) >= abs(s_hi) else (hi, s_hi)
        f = allowable_factor(basin_value, el.local_load)
        s0 = sigma_0(el.ReH)
        out.append(ElementCheck(i, el.name or el.kind, cr.case, z, s, f * s0, f, s0))
    return out


def calculate_beam(els: list[Element], half: bool = True, M_sag: Optional[float] = None,
                   M_hog: Optional[float] = None, basin_value: str = "О") -> BeamResult:
    """Полный расчёт. M_sag, M_hog — абсолютные значения расчётного момента, кН·м
    (None — проверка напряжений не выполняется)."""
    errs = [m for i, e in enumerate(els, 1) for m in e.validate(i)]
    if errs:
        raise BeamError("\n".join(errs))
    zero = section_props(els, None, half)
    if zero.I <= 0:
        raise BeamError("Момент инерции получился неположительным — проверьте отстояния z.")
    res = BeamResult(els, half, zero)
    for case, M in ((SAG, M_sag), (HOG, M_hog)):
        if M is None or M == 0:
            continue
        signed = -abs(M) if case == SAG else abs(M)
        cr = run_case(els, signed, case, half)
        res.cases.append(cr)
        if not cr.converged:
            res.warnings.append(f"{case}: приближения не сошлись за {MAX_ITER} шагов — проверьте гибкие пластины.")
        res.checks.extend(check_elements(els, cr, basin_value))
    if not any(e.flexible for e in els):
        res.warnings.append("Нет гибких пластин: редуцирование (2.2.36) не выполнялось, "
                            "все связи учтены полностью.")
    return res


# ----------------------------------------------------------------------------
# Пример — грузовое судно, полусечение (числа условные, для знакомства с таблицей)
# ----------------------------------------------------------------------------
def example_records() -> list[dict]:
    def row(**kw):
        r = dict(DEFAULT_ROW)
        r.update(kw)
        return r
    H = 5.0
    return [
        row(**{"Связь": "Палубный стрингер", "Тип": T_HPLATE, "b, мм": 1200, "t, мм": 12, "z, м": H, "y, м": 7.65}),
        row(**{"Связь": "Настил палубы", "Тип": T_HPLATE, "b, мм": 2000, "t, мм": 10, "z, м": H, "y, м": 6.05,
               "Гибкая": True, "Система": SYS_LONG, "a, мм": 550, "b пан., мм": 2000}),
        row(**{"Связь": "Продольный комингс", "Тип": T_VPLATE, "b, мм": 1200, "t, мм": 12, "z, м": H + 0.6, "y, м": 5.05}),
        row(**{"Связь": "Палубные балки", "Тип": T_FLAT, "n": 4, "b, мм": 160, "t, мм": 10, "z, м": H, "y, м": 6.2,
               "Вниз": True}),
        row(**{"Связь": "Ширстрек", "Тип": T_VPLATE, "b, мм": 1500, "t, мм": 12, "z, м": H - 0.75, "y, м": 8.25}),
        row(**{"Связь": "Бортовая обшивка", "Тип": T_VPLATE, "b, мм": 2950, "t, мм": 9, "z, м": 2.0, "y, м": 8.25,
               "Гибкая": True, "Система": SYS_TRANS, "a, мм": 550, "b пан., мм": 2950}),
        row(**{"Связь": "Скуловой лист", "Тип": T_IPLATE, "b, мм": 900, "t, мм": 10, "θ, °": 45, "z, м": 0.32,
               "y, м": 7.95}),
        row(**{"Связь": "Днищевая обшивка", "Тип": T_HPLATE, "b, мм": 7600, "t, мм": 10, "z, м": 0.0, "y, м": 3.8,
               "Гибкая": True, "Система": SYS_LONG, "a, мм": 550, "b пан., мм": 2200, "Местная нагрузка": True}),
        row(**{"Связь": "Днищевые балки", "Тип": T_TEE, "n": 12, "b, мм": 200, "t, мм": 10, "bп, мм": 80,
               "tп, мм": 10, "z, м": 0.0, "y, м": 3.8, "Местная нагрузка": True}),
        row(**{"Связь": "Вертикальный киль (½)", "Тип": T_VPLATE, "b, мм": 800, "t, мм": 6, "z, м": 0.4, "y, м": 0.0,
               "Местная нагрузка": True}),
    ]
