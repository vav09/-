"""
pkps_steps.py — «ход решения»: каждая формула п. 2.2.10 в общем виде,
с подстановкой чисел и результатом.

Модуль не считает ничего нового: все значения берутся из результата
pkps_core.calculate(). Задача — показать расчёт так, как его пишут в
пояснительной записке. Формулы возвращаются в LaTeX (для st.latex).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pkps_core as core


# --------------------------------------------------------------------------
# Форматирование чисел и конструкторы LaTeX (десятичная запятая)
# --------------------------------------------------------------------------
def tn(v: Optional[float], nd: int = 3) -> str:
    """Число для LaTeX: 1234.5 → '1\\,234{,}500'."""
    if v is None:
        return r"\text{—}"
    s = f"{v:,.{nd}f}"
    return s.replace(",", r"\,").replace(".", "{,}")


def tv(v: Optional[float]) -> str:
    """Число с автоматическим числом знаков (для моментов и сил)."""
    if v is None:
        return tn(None)
    a = abs(v)
    nd = 0 if a >= 1000 else 1 if a >= 100 else 2 if a >= 10 else 3
    return tn(v, nd)


def frac(a: str, b: str) -> str:
    return r"\dfrac{" + a + "}{" + b + "}"


def sqrt_(a: str) -> str:
    return r"\sqrt{" + a + "}"


def par(a: str) -> str:
    return r"\left(" + a + r"\right)"


_CYR = "А-Яа-яЁё"
_RE_BRACED = re.compile(r"_\{([" + _CYR + r"]+)\}")
_RE_SINGLE = re.compile(r"_([" + _CYR + r"])")


def cyr_text(tex: str) -> str:
    """Кириллические индексы заворачивает в \\text{}: k_в → k_{\\text{в}}.
    Так формулы одинаково рисуются любым движком LaTeX."""
    tex = _RE_BRACED.sub(lambda m: "_{\\text{" + m.group(1) + "}}", tex)
    tex = _RE_SINGLE.sub(lambda m: "_{\\text{" + m.group(1) + "}}", tex)
    return tex


def comma(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _cmp(label: str, formula: float, table: Optional[float]) -> str:
    """Строка сверки значения по формуле со значением по таблице Правил."""
    if table is None:
        return f"{label}: значение вне диапазона таблицы — сверка по таблице не выполняется."
    pct = 100.0 * (formula - table) / table if table else 0.0
    return (f"{label}: по формуле {comma(formula)} · по таблице {comma(table)} · "
            f"разница {comma(pct, 2)} %")


# --------------------------------------------------------------------------
# Структуры
# --------------------------------------------------------------------------
@dataclass
class Step:
    name: str                       # что считаем
    ref: str                        # пункт / формула ПКПС
    general: str = ""               # формула в общем виде (LaTeX)
    calc: str = ""                  # «= подстановка = результат» (LaTeX)
    value: Optional[float] = None   # числовой результат (для проверки и тестов)
    unit: str = ""
    check: str = ""                 # сверка с таблицей Правил (обычный текст)
    note: str = ""                  # пояснение (обычный текст)

    def __post_init__(self) -> None:
        self.general = cyr_text(self.general)
        self.calc = cyr_text(self.calc)


@dataclass
class Section:
    title: str
    steps: list[Step] = field(default_factory=list)
    note: str = ""
    warning: str = ""


# --------------------------------------------------------------------------
# Сборка решения
# --------------------------------------------------------------------------
def build_solution(res: core.WaveMomentResult, plateau: float = 0.5) -> list[Section]:
    inp, p = res.inp, res.params
    L, B, T, Tn, cb = inp.L, inp.B, inp.T, inp.Tn, inp.Cb
    v, D = inp.V0_kmh, res.D
    h, eta, eps = p.h, p.eta, p.eps
    N = tn
    cdot = r"\cdot "
    sections: list[Section] = []

    # ---- 0. исходные данные -------------------------------------------------
    d_note = "задано" if inp.D is not None else "оценено по Cb: D = 9,81·ρ·Cb·L·B·T"
    s0 = Section("Исходные данные")
    rows = [
        ("Класс судна", r"\text{«" + inp.basin.value + r"»}", "", None),
        ("Тип судна", r"\text{" + inp.ship_type.value + "}", "", None),
        ("Длина", "L = " + N(L, 2) + r"\ \text{м}", "", L),
        ("Ширина", "B = " + N(B, 2) + r"\ \text{м}", "", B),
        ("Осадка", "T = " + N(T, 2) + r"\ \text{м}", "", T),
        ("Осадка носом", "T_н = " + N(Tn, 2) + r"\ \text{м}", "", Tn),
        ("Коэффициент полноты", r"C_b = \delta = " + N(cb, 3), "", cb),
        ("Скорость хода", r"V_0 = v_{тв} = " + N(v, 2) + r"\ \text{км/ч}", "", v),
        ("Момент инерции миделя", "I = " + N(inp.I, 3) + r"\ \text{м}^4", "", inp.I),
        ("Водоизмещение", "D = " + N(D, 0) + r"\ \text{кН}", d_note, D),
    ]
    for name, tex, note, val in rows:
        s0.steps.append(Step(name, "исходные данные", general=tex, value=val, note=note))
    sections.append(s0)

    # ---- 1. параметры класса -------------------------------------------------
    s1 = Section("Параметры класса судна")
    s1.steps = [
        Step("Расчётная высота волны", "табл. 2.1.2", "h = " + N(h, 1) + r"\ \text{м}", value=h, unit="м"),
        Step("Коэффициент ε", "табл. 2.2.10-1", r"\varepsilon = " + N(eps, 3), value=eps),
        Step("Коэффициент η", "табл. 2.2.10-1", r"\eta = " + N(eta, 3), value=eta),
        Step("Частота волны", "табл. 2.2.10-1", r"\omega_{ср} = " + N(p.omega_sr, 2) + r"\ \text{с}^{-1}",
             value=p.omega_sr, unit="с⁻¹"),
        Step("Параметр v₁", "табл. 2.2.10-1", "v_1 = " + N(p.v1, 2) + r"\ \text{м/с}", value=p.v1, unit="м/с"),
    ]
    sections.append(s1)

    # ---- 2. волновой момент Mв -----------------------------------------------------
    s2 = Section("Волновой изгибающий момент Mв",
                 note="Значения L, B, T и δ принимаются при посадке, соответствующей расчётному "
                      "случаю нагрузки для момента на тихой воде.")
    s2.steps.append(Step(
        "Параметр a (для табл. 2.2.10-3)", "2.2.10-6",
        r"a = " + frac(r"3T\delta", r"2\delta+1"),
        "= " + frac("3" + cdot + N(T, 2) + cdot + N(cb, 3), "2" + cdot + N(cb, 3) + "+1")
        + " = " + N(res.a, 3) + r"\ \text{м}", res.a, "м"))
    s2.steps.append(Step(
        "Параметр b (для табл. 2.2.10-4)", "2.2.10-7",
        r"b = \delta B", "= " + N(cb, 3) + cdot + N(B, 2) + " = " + N(res.b, 3) + r"\ \text{м}", res.b, "м"))
    s2.steps.append(Step(
        "Коэффициент kδ", "2.2.10-3",
        r"k_\delta = \exp\left[-1{,}6\,(1-\delta)\right]",
        r"= \exp\left[-1{,}6" + cdot + "(1-" + N(cb, 3) + r")\right] = " + N(res.k_delta, 4),
        res.k_delta, check=_cmp("табл. 2.2.10-2", res.k_delta, core.table_k_delta(cb))))
    kt_arg = frac(r"1{,}14" + cdot + N(T, 2) + cdot + N(cb, 3),
                  N(eta, 3) + cdot + N(h, 1) + cdot + "(2" + cdot + N(cb, 3) + "+1)")
    s2.steps.append(Step(
        "Коэффициент kт", "2.2.10-4",
        r"k_т = \exp\left\{-" + frac(r"1{,}14\,T\delta", r"\eta\, h\,(2\delta+1)") + r"\right\}",
        r"= \exp\left\{-" + kt_arg + r"\right\} = " + N(res.k_t, 4),
        res.k_t, check=_cmp("табл. 2.2.10-3", res.k_t, core.table_k_t(inp.basin, res.a))))
    x_num = r"0{,}19" + cdot + N(cb, 3) + cdot + N(B, 2)
    eh = N(eta, 3) + cdot + N(h, 1)
    s2.steps.append(Step(
        "Коэффициент kв", "2.2.10-5",
        r"k_в = \left\{1-\exp\left[-" + frac(r"0{,}19\,\delta B", r"\eta h") + r"\right]\right\}"
        + frac(r"\eta h", r"0{,}19\,\delta B"),
        r"= \left\{1-\exp\left[-" + frac(x_num, eh) + r"\right]\right\}" + frac(eh, x_num)
        + " = " + N(res.k_b, 4),
        res.k_b, check=_cmp("табл. 2.2.10-4", res.k_b, core.table_k_b(inp.basin, res.b))))
    s2.steps.append(Step(
        "Волновой изгибающий момент", "2.2.10-2",
        r"M_в = 0{,}255\,\varepsilon\, k_\delta k_т k_в\, B L^2 h",
        "= 0{,}255" + cdot + N(eps, 3) + cdot + N(res.k_delta, 4) + cdot + N(res.k_t, 4) + cdot
        + N(res.k_b, 4) + cdot + N(B, 2) + cdot + N(L, 2) + "^2" + cdot + N(h, 1)
        + " = " + tv(res.Mv) + r"\ \text{кН·м}", res.Mv, "кН·м"))
    sections.append(s2)

    # ---- 3. вибрация: kр ----------------------------------------------------------------
    s3 = Section("Влияние волновой вибрации kр")
    hL = frac(N(h, 1), N(L, 2))
    ks_e4 = res.ks / 1e4
    s3.steps.append(Step(
        "Коэффициент kv", "2.2.10-12",
        r"k_v = 1-19{,}9\," + frac("h", "L") + r"+143\left(" + frac("h", "L") + r"\right)^2",
        "= 1-19{,}9" + cdot + hL + "+143" + cdot + par(hL) + "^2 = " + N(res.kv, 4), res.kv))
    s3.steps.append(Step(
        "Частота волны с движущегося судна", "2.2.10-9",
        r"\omega_к = \omega_{ср} + 1{,}92\,k_v\," + frac("v_{тв}", "L"),
        "= " + N(p.omega_sr, 2) + " + 1{,}92" + cdot + N(res.kv, 4) + cdot + frac(N(v, 2), N(L, 2))
        + " = " + N(res.omega_k, 4) + r"\ \text{с}^{-1}", res.omega_k, "с⁻¹",
        note="Скорость v_тв входит в формулу в км/ч, как в Правилах."))
    sig_den = (r"\left[1{,}2+" + frac(N(B, 2), "3" + cdot + N(T, 2)) + r"\right]" + cdot + N(D, 0) + cdot
               + N(L, 2) + "^3")
    s3.steps.append(Step(
        "Частота собственных колебаний корпуса", "2.2.10-10",
        r"\sigma = k_s\sqrt{" + frac("I", r"\left[1{,}2+" + frac("B", "3T") + r"\right] D L^3") + "}",
        "= " + N(ks_e4, 0) + r"\cdot 10^4" + cdot + sqrt_(frac(N(inp.I, 3), sig_den))
        + " = " + N(res.sigma, 4) + r"\ \text{с}^{-1}", res.sigma, "с⁻¹",
        note=f"k_s = {ks_e4:.0f}·10⁴ для типа «{inp.ship_type.value}»."))
    sg = N(res.sigma, 4)
    s3.steps.append(Step(
        "Коэффициент kμ", "2.2.10-11",
        r"k_\mu = 0{,}0612\,(1-0{,}047\sigma-0{,}0077\sigma^2)\ \ge\ 0",
        "= 0{,}0612" + cdot + "(1-0{,}047" + cdot + sg + "-0{,}0077" + cdot + sg + "^2) = " + N(res.k_mu, 5),
        res.k_mu))
    r = res.ratio
    s3.steps.append(Step(
        "Отношение частот", "—",
        frac(r"\omega_к", r"\sigma"), "= " + frac(N(res.omega_k, 4), sg) + " = " + N(r, 4), r))

    if res.psi is None:
        s3.warning = (f"ωк/σ = {comma(r)} > 1,4: Правила не определяют коэффициент ψ "
                      f"(формулы 2.2.10-13, 2.2.10-14), поэтому kр, Mдв и Nдв не рассчитаны. "
                      f"Проверьте I, D и скорость; при необходимости обратитесь в РКО.")
    else:
        if r <= 1.0:
            s3.steps.append(Step(
                "Коэффициент ψ", "2.2.10-13 (0 ≤ ωк/σ ≤ 1,0)",
                r"\psi = \cos\left[1{,}3\left(" + frac(r"\omega_к", r"\sigma") + r"\right)^2\right]",
                r"= \cos\left[1{,}3" + cdot + r"\left(" + frac(N(res.omega_k, 4), N(res.sigma, 4))
                + r"\right)^2\right] = \cos\left[1{,}3" + cdot + N(r, 4) + r"^2\right] = " + N(res.psi, 4),
                res.psi))
        else:
            s3.steps.append(Step(
                "Коэффициент ψ", "2.2.10-14 (1,0 < ωк/σ ≤ 1,4)",
                r"\psi = 2{,}167\," + frac(r"\omega_к", r"\sigma") + "-1{,}9",
                "= 2{,}167" + cdot + frac(N(res.omega_k, 4), N(res.sigma, 4)) + "-1{,}9 = 2{,}167" + cdot
                + N(r, 4) + "-1{,}9 = " + N(res.psi, 4), res.psi))
        rr = N(r, 4)
        wk_s, sg_s = N(res.omega_k, 4), N(res.sigma, 4)
        s3.steps.append(Step(
            "Коэффициент влияния вибрации", "2.2.10-8",
            r"k_р = \psi\left\{1+" + frac(r"\omega_к^2", r"\sigma^2\sqrt{\left(1-\omega_к^2/\sigma^2\right)^2"
                                          r"+\left(2k_\mu\omega_к/\sigma\right)^2}") + r"\right\}",
            "= " + N(res.psi, 4) + cdot + r"\left\{1+"
            + frac(wk_s + "^2", sg_s + "^2" + cdot + sqrt_(
                r"\left(1-" + wk_s + "^2/" + sg_s + r"^2\right)^2+\left(2" + cdot + N(res.k_mu, 5) + cdot
                + wk_s + "/" + sg_s + r"\right)^2"))
            + r"\right\} = " + N(res.kp, 4), res.kp,
            note="ωк = " + comma(res.omega_k, 4) + " с⁻¹ и σ = " + comma(res.sigma, 4)
                 + " с⁻¹ — из формул 2.2.10-9 и 2.2.10-10."))
    sections.append(s3)

    # ---- 4. ударный момент Mу ---------------------------------------------------------
    s4 = Section("Ударный изгибающий момент Mу",
                 note="Возникает при малой осадке носом: если Tн ≥ 1,5·Tн⁰, удар волн в носовую "
                      "оконечность не учитывается.")
    kv_s = N(res.kv, 4)
    vsq = frac(N(v, 2), sqrt_(N(L, 2)))
    s4.steps.append(Step(
        "«Пороговая» осадка носом", "2.2.10-17",
        r"T_н^0 = \left(0{,}68+0{,}21\,k_v" + frac("v_{тв}", r"\sqrt{L}") + r"\right)\eta\, h",
        r"= \left(0{,}68+0{,}21" + cdot + kv_s + cdot + vsq + r"\right)" + cdot + N(eta, 3) + cdot + N(h, 1)
        + " = " + N(res.Tn0, 4) + r"\ \text{м}", res.Tn0, "м"))
    t0 = res.Tn0
    if Tn <= t0:
        case_tex = "T_н = " + N(Tn, 2) + r" \le T_н^0 = " + N(t0, 3) + r"\ \Rightarrow\ \varphi_1 = 1"
    elif Tn < 1.5 * t0:
        case_tex = ("T_н^0 < T_н = " + N(Tn, 2) + " < 1{,}5T_н^0 = " + N(1.5 * t0, 3) + r"\ \Rightarrow\ "
                    + r"\varphi_1 = 3-" + frac("2T_н", "T_н^0") + "=3-" + frac("2" + cdot + N(Tn, 2), N(t0, 3)))
    else:
        case_tex = ("T_н = " + N(Tn, 2) + r" \ge 1{,}5T_н^0 = " + N(1.5 * t0, 3) + r"\ \Rightarrow\ \varphi_1 = 0")
    s4.steps.append(Step(
        "Коэффициент φ₁", "к 2.2.10-16",
        r"\varphi_1 = \begin{cases}1, & T_н\le T_н^0\\ 3-2T_н/T_н^0, & T_н^0<T_н<1{,}5T_н^0\\"
        r" 0, & T_н\ge 1{,}5T_н^0\end{cases}",
        case_tex + " = " + N(res.phi1, 4), res.phi1))
    b0s = N(res.b0, 4)
    s4.steps.append(Step(
        "Параметр b₀", "2.2.10-19",
        r"b_0 = 4{,}32\sqrt{\delta\," + frac("B", "L") + r"\," + frac("T", "L") + "}",
        "= 4{,}32" + cdot + sqrt_(N(cb, 3) + cdot + frac(N(B, 2), N(L, 2)) + cdot + frac(N(T, 2), N(L, 2)))
        + " = " + b0s, res.b0))
    s4.steps.append(Step(
        "Коэффициент φ₀", "2.2.10-18",
        r"\varphi_0 = 1-1{,}03\,b_0+b_0^2-0{,}417\,b_0^3",
        "= 1-1{,}03" + cdot + b0s + "+" + b0s + "^2-0{,}417" + cdot + b0s + "^3 = " + N(res.phi0, 4), res.phi0))
    s4.steps.append(Step(
        "Скорость v₀ (по Правилам)", "2.2.10-20",
        r"v_0 = \left(0{,}336+0{,}104\,k_v" + frac("v_{тв}", r"\sqrt{L}") + r"\right)v_1+0{,}024\,k_v v_{тв}",
        r"= \left(0{,}336+0{,}104" + cdot + kv_s + cdot + vsq + r"\right)" + cdot + N(p.v1, 2)
        + "+0{,}024" + cdot + kv_s + cdot + N(v, 2) + " = " + N(res.v0, 4) + r"\ \text{м/с}",
        res.v0, "м/с", note="Не путать с V₀ — скоростью хода, которую вы вводите."))
    s4.steps.append(Step(
        "Коэффициент kу", "2.2.10-16",
        r"k_у = 5{,}3\cdot 10^{-4}\,\varphi_0\,\sigma\, v_0",
        r"= 5{,}3\cdot 10^{-4}" + cdot + N(res.phi0, 4) + cdot + sg + cdot + N(res.v0, 4)
        + " = " + N(res.ky, 5), res.ky))
    s4.steps.append(Step(
        "Ударный изгибающий момент", "2.2.10-15",
        r"M_у = k_у\,\varphi_1\, D\, L",
        "= " + N(res.ky, 5) + cdot + N(res.phi1, 4) + cdot + N(D, 0) + cdot + N(L, 2)
        + " = " + tv(res.My) + r"\ \text{кН·м}", res.My, "кН·м"))
    sections.append(s4)

    # ---- 5. итог ---------------------------------------------------------------------------
    s5 = Section("Итог")
    if res.Mdv is None:
        s5.warning = "Mдв и Nдв не определены: см. предупреждение в разделе «Влияние волновой вибрации»."
    else:
        s5.steps.append(Step(
            "Дополнительный волновой изгибающий момент", "2.2.10-1",
            r"M_{дв} = \pm\left(k_р M_в + M_у\right)",
            r"= \pm\left(" + N(res.kp, 4) + cdot + tv(res.Mv) + " + " + tv(res.My) + r"\right) = \pm "
            + tv(res.Mdv) + r"\ \text{кН·м}", res.Mdv, "кН·м"))
        s5.steps.append(Step(
            "Дополнительная волновая перерезывающая сила", "2.2.10-21",
            r"N_{дв} = " + frac("4 M_{дв}", "L"),
            "= " + frac("4" + cdot + tv(res.Mdv), N(L, 2)) + r" = \pm " + tv(res.Ndv) + r"\ \text{кН}",
            res.Ndv, "кН"))
        edge = (1.0 - plateau) / 2.0
        ndv = f"{res.Ndv:,.0f}".replace(",", " ")
        s5.note = (f"Эпюра Mдв постоянна на участке {comma(plateau, 2)}L (от {comma(edge * L, 1)} до "
                   f"{comma((1 - edge) * L, 1)} м от кормы) и уменьшается к оконечностям до нуля по "
                   f"линейному закону. Эпюра Nдв строится по рис. 2.2.10: максимум {ndv} кН, "
                   f"в средней части 0,7·Nдв.")
    sections.append(s5)
    return sections


# --------------------------------------------------------------------------
# Экспорт в текст (Markdown с формулами LaTeX)
# --------------------------------------------------------------------------
def solution_markdown(sections: list[Section], title: str = "Ход решения") -> str:
    """Решение одним Markdown-файлом: формулы в $$…$$ (открывается в VS Code, Obsidian, Typora, GitHub)."""
    out = [f"# {title}", "", "ПКПС РКО, часть I, п. 2.2.10", ""]
    for sec in sections:
        out += [f"## {sec.title}", ""]
        if sec.note:
            out += [f"_{sec.note}_", ""]
        for st in sec.steps:
            out.append(f"**{st.name}** ({st.ref})")
            out.append("")
            if st.general or st.calc:
                out += [f"$$ {st.general} {st.calc} $$", ""]
            if st.check:
                out += [f"_{st.check}_", ""]
            if st.note:
                out += [f"_{st.note}_", ""]
        if sec.warning:
            out += [f"> **Внимание.** {sec.warning}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Самопроверка LaTeX (используется в тестах)
# --------------------------------------------------------------------------
def latex_problems(tex: str) -> list[str]:
    """Находит типичные ошибки: несбалансированные скобки, \\left/\\right, \\begin/\\end,
    кириллицу вне \\text{}."""
    problems: list[str] = []
    clean = tex.replace(r"\{", "").replace(r"\}", "")
    depth = 0
    for ch in clean:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            problems.append("лишняя закрывающая скобка }")
            break
    if depth > 0:
        problems.append("не закрыта скобка {")
    if tex.count(r"\left") != tex.count(r"\right"):
        problems.append(r"число \left и \right не совпадает")
    if tex.count(r"\begin") != tex.count(r"\end"):
        problems.append(r"число \begin и \end не совпадает")
    # кириллица должна быть внутри \text{…}
    stripped = re.sub(r"\\text\{[^{}]*\}", "", tex)
    if re.search("[" + _CYR + "]", stripped):
        problems.append("кириллица вне \\text{}")
    if "nan" in tex.lower() or "inf" in tex.lower().replace(r"\infty", ""):
        problems.append("в формуле nan/inf")
    return problems
