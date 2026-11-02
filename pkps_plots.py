"""
pkps_plots.py — эпюры по результатам расчёта (matplotlib).

Все функции возвращают matplotlib.figure.Figure; Streamlit показывает их через
st.pyplot, а в скриптах их можно сохранить через fig.savefig("name.png", dpi=200).
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
from matplotlib.figure import Figure

ACCENT = "#1f5f8b"      # основная линия
ACCENT_SOFT = "#cfe0ee"
SECOND = "#b5483a"      # вторая линия (для Mр)
GRID = "#d9d9d9"
TEXT = "#222222"


def _num(v: float) -> str:
    """12345.6 -> '12 346' (неразрывный пробел между тысячами)."""
    return f"{v:,.0f}".replace(",", "\u00a0")


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=12, loc="left", color=TEXT)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.tick_params(colors=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _length_axis(ax, L: float) -> None:
    """Вторая шкала сверху — в метрах."""
    sec = ax.secondary_xaxis("top", functions=(lambda v: v * L, lambda m: m / L))
    sec.set_xlabel("м по длине судна", color=TEXT, fontsize=9)
    sec.tick_params(colors=TEXT, labelsize=8)


def fig_moment(x: np.ndarray, M: np.ndarray, L: float, Mdv: float,
               plateau: float, basin_label: str) -> Figure:
    """Эпюра Mдв (±) по длине судна."""
    fig = Figure(figsize=(8, 3.8), dpi=110)
    ax = fig.subplots()
    ax.fill_between(x, M, -M, color=ACCENT_SOFT, alpha=0.9)
    ax.plot(x, M, color=ACCENT, linewidth=2.0)
    ax.plot(x, -M, color=ACCENT, linewidth=2.0)
    edge = (1 - plateau) / 2
    ax.annotate(f"±{_num(Mdv)} кН·м",
                xy=(0.5, Mdv), xytext=(0.5, Mdv * 1.12), ha="center",
                fontsize=10, color=TEXT)
    ax.set_ylim(-Mdv * 1.3, Mdv * 1.3)
    for xv in (edge, 1 - edge):
        ax.axvline(xv, color="#888888", linestyle=":", linewidth=1)
    _style(ax, "x / L", "Mдв, кН·м",
           f"Эпюра дополнительного волнового момента, класс «{basin_label}»")
    ax.text(0.5, -Mdv * 1.22, f"постоянный участок {plateau:.2f}L",
            ha="center", fontsize=8, color="#555555")
    _length_axis(ax, L)
    fig.tight_layout()
    return fig


def fig_shear(x: np.ndarray, N: np.ndarray, L: float, Ndv: float) -> Figure:
    """Эпюра Nдв (±) по рис. 2.2.10."""
    fig = Figure(figsize=(8, 3.8), dpi=110)
    ax = fig.subplots()
    ax.fill_between(x, N, -N, color=ACCENT_SOFT, alpha=0.9)
    ax.plot(x, N, color=ACCENT, linewidth=2.0)
    ax.plot(x, -N, color=ACCENT, linewidth=2.0)
    ax.annotate(f"±{_num(Ndv)} кН",
                xy=(0.22, Ndv), xytext=(0.22, Ndv * 1.12), ha="center",
                fontsize=10, color=TEXT)
    ax.annotate(f"0,7·Nдв = {_num(0.7 * Ndv)} кН",
                xy=(0.5, 0.7 * Ndv), xytext=(0.5, 0.7 * Ndv * 1.12), ha="center",
                fontsize=9, color=TEXT)
    ax.set_ylim(-Ndv * 1.3, Ndv * 1.3)
    _style(ax, "x / L", "Nдв, кН",
           "Эпюра дополнительной волновой перерезывающей силы (рис. 2.2.10)")
    _length_axis(ax, L)
    fig.tight_layout()
    return fig


def fig_total(x: np.ndarray, mtv: np.ndarray, m_plus: np.ndarray,
              m_minus: np.ndarray, L: float) -> Figure:
    """Mтв и Mр = Mтв ± Mдв по длине (п. 2.2.12-1)."""
    fig = Figure(figsize=(8, 3.8), dpi=110)
    ax = fig.subplots()
    ax.plot(x, mtv, color="#666666", linewidth=1.5, linestyle="--", label="Mтв")
    ax.plot(x, m_plus, color=ACCENT, linewidth=2.0, label="Mр = Mтв + Mдв")
    ax.plot(x, m_minus, color=SECOND, linewidth=2.0, label="Mр = Mтв − Mдв")
    ax.fill_between(x, m_minus, m_plus, color=ACCENT_SOFT, alpha=0.6)
    top = max(float(np.max(m_plus)), float(np.max(m_minus)))
    bot = min(float(np.min(m_plus)), float(np.min(m_minus)))
    pad = 0.12 * max(abs(top), abs(bot), 1.0)
    ax.set_ylim(bot - pad, top + pad)
    _style(ax, "x / L", "M, кН·м", "Расчётный изгибающий момент Mр = Mтв ± Mдв")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=3)
    _length_axis(ax, L)
    fig.tight_layout()
    return fig


def fig_to_png(fig: Figure, dpi: int = 200) -> bytes:
    """PNG-байты для кнопки скачивания."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Эквивалентный брус: эскиз сечения и эпюра напряжений
# ----------------------------------------------------------------------------
def fig_section(beam, case_index: Optional[int] = None) -> Figure:
    """Эскиз полусечения (по столбцу y) с нейтральной осью и эпюрой σ(z) для выбранного случая.
    beam — pkps_beam.BeamResult."""
    import math
    import pkps_beam as bm
    from matplotlib.patches import Polygon, Rectangle

    has_case = case_index is not None and 0 <= case_index < len(beam.cases)
    fig = Figure(figsize=(9, 4.6) if has_case else (6.5, 4.6), dpi=110)
    if has_case:
        ax, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1]})
    else:
        ax, ax2 = fig.subplots(), None
    sp = beam.cases[case_index].final.props if has_case else beam.zero
    phis = beam.cases[case_index].final.phis if has_case else [1.0] * len(beam.elements)

    ys = [abs(e.y) for e in beam.elements] or [1.0]
    zs = [v for e in beam.elements for v in e.z_range()]
    span = max(max(ys), (max(zs) - min(zs)) * 0.3, 1.0)
    vis = span * 0.012                                     # минимальная видимая толщина
    for el, ph in zip(beam.elements, phis):
        col = ACCENT if ph >= 0.999 else "#d08a2c"
        t = max(el.t / 1000, vis)
        b = el.b / 1000
        y = el.y
        if el.kind == bm.T_HPLATE:
            patches = [Rectangle((y - b / 2, el.z - t / 2), b, t)]
        elif el.kind == bm.T_VPLATE:
            patches = [Rectangle((y - t / 2, el.z - b / 2), t, b)]
        elif el.kind == bm.T_IPLATE:
            th = math.radians(el.theta)
            dx, dz = b / 2 * math.cos(th), b / 2 * math.sin(th)
            nx, nz = -math.sin(th) * t / 2, math.cos(th) * t / 2
            patches = [Polygon([(y - dx + nx, el.z - dz + nz), (y + dx + nx, el.z + dz + nz),
                                (y + dx - nx, el.z + dz - nz), (y - dx - nx, el.z - dz - nz)])]
        elif el.kind in bm.PROFILES:
            sg = -1 if el.down else 1
            z1 = el.z + sg * b
            patches = [Rectangle((y - t / 2, min(el.z, z1)), t, b)]
            if el.kind != bm.T_FLAT and el.bf > 0:
                bf, tf = el.bf / 1000, max(el.tf / 1000, vis)
                x0 = y - bf / 2 if el.kind == bm.T_TEE else y - t / 2
                patches.append(Rectangle((x0, z1 if sg > 0 else z1 - tf), bf, tf))
        else:
            h = max(b, vis * 3)
            patches = [Rectangle((y - h / 4, el.z - h / 2), h / 2, h)]
        for pa in patches:
            pa.set_facecolor(col)
            pa.set_edgecolor(col)
            pa.set_alpha(0.85)
            ax.add_patch(pa)
        if el.n > 1:
            ax.text(y, el.z, f" ×{el.n}", fontsize=7, color=TEXT, va="center")
    ax.axhline(sp.z0, color=SECOND, linewidth=1.2, linestyle="--")
    ax.text(0.01, sp.z0, f" НО: z0 = {sp.z0:.3f} м".replace(".", ","), color=SECOND,
            fontsize=9, va="bottom", transform=ax.get_yaxis_transform())
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xlim(min(-0.05 * span, min(e.y for e in beam.elements) - span * 0.1), max(ys) + span * 0.15)
    ylim = (min(zs) - 0.08 * (max(zs) - min(zs) + 1), max(zs) + 0.08 * (max(zs) - min(zs) + 1))
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y — от ДП, м", color=TEXT)
    ax.set_ylabel("z — от ОП, м", color=TEXT)
    title = ("Полусечение" if beam.half else "Сечение") + (f", {beam.cases[case_index].case}" if has_case else ", φ = 1")
    ax.set_title(title, fontsize=10, loc="left", color=TEXT)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if ax2 is not None:
        cr = beam.cases[case_index]
        zz = np.linspace(sp.z_bot, sp.z_top, 50)
        sig = cr.M * (zz - sp.z0) * 1e-3 / sp.I
        ax2.fill_betweenx(zz, 0, sig, color=ACCENT_SOFT)
        ax2.plot(sig, zz, color=ACCENT, linewidth=2)
        ax2.axvline(0, color="#555555", linewidth=0.9)
        ax2.axhline(sp.z0, color=SECOND, linewidth=1.2, linestyle="--")
        for zv, sv in ((sp.z_top, sig[-1]), (sp.z_bot, sig[0])):
            ax2.annotate(f"{sv:+.1f} МПа".replace(".", ","), (sv, zv), fontsize=9, color=TEXT,
                         ha="center", va="bottom" if zv > sp.z0 else "top")
        lim = max(abs(sig).max() * 1.35, 1.0)
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(*ylim)
        ax2.set_xlabel("σ, МПа (+ растяжение)", color=TEXT)
        ax2.set_title("σ = Mр·z·10⁻³/I", fontsize=10, loc="left", color=TEXT)
        ax2.grid(True, color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
