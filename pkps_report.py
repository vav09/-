"""
pkps_report.py — отчёт о расчёте в PDF (reportlab) и Word (python-docx).

Отчёт собирается из ReportData; какие разделы войдут — решает набор ключей SECTIONS.
Формулы хода решения рисуются в PNG средствами matplotlib (mathtext), поэтому
в PDF и Word они выглядят одинаково и не требуют установленного LaTeX.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib
from matplotlib.figure import Figure

SECTIONS = {
    "inputs": "Исходные данные",
    "results": "Основные результаты",
    "steps": "Ход решения (формулы)",
    "epures": "Эпюры Mдв и Nдв",
    "model3d": "Вид судна (3D-модель)",
    "total": "Расчётный момент Mр",
    "beam": "Эквивалентный брус миделя",
    "protocol": "Протокол расчёта",
}
FORMATS = ("PDF", "Word")


@dataclass
class ReportData:
    title: str
    project: str
    inputs: list[tuple[str, str]]                       # (величина, значение)
    results: list[tuple[str, str]]
    warnings: list[str] = field(default_factory=list)
    sections: list = field(default_factory=list)        # pkps_steps.Section
    epure_png: list[bytes] = field(default_factory=list)
    model_png: Optional[bytes] = None
    total_png: Optional[bytes] = None
    total_rows: list[tuple[str, str]] = field(default_factory=list)
    beam_props: list[tuple[str, str]] = field(default_factory=list)
    beam_elements: list[list[str]] = field(default_factory=list)   # первая строка — заголовок
    beam_checks: list[list[str]] = field(default_factory=list)     # первая строка — заголовок
    beam_png: list[bytes] = field(default_factory=list)
    beam_notes: list[str] = field(default_factory=list)
    protocol: list[list[str]] = field(default_factory=list)        # первая строка — заголовок


# ----------------------------------------------------------------------------
# Формулы → PNG
# ----------------------------------------------------------------------------
_REPL = [(r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"), (r"\ge ", r"\geq "), (r"\le ", r"\leq ")]


def to_mathtext(tex: str) -> list[str]:
    """LaTeX из pkps_steps → одна или несколько строк, понятных matplotlib mathtext."""
    t = tex
    for a, b in _REPL:
        t = t.replace(a, b)
    t = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", t)
    t = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", t)
    m = re.search(r"\\begin\{cases\}(.*?)\\end\{cases\}", t, flags=re.S)
    if not m:
        return [t]
    head, tail = t[:m.start()], t[m.end():]
    rows = [r.replace("&", r"\quad ").strip() for r in m.group(1).split(r"\\") if r.strip()]
    out = []
    for i, r in enumerate(rows):
        out.append((head if i == 0 else r"\quad\quad ") + r + (tail if i == len(rows) - 1 else ""))
    return out


@lru_cache(maxsize=512)
def formula_png(tex: str, fontsize: int = 12, dpi: int = 200) -> Optional[bytes]:
    """PNG одной строки формулы; None, если mathtext её не понял (тогда в отчёт идёт текст)."""
    fig = Figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color="#111111")
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03,
                    facecolor="white")
    except Exception:
        return None
    return buf.getvalue()


def formula_images(tex: str) -> list[tuple[Optional[bytes], str]]:
    return [(formula_png(line), line) for line in to_mathtext(tex)]


def _png_size(png: bytes) -> tuple[int, int]:
    w = int.from_bytes(png[16:20], "big")
    h = int.from_bytes(png[20:24], "big")
    return w, h


# ----------------------------------------------------------------------------
# 3D-модель → PNG (статичная картинка для отчёта)
# ----------------------------------------------------------------------------
def hull_png(hull, z_stretch: float = 3.0, dpi: int = 150) -> bytes:
    """Статичный вид корпуса (pkps_hull.Hull3D) с ватерлинией."""
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = {"under": "#b5483a", "above": "#2f4f6f", "deck": "#c9b38a",
              "transom": "#2f4f6f", "house": "#e8ecef"}
    v = hull.verts.copy()
    v[:, 2] *= z_stretch
    fig = Figure(figsize=(9, 9), dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    light = np.array([-0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    for name, tris in hull.groups.items():
        if len(tris) == 0:
            continue
        pts = v[tris]
        n = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        norm[norm == 0] = 1
        shade = 0.55 + 0.45 * np.abs((n / norm) @ light)
        base = np.array(matplotlib.colors.to_rgb(colors.get(name, "#888888")))
        fc = np.clip(base[None, :] * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(pts, facecolors=fc, edgecolors=fc, linewidths=0.3))
    wl = hull.waterline.copy()
    wl[:, 2] *= z_stretch
    ax.plot(wl[:, 0], wl[:, 1], wl[:, 2], color="#1f5f8b", linewidth=1.2)
    lo, hi = v.min(axis=0), v.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), zoom=1.0)
    except TypeError:                                  # matplotlib < 3.6
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    # обрезаем белые поля вокруг судна
    from PIL import Image, ImageChops
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if box:
        pad = 12
        im = im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                      min(box[2] + pad, im.width), min(box[3] + pad, im.height)))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
def _font_paths() -> tuple[str, str]:
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return str(base / "DejaVuSans.ttf"), str(base / "DejaVuSans-Bold.ttf")


def _register_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if "DejaVu" in pdfmetrics.getRegisteredFontNames():
        return
    reg, bold = _font_paths()
    pdfmetrics.registerFont(TTFont("DejaVu", reg))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold))
    from reportlab.lib.fonts import addMapping
    addMapping("DejaVu", 0, 0, "DejaVu")
    addMapping("DejaVu", 1, 0, "DejaVu-Bold")
    addMapping("DejaVu", 0, 1, "DejaVu")
    addMapping("DejaVu", 1, 1, "DejaVu-Bold")


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_pdf(d: ReportData, sections: list[str]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    _register_fonts()
    body = ParagraphStyle("b", fontName="DejaVu", fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
    warn = ParagraphStyle("w", parent=body, textColor=colors.HexColor("#8a4b00"))
    h1 = ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16, leading=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5,
                        textColor=colors.HexColor("#1f3b5c"), keepWithNext=1)
    h3 = ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=2,
                        keepWithNext=1)
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)
    W = A4[0] - 30 * mm

    def table(rows: list[list[str]], widths=None, header=True):
        data = [[Paragraph(_esc(c), cell) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")),
              ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf4")))
        t.setStyle(TableStyle(st))
        return t

    def img(png: bytes, max_w=W, max_h=110 * mm, scale=1.0):
        w, h = _png_size(png)
        k = min(max_w / w, max_h / h) * scale
        return Image(io.BytesIO(png), width=w * k, height=h * k, hAlign="LEFT")

    def formula(tex: str):
        out = []
        for png, line in formula_images(tex):
            if png is None:
                out.append(Paragraph(_esc(line), small))
            else:
                w, h = _png_size(png)
                k = 72 / 200            # 200 dpi → пункты
                k = min(k, (W - 10) / w)
                out.append(Image(io.BytesIO(png), width=w * k, height=h * k, hAlign="LEFT"))
        return out

    story = [Paragraph(_esc(d.title), h1),
             Paragraph(_esc(f"Проект: {d.project} · {date.today().strftime('%d.%m.%Y')}"), small),
             Paragraph("Расчёт по ПКПС РКО, часть I, п. 2.2.10. Результаты справочные: соответствие "
                       "требованиям Правил подтверждает расчётчик и РКО.", small),
             Spacer(1, 6)]
    for key in sections:
        name = SECTIONS[key]
        if key == "inputs":
            story += [Paragraph(name, h2), table([["Величина", "Значение"]] + [list(r) for r in d.inputs],
                                                 [W * 0.6, W * 0.4])]
        elif key == "results":
            story += [Paragraph(name, h2), table([["Величина", "Значение"]] + [list(r) for r in d.results],
                                                 [W * 0.6, W * 0.4])]
            for w_ in d.warnings:
                story.append(Paragraph("⚠ " + _esc(w_), warn))
        elif key == "steps" and d.sections:
            story.append(Paragraph(name, h2))
            for sec in d.sections:
                story.append(Paragraph(_esc(sec.title), h3))
                if sec.note:
                    story.append(Paragraph(_esc(sec.note), small))
                for stp in sec.steps:
                    block = [Paragraph(f"<b>{_esc(stp.name)}</b> · {_esc(stp.ref)}", body)]
                    if stp.general:
                        block += formula(stp.general)
                    if stp.calc:
                        block += formula(stp.calc)
                    for extra in (stp.check, stp.note):
                        if extra:
                            block.append(Paragraph(_esc(extra), small))
                    block.append(Spacer(1, 3))
                    story.append(KeepTogether(block))
                if sec.warning:
                    story.append(Paragraph("⚠ " + _esc(sec.warning), warn))
        elif key == "epures" and d.epure_png:
            story.append(Paragraph(name, h2))
            story += [img(p) for p in d.epure_png]
        elif key == "model3d" and d.model_png:
            story += [Paragraph(name, h2), img(d.model_png),
                      Paragraph("Схематичная модель: размеры точные, обводы типовые, высота растянута ×3.", small)]
        elif key == "total" and (d.total_png or d.total_rows):
            story.append(Paragraph(name, h2))
            if d.total_png:
                story.append(img(d.total_png))
            if d.total_rows:
                story.append(table([["Величина", "Значение"]] + [list(r) for r in d.total_rows],
                                   [W * 0.6, W * 0.4]))
        elif key == "beam" and d.beam_props:
            story += [Paragraph(name, h2),
                      table([["Характеристика", "Значение"]] + [list(r) for r in d.beam_props], [W * 0.6, W * 0.4])]
            for n_ in d.beam_notes:
                story.append(Paragraph(_esc(n_), small))
            story += [img(p, scale=0.95) for p in d.beam_png]
            if d.beam_elements:
                story += [Paragraph("Связи эквивалентного бруса", h3), table(d.beam_elements)]
            if d.beam_checks:
                story += [Paragraph("Проверка нормальных напряжений (2.2.40, 2.2.67, табл. 2.2.68)", h3),
                          table(d.beam_checks)]
        elif key == "protocol" and d.protocol:
            story += [Paragraph(name, h2), table(d.protocol, [W * .12, W * .40, W * .18, W * .18, W * .12])]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(15 * mm, 10 * mm, f"{d.project} — дополнительный волновой момент, ПКПС РКО п. 2.2.10")
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"стр. {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=17 * mm, title=d.title, author=d.project)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Word
# ----------------------------------------------------------------------------
def build_docx(d: ReportData, sections: list[str]) -> bytes:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec0 = doc.sections[0]
    sec0.left_margin = sec0.right_margin = Cm(2)
    sec0.top_margin = sec0.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    usable = sec0.page_width - sec0.left_margin - sec0.right_margin

    def para(text: str, size=None, color=None, bold=False, italic=False):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        p.paragraph_format.space_after = Pt(3)
        return p

    def shade(cell, hex_color: str):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)

    def table(rows: list[list[str]], header=True):
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.LEFT
        small = len(rows[0]) > 5
        for i, r in enumerate(rows):
            for j, c in enumerate(r):
                cell = t.cell(i, j)
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(c))
                run.font.size = Pt(7.5 if small else 9)
                if header and i == 0:
                    run.bold = True
                    shade(cell, "E6EDF4")
        doc.add_paragraph()
        return t

    def picture(png: bytes, width=None):
        w, h = _png_size(png)
        width = width or usable
        doc.add_picture(io.BytesIO(png), width=min(width, int(usable)))

    def formula(tex: str):
        for png, line in formula_images(tex):
            if png is None:
                para(line, size=9, italic=True)
            else:
                w, _ = _png_size(png)
                width = min(int(Cm(2.54).emu * w / 200), int(usable))     # 200 dpi
                doc.add_picture(io.BytesIO(png), width=width)
                doc.paragraphs[-1].paragraph_format.space_after = Pt(2)

    doc.add_heading(d.title, level=0)
    para(f"Проект: {d.project} · {date.today().strftime('%d.%m.%Y')}", size=9, color="555555")
    para("Расчёт по ПКПС РКО, часть I, п. 2.2.10. Результаты справочные: соответствие требованиям "
         "Правил подтверждает расчётчик и РКО.", size=8.5, color="555555")

    for key in sections:
        name = SECTIONS[key]
        if key == "inputs":
            doc.add_heading(name, level=1)
            table([["Величина", "Значение"]] + [list(r) for r in d.inputs])
        elif key == "results":
            doc.add_heading(name, level=1)
            table([["Величина", "Значение"]] + [list(r) for r in d.results])
            for w_ in d.warnings:
                para("⚠ " + w_, size=9, color="8A4B00")
        elif key == "steps" and d.sections:
            doc.add_heading(name, level=1)
            for sec in d.sections:
                doc.add_heading(sec.title, level=2)
                if sec.note:
                    para(sec.note, size=8.5, color="555555")
                for stp in sec.steps:
                    p = doc.add_paragraph()
                    p.add_run(stp.name).bold = True
                    p.add_run(f" · {stp.ref}")
                    if stp.general:
                        formula(stp.general)
                    if stp.calc:
                        formula(stp.calc)
                    for extra in (stp.check, stp.note):
                        if extra:
                            para(extra, size=8.5, color="555555")
                if sec.warning:
                    para("⚠ " + sec.warning, size=9, color="8A4B00")
        elif key == "epures" and d.epure_png:
            doc.add_heading(name, level=1)
            for pic in d.epure_png:
                picture(pic)
        elif key == "model3d" and d.model_png:
            doc.add_heading(name, level=1)
            picture(d.model_png)
            para("Схематичная модель: размеры точные, обводы типовые, высота растянута ×3.", size=8.5,
                 color="555555")
        elif key == "total" and (d.total_png or d.total_rows):
            doc.add_heading(name, level=1)
            if d.total_png:
                picture(d.total_png)
            if d.total_rows:
                table([["Величина", "Значение"]] + [list(r) for r in d.total_rows])
        elif key == "beam" and d.beam_props:
            doc.add_heading(name, level=1)
            table([["Характеристика", "Значение"]] + [list(r) for r in d.beam_props])
            for n_ in d.beam_notes:
                para(n_, size=8.5, color="555555")
            for pic in d.beam_png:
                picture(pic)
            if d.beam_elements:
                doc.add_heading("Связи эквивалентного бруса", level=2)
                table(d.beam_elements)
            if d.beam_checks:
                doc.add_heading("Проверка нормальных напряжений (2.2.40, 2.2.67, табл. 2.2.68)", level=2)
                table(d.beam_checks)
        elif key == "protocol" and d.protocol:
            doc.add_heading(name, level=1)
            table(d.protocol)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
