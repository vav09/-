"""
app.py — веб-приложение «Дополнительный волновой изгибающий момент».
ПКПС РКО, часть I, п. 2.2.10 (+ эквивалентный брус миделя, п. 2.2.31–2.2.40, 2.2.67–2.2.68).

Запуск:   streamlit run app.py
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import pkps_beam as beam_mod
import pkps_core as core
import pkps_hull as hull_mod
import pkps_plots as plots
import pkps_project as project
import pkps_report as report
import pkps_steps as steps_mod
import pkps_theme
import pkps_view3d as view3d
from pkps_core import BasinClass, ShipType

st.set_page_config(page_title="Волновой момент — ПКПС РКО", page_icon="⚓", layout="wide")
pkps_theme.apply_theme()

ss = st.session_state
MTV_COLUMNS = ["x/L", "Mтв, кН·м"]
MTV_DEFAULT = [{"x/L": x, "Mтв, кН·м": m} for x, m in
               zip([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, None, None, None, 0.0])]
M_SOURCES = ["Mдв (Mтв = 0)", "Mтв ± Mдв из вкладки «Расчётный момент Mр»", "Ввести вручную"]


# ----------------------------------------------------------------------------
# Состояние: значения по умолчанию и обработчики (выполняются до отрисовки виджетов)
# ----------------------------------------------------------------------------
def init_state() -> None:
    # Значения полей ввода хранятся в ss["_vals"] и подставляются в виджеты как value=.
    # Ключи виджетов содержат номер версии: при открытии проекта номер растёт, и все поля
    # создаются заново с загруженными значениями. Внутри одной версии поле работает как
    # обычный виджет Streamlit — любое изменение сразу идёт в расчёт.
    ss.setdefault("_vals", dict(project.DEFAULTS))
    ss.setdefault("_ver", 0)
    ss.setdefault("beam_rows", beam_mod.example_records())
    ss.setdefault("mtv_rows", [dict(r) for r in MTV_DEFAULT])
    ss.setdefault("beam_ver", 0)
    ss.setdefault("mtv_ver", 0)


def K(name: str) -> str:
    """Ключ виджета поля name в текущей версии."""
    return f"in_{name}__{ss._ver}"


def cur(name: str):
    """Текущее значение поля (даже если виджет в этом прогоне ещё не нарисован)."""
    return ss.get(K(name), ss._vals[name])


def rebuild(values: dict) -> None:
    """Пересоздать все поля ввода с новыми значениями."""
    ss._vals = dict(values)
    ss._ver += 1


def beam_key() -> str:
    return f"beam_ed_{ss.beam_ver}"


def mtv_key() -> str:
    return f"mtv_ed_{ss.mtv_ver}"


def current_beam_rows() -> list[dict]:
    return project.apply_editor_state(ss.beam_rows, ss.get(beam_key()), beam_mod.COLUMNS,
                                      beam_mod.DEFAULT_ROW)


def current_mtv_rows() -> list[dict]:
    return project.apply_editor_state(ss.mtv_rows, ss.get(mtv_key()), MTV_COLUMNS,
                                      {"x/L": None, "Mтв, кН·м": None})


def set_tables(beam_rows: list[dict], mtv_rows: list[dict]) -> None:
    ss.beam_rows = beam_rows
    ss.mtv_rows = mtv_rows
    ss.beam_ver += 1                # новый ключ редактора — старые правки сбрасываются
    ss.mtv_ver += 1


def on_project_upload() -> None:
    f = ss.get("project_file")
    if f is None:
        return
    try:
        values, beam_rows, mtv_rows = project.load(f.getvalue())
    except project.ProjectError as exc:
        ss["_flash"] = ("error", f"Проект не открыт: {exc}")
        return
    rebuild(values)
    set_tables(beam_rows, mtv_rows or [dict(r) for r in MTV_DEFAULT])
    ss.pop("_report", None)
    ss["_flash"] = ("success", f"Проект «{values['project_name']}» открыт.")


def on_new_project() -> None:
    rebuild(project.DEFAULTS)
    set_tables(beam_mod.example_records(), [dict(r) for r in MTV_DEFAULT])
    ss.pop("_report", None)
    ss["_flash"] = ("info", "Создан новый проект со значениями по умолчанию.")


def on_beam_example() -> None:
    ss.beam_rows = beam_mod.example_records()
    ss.beam_ver += 1


def on_beam_clear() -> None:
    ss.beam_rows = []
    ss.beam_ver += 1


def on_use_beam_I() -> None:
    vals = {k: cur(k) for k in project.DEFAULTS}
    vals["I_from_beam"] = True
    rebuild(vals)


init_state()
V = ss._vals          # значения по умолчанию для полей этой версии


# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------
def csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV, который корректно открывается в русском Excel (разделитель «;», запятая в числах)."""
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def profile_table(L: float, xm, m, xn, n) -> pd.DataFrame:
    x = np.union1d(xm, xn)
    return pd.DataFrame({
        "x/L": x,
        "x, м": x * L,
        "Mдв, кН·м": np.interp(x, xm, m),
        "Nдв, кН": np.interp(x, xn, n),
    })


def protocol_table(res: core.WaveMomentResult) -> pd.DataFrame:
    return pd.DataFrame({
        "Обозначение": [s.symbol for s in res.steps],
        "Величина": [s.name for s in res.steps],
        "Пункт ПКПС": [s.ref for s in res.steps],
        "Значение": [core.format_value(s.value) for s in res.steps],
        "Ед. изм.": [s.unit for s in res.steps],
    })


def metric(col, label: str, value, unit: str = "") -> None:
    text = core.format_value(value)
    col.metric(label, f"{text} {unit}".strip() if value is not None else "—")


def c(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def beam_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=beam_mod.COLUMNS)
    for col, v in beam_mod.DEFAULT_ROW.items():
        if isinstance(v, bool):
            df[col] = df[col].fillna(False).astype(bool)
        elif isinstance(v, (int, float)):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("object")
    return df


@st.cache_data(show_spinner=False, max_entries=8)
def pdf_page_count(data: bytes) -> int:
    import pypdfium2 as pdfium
    return len(pdfium.PdfDocument(data))


@st.cache_data(show_spinner="Открываю чертёж…", max_entries=16)
def pdf_page_png(data: bytes, page: int, dpi: int) -> bytes:
    import io
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(data)
    img = doc[page].render(scale=dpi / 72).to_pil()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Эквивалентный брус — нулевое приближение считаем заранее, чтобы I можно было
# взять в расчёт Mдв прямо из боковой панели
# ----------------------------------------------------------------------------
beam_rows_now = current_beam_rows()
beam_zero = None
try:
    _els_now = beam_mod.elements_from_records(beam_rows_now)
    if _els_now:
        beam_zero = beam_mod.calculate_beam(_els_now, bool(cur("beam_half"))).zero
except beam_mod.BeamError:
    beam_zero = None

# ----------------------------------------------------------------------------
# Ввод данных
# ----------------------------------------------------------------------------
with st.sidebar:
    if "_flash" in ss:
        kind, msg = ss.pop("_flash")
        getattr(st, kind)(msg)

    st.header("Исходные данные")

    basins, types, units = list(BasinClass), list(ShipType), list(core.SPEED_UNITS)
    basin = st.selectbox("Класс судна (разряд бассейна)", basins, index=basins.index(V["basin"]),
                         key=K("basin"), format_func=lambda cl: f"«{cl.value}»")
    ship_type = st.selectbox(
        "Тип судна", types, index=types.index(V["ship_type"]), key=K("ship_type"),
        format_func=lambda s: s.value, help="Определяет коэффициент ks в формуле 2.2.10-10.")

    st.subheader("Главные размерения, м")
    L = st.number_input("L — длина", min_value=1.0, max_value=400.0, value=float(V["L"]),
                        step=1.0, format="%.2f", key=K("L"))
    B = st.number_input("B — ширина", min_value=0.5, max_value=80.0, value=float(V["B"]),
                        step=0.1, format="%.2f", key=K("B"))
    H = st.number_input("H — высота борта", min_value=0.3, max_value=30.0, value=float(V["H"]),
                        step=0.1, format="%.2f", key=K("H"),
                        help="Нужна для 3D-модели и проверки пропорций по табл. 2.1.3-1 "
                             "(L/H и B/H). В формулы п. 2.2.10 не входит.")
    T = st.number_input("T — осадка", min_value=0.1, max_value=20.0, value=float(V["T"]),
                        step=0.1, format="%.2f", key=K("T"))
    Tn = st.number_input("Tн — осадка носом", min_value=0.1, max_value=40.0, value=float(V["Tn"]),
                         step=0.1, format="%.2f", key=K("Tn"),
                         help="Для расчётного случая нагрузки. От неё зависит ударный момент Mу.")

    st.subheader("Полнота и скорость")
    Cb = st.number_input("Cb — коэффициент полноты (δ)", min_value=0.30, max_value=1.00,
                         value=float(V["Cb"]), step=0.01, format="%.3f", key=K("Cb"))
    c1, c2 = st.columns([3, 2])
    v_in = c1.number_input("V₀ — скорость хода", min_value=0.0, max_value=100.0, value=float(V["v_in"]),
                           step=0.5, format="%.2f", key=K("v_in"))
    v_unit = c2.selectbox("Ед.", units, index=units.index(V["v_unit"]), key=K("v_unit"))
    V0_kmh = core.speed_to_kmh(v_in, v_unit)
    if v_unit != "км/ч":
        st.caption(f"В расчёт идёт {V0_kmh:.2f} км/ч — в таких единицах скорость входит в формулы Правил.")

    st.subheader("Прочность и масса")
    I_from_beam = st.checkbox("Взять I из вкладки «Эквивалентный брус»", value=bool(V["I_from_beam"]),
                              key=K("I_from_beam"),
                              help="Момент инерции нулевого приближения (φ = 1) по таблице связей — "
                                   "именно такой I требует п. 2.2.10.")
    use_beam_I = I_from_beam and beam_zero is not None
    I_manual = st.number_input("I — момент инерции миделевого сечения, м⁴", min_value=0.001,
                               max_value=1000.0, value=float(V["I"]), step=0.1, format="%.4f",
                               key=K("I"), disabled=use_beam_I,
                               help="Эквивалентного бруса при проектных толщинах и "
                                    "редукционных коэффициентах, равных единице.")
    if use_beam_I:
        I = float(beam_zero.I)
        st.caption(f"В расчёт идёт I0 по брусу = **{c(I, 4)} м⁴** (поле выше не используется).")
    else:
        I = I_manual
        if I_from_beam:
            st.warning("Таблица бруса не посчитана — используется I, введённый вручную.")
    manual_D = st.checkbox("Задать водоизмещение D вручную", value=bool(V["manual_D"]), key=K("manual_D"))
    D_input = cur("D")
    rho = cur("rho")
    if manual_D:
        D_input = st.number_input("D — водоизмещение, кН", min_value=1.0, max_value=5.0e7,
                                  value=float(V["D"]), step=500.0, format="%.0f", key=K("D"),
                                  help="Для расчётного случая нагрузки.")
        D_val = D_input
        rho = 1.0
    else:
        D_val = None
        rhos = [1.0, 1.025]
        rho = st.selectbox("Плотность воды для оценки D", rhos,
                           index=rhos.index(V["rho"]) if V["rho"] in rhos else 0, key=K("rho"),
                           format_func=lambda r: "пресная, 1,000 т/м³" if r == 1.0
                           else "морская, 1,025 т/м³")

    st.subheader("Эпюра")
    pmin = core.min_plateau(basin)
    if basin in (BasinClass.M, BasinClass.O):
        plateau = st.slider("Постоянный участок Mдв, доли L", min_value=float(pmin), max_value=0.9,
                            value=float(min(max(V["plateau"], pmin), 0.9)), step=0.05,
                            key=K("plateau") + basin.name,
                            help="Для «М» и «О» границы участка не ближе 0,15L к миделю "
                                 "(то есть не менее 0,3L). По умолчанию 0,5L, как на рис. 2.2.10.")
    else:
        plateau = 0.5
        st.caption("Для «Р» и «Л» постоянный участок — 0,5L (п. 2.2.10).")

    st.divider()
    with st.expander("📁 Проект: сохранить / открыть", expanded=False):
        project_name = st.text_input("Название проекта", value=V["project_name"], key=K("project_name"))
        values_now = {
            "project_name": project_name, "basin": basin, "ship_type": ship_type,
            "L": L, "B": B, "H": H, "T": T, "Tn": Tn, "Cb": Cb, "v_in": v_in, "v_unit": v_unit,
            "I": I_manual, "I_from_beam": I_from_beam, "manual_D": manual_D, "D": D_input, "rho": rho,
            "plateau": plateau if basin in (BasinClass.M, BasinClass.O) else V["plateau"],
            "beam_half": cur("beam_half"), "beam_m_source": cur("beam_m_source"),
            "beam_M_sag": cur("beam_M_sag"), "beam_M_hog": cur("beam_M_hog"),
        }
        st.download_button("💾 Сохранить проект",
                           project.dump(values_now, beam_rows_now, current_mtv_rows()),
                           file_name=project.file_name(project_name), mime="application/json",
                           use_container_width=True,
                           help="Файл .json со всеми исходными данными, таблицей бруса и Mтв. "
                                "Чертёж PDF в проект не входит.")
        st.file_uploader("Открыть проект", type=["json"], key="project_file",
                         on_change=on_project_upload)
        st.button("Новый проект", on_click=on_new_project, use_container_width=True)


# ----------------------------------------------------------------------------
# Расчёт
# ----------------------------------------------------------------------------
st.title("Дополнительный волновой изгибающий момент")
st.caption("ПКПС РКО, часть I «Корпус и его оборудование», п. 2.2.10 — классы «М», «О», «Р», «Л»")

inp = core.ShipInput(basin=basin, ship_type=ship_type, L=L, B=B, T=T, Tn=Tn, Cb=Cb,
                     V0_kmh=V0_kmh, I=I, D=D_val, rho=rho, H=H)
try:
    res = core.calculate(inp)
except core.InputError as exc:
    st.error(str(exc))
    st.stop()

for w in res.warnings:
    (st.info if w.startswith("D оценено") else st.warning)(w)

cols = st.columns(5)
metric(cols[0], "Mв — волновой", res.Mv, "кН·м")
metric(cols[1], "kр — вибрация", res.kp)
metric(cols[2], "Mу — ударный", res.My, "кН·м")
metric(cols[3], "Mдв = kр·Mв + Mу", res.Mdv, "кН·м")
metric(cols[4], "Nдв = 4Mдв/L", res.Ndv, "кН")

export_slot = st.container()          # кнопка отчёта — заполняется в конце, когда всё посчитано

(tab_epure, tab_3d, tab_steps, tab_total, tab_beam,
 tab_protocol, tab_help) = st.tabs(["Эпюры", "3D-модель", "Ход решения", "Расчётный момент Mр",
                                    "Эквивалентный брус", "Протокол расчёта", "Справка"])

# для отчёта
fig_m = fig_n = fig_t = None
total_rows: list[tuple[str, str]] = []
mr_from_tab = None                    # (|Mр| прогиб, |Mр| перегиб) из вкладки Mр
beam_full = None
els: list = []
sections = steps_mod.build_solution(res, plateau)

# ----------------------------------------------------------------------------
# Эпюры
# ----------------------------------------------------------------------------
with tab_epure:
    if res.Mdv is None:
        st.info("Эпюры не построены: коэффициент kр определить нельзя (см. предупреждение выше).")
    else:
        xm, m = core.moment_profile(res.Mdv, plateau)
        xn, n = core.shear_profile(res.Ndv)

        fig_m = plots.fig_moment(xm, m, L, res.Mdv, plateau, basin.value)
        st.pyplot(fig_m)
        fig_n = plots.fig_shear(xn, n, L, res.Ndv)
        st.pyplot(fig_n)
        st.caption("Эпюры даны по модулю с двумя знаками (±): дополнительный волновой момент "
                   "учитывается и на прогиб, и на перегиб.")

        d1, d2, d3 = st.columns(3)
        d1.download_button("Эпюра Mдв (PNG)", plots.fig_to_png(fig_m),
                           file_name="epure_Mdv.png", mime="image/png")
        d2.download_button("Эпюра Nдв (PNG)", plots.fig_to_png(fig_n),
                           file_name="epure_Ndv.png", mime="image/png")
        d3.download_button("Данные эпюр (CSV)", csv_bytes(profile_table(L, xm, m, xn, n)),
                           file_name="epure_data.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# 3D-модель
# ----------------------------------------------------------------------------
WAVE_CHOICES = {
    "Без волны": hull_mod.WAVE_NONE,
    "Перегиб: гребень на миделе": hull_mod.WAVE_HOG,
    "Прогиб: подошва на миделе": hull_mod.WAVE_SAG,
}
hull3d = None

with tab_3d:
    st.markdown("Модель перестраивается при любом изменении данных слева: **L, B, H** задают "
                "габариты, **T** — уровень воды на борту, **Tн** — дифферент (нос глубже или мельче "
                "кормы), **Cb** — полноту обводов.")
    o1, o2, o3 = st.columns([3, 2, 2])
    wave_label = o1.radio("Волна длиной L", list(WAVE_CHOICES), horizontal=True,
                          help=f"Высота волны h = {res.params.h:g} м — по табл. 2.1.2 для класса "
                               f"«{basin.value}». Это та волна, для которой считается Mв.")
    z_stretch = o2.slider("Растянуть по высоте ×", min_value=1.0, max_value=5.0, value=1.0, step=0.5,
                          help="Настоящие суда очень вытянуты, и изменение высоты борта или осадки "
                               "почти не видно. Растяжение помогает разглядеть; подписи размеров "
                               "всегда показывают истинные значения.")
    fixed_scale = o3.checkbox("Фиксированный масштаб", value=True,
                              help="Сетка на воде — 5 × 5 м, сцена не масштабируется под судно: "
                                   "изменение длины видно сразу. Выключите, чтобы судно "
                                   "всегда занимало весь экран.")
    wave_mode = WAVE_CHOICES[wave_label]
    try:
        hull3d = hull_mod.build_hull(L, B, H, T, Tn, Cb)
        scene = view3d.build_scene(hull3d, wave_mode, res.params.h, fixed_scale, z_stretch)
        html = view3d.build_viewer_html(
            scene, view3d.info_lines(hull3d, wave_mode, res.params.h, z_stretch), height=540)
        components.html(html, height=550)
    except Exception as exc:                                    # модель вспомогательная, расчёт не ломаем
        hull3d = None
        st.error(f"3D-модель не построена: {exc}")
    else:
        for note in hull3d.notes:
            st.info(note)
        st.caption("Модель схематична: размеры (L, B, H, T, Tн) точные, форма обводов типовая, "
                   "полнота подогнана под введённый Cb. Это иллюстрация, а не теоретический чертёж. "
                   "Вращение — перетаскиванием, масштаб — колесом мыши или двумя пальцами.")

# ----------------------------------------------------------------------------
# Ход решения
# ----------------------------------------------------------------------------
def keyed_container(key: str):
    """Контейнер с ключом: при смене ключа Streamlit создаёт его содержимое заново
    (для версий без параметра key — обычный контейнер)."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


with tab_steps:
    st.caption("Каждая формула дана в общем виде, затем с подстановкой чисел и результатом. "
               "Номера формул и таблиц — по ПКПС РКО, часть I, п. 2.2.10.")
    # Ключ зависит от всех исходных данных: любое изменение — и блок рисуется с нуля,
    # формулы на экране всегда совпадают с расчётом (и с отчётом).
    steps_sig = hashlib.sha1(repr((inp, plateau)).encode()).hexdigest()[:12]
    with keyed_container(f"steps_{steps_sig}"):
        for sec in sections:
            with st.expander(sec.title, expanded=True):
                if sec.note:
                    st.caption(sec.note)
                for step in sec.steps:
                    st.markdown(f"**{step.name}** · {step.ref}")
                    if step.general:
                        st.latex(step.general)
                    if step.calc:
                        st.latex(step.calc)
                    if step.check:
                        st.caption(step.check)
                    if step.note:
                        st.caption(step.note)
                if sec.warning:
                    st.warning(sec.warning)
    st.download_button("Скачать решение (Markdown)",
                       steps_mod.solution_markdown(sections).encode("utf-8"),
                       file_name="reshenie_2_2_10.md", mime="text/markdown")

# ----------------------------------------------------------------------------
# Mр = Mтв ± Mдв
# ----------------------------------------------------------------------------
with tab_total:
    st.markdown("Расчётный момент **Mр = Mтв + Mдв** (п. 2.2.12-1). Момент на тихой воде Mтв "
                "в п. 2.2.10 не определяется — введите его эпюру из расчёта загрузки "
                "(x/L от 0 до 1, значения в кН·м). Знак: **плюс — перегиб** (палуба растянута), "
                "минус — прогиб; так же его понимает проверка напряжений во вкладке "
                "«Эквивалентный брус».")
    if res.Mdv is None:
        st.info("Mдв не определён — расчёт Mр недоступен.")
    else:
        edited = st.data_editor(
            pd.DataFrame(ss.mtv_rows, columns=MTV_COLUMNS), num_rows="dynamic", key=mtv_key(),
            column_config={
                "x/L": st.column_config.NumberColumn("x/L", min_value=0.0, max_value=1.0,
                                                     format="%.3f"),
                "Mтв, кН·м": st.column_config.NumberColumn("Mтв, кН·м", format="%.1f"),
            })
        valid = edited.dropna()
        if len(valid) < 3:
            st.info("Заполните Mтв хотя бы в одной точке между концами — "
                    "тогда будет построена эпюра Mр.")
        else:
            xm, m = core.moment_profile(res.Mdv, plateau)
            try:
                mtv, m_plus, m_minus = core.combine_with_still_water(
                    valid["x/L"].tolist(), valid["Mтв, кН·м"].tolist(), xm, m)
            except core.InputError as exc:
                st.error(str(exc))
            else:
                fig_t = plots.fig_total(xm, mtv, m_plus, m_minus, L)
                st.pyplot(fig_t)
                k1, k2 = st.columns(2)
                hog, sag = float(m_plus.max()), float(m_minus.min())
                k1.metric("max (Mтв + Mдв)", core.format_value(hog) + " кН·м")
                k2.metric("min (Mтв − Mдв)", core.format_value(sag) + " кН·м")
                mr_from_tab = (max(0.0, -sag), max(0.0, hog))
                total_rows = [("max (Mтв + Mдв), кН·м", core.format_value(hog)),
                              ("min (Mтв − Mдв), кН·м", core.format_value(sag))]
                out = pd.DataFrame({"x/L": xm, "x, м": xm * L, "Mтв, кН·м": mtv,
                                    "Mдв, кН·м": m, "Mтв+Mдв, кН·м": m_plus,
                                    "Mтв−Mдв, кН·м": m_minus})
                st.download_button("Данные Mр (CSV)", csv_bytes(out),
                                   file_name="Mr_data.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Эквивалентный брус
# ----------------------------------------------------------------------------
BEAM_COLCFG = {
    "Связь": st.column_config.TextColumn("Связь", width="medium"),
    "Тип": st.column_config.SelectboxColumn("Тип", options=beam_mod.TYPES, required=True,
                                            default=beam_mod.T_HPLATE, width="medium",
                                            help=beam_mod.HELP["Тип"]),
    "n": st.column_config.NumberColumn("n", min_value=1, step=1, default=1, help=beam_mod.HELP["n"]),
    "b, мм": st.column_config.NumberColumn("b, мм", min_value=0.0, format="%.0f"),
    "t, мм": st.column_config.NumberColumn("t, мм", min_value=0.0, format="%.1f"),
    "bп, мм": st.column_config.NumberColumn("bп, мм", min_value=0.0, format="%.0f"),
    "tп, мм": st.column_config.NumberColumn("tп, мм", min_value=0.0, format="%.1f"),
    "θ, °": st.column_config.NumberColumn("θ, °", min_value=0.0, max_value=90.0, format="%.0f",
                                          help=beam_mod.HELP["θ, °"]),
    "A, см²": st.column_config.NumberColumn("A, см²", min_value=0.0, format="%.2f"),
    "i, см⁴": st.column_config.NumberColumn("i, см⁴", min_value=0.0, format="%.1f"),
    "z, м": st.column_config.NumberColumn("z, м", format="%.3f", help=beam_mod.HELP["z, м"]),
    "y, м": st.column_config.NumberColumn("y, м", format="%.3f", help=beam_mod.HELP["y, м"]),
    "Вниз": st.column_config.CheckboxColumn("Вниз", default=False, help=beam_mod.HELP["Вниз"]),
    "ReH, МПа": st.column_config.NumberColumn("ReH, МПа", min_value=1.0, default=235.0, format="%.0f"),
    "Гибкая": st.column_config.CheckboxColumn("Гибкая", default=False, help=beam_mod.HELP["Гибкая"]),
    "Система": st.column_config.SelectboxColumn("Система", options=beam_mod.SYSTEMS,
                                                default=beam_mod.SYS_LONG, help=beam_mod.HELP["Система"]),
    "a, мм": st.column_config.NumberColumn("a, мм", min_value=0.0, format="%.0f", help=beam_mod.HELP["a, мм"]),
    "b пан., мм": st.column_config.NumberColumn("b пан., мм", min_value=0.0, format="%.0f",
                                                help=beam_mod.HELP["b пан., мм"]),
    "Местная нагрузка": st.column_config.CheckboxColumn("Местная нагрузка", default=False,
                                                        help=beam_mod.HELP["Местная нагрузка"]),
}

with tab_beam:
    st.markdown("Связи миделевого сечения, которые идут в эквивалентный брус (п. 2.2.31–2.2.35). "
                "Загрузите чертёж — он будет рядом с таблицей, и по нему удобно заполнять строки. "
                "**z** — от основной плоскости, размеры листов и профилей — в мм.")
    drawing = st.file_uploader("Чертёж миделевого сечения (PDF или картинка)",
                               type=["pdf", "png", "jpg", "jpeg"], key="drawing_file")
    side_by_side = False
    if drawing is not None:
        toggle = getattr(st, "toggle", st.checkbox)
        side_by_side = toggle("Чертёж рядом с таблицей", value=True, key="drawing_side",
                              help="Выключите на телефоне или узком экране — чертёж будет над таблицей.")
    if side_by_side:
        col_draw, col_tab = st.columns([1, 1])
    else:
        col_draw, col_tab = st.container(), st.container()

    if drawing is not None:
        with col_draw:
            data = drawing.getvalue()
            if drawing.name.lower().endswith(".pdf"):
                try:
                    pages = pdf_page_count(data)
                    p1, p2 = st.columns(2)
                    page = int(p1.number_input("Лист", min_value=1, max_value=pages, value=1, step=1)) - 1
                    dpi = p2.select_slider("Чёткость", options=[72, 110, 150, 200, 300], value=150)
                    png = pdf_page_png(data, page, int(dpi))
                    st.image(png, caption=f"{drawing.name}, лист {page + 1} из {pages}",
                             use_container_width=True)
                    st.download_button("Лист отдельной картинкой (PNG, для увеличения)", png,
                                       file_name=f"list_{page + 1}.png", mime="image/png")
                except Exception as exc:
                    st.error(f"PDF не открылся: {exc}")
            else:
                st.image(data, caption=drawing.name, use_container_width=True)

    with col_tab:
        b1, b2, b3 = st.columns([1, 1, 2])
        b1.button("Пример", on_click=on_beam_example, help="Заполнить таблицу условным примером "
                                                          "сечения грузового судна.")
        b2.button("Очистить", on_click=on_beam_clear)
        beam_half = b3.checkbox("Таблица на полусечение (×2)", value=bool(V["beam_half"]), key=K("beam_half"),
                    help="Связи одного борта; результат удваивается. Элементы в ДП (вертикальный киль, "
                         "кильсон) вводите половинной толщины.")
        edited_beam = st.data_editor(beam_frame(ss.beam_rows), num_rows="dynamic", key=beam_key(),
                                     column_config=BEAM_COLCFG, use_container_width=True, hide_index=True)
        with st.expander("Как заполнять таблицу"):
            st.markdown("\n".join(f"- **{k}** — {v}" for k, v in beam_mod.HELP.items()))
            st.markdown("- Лист борта или переборки удобно разбить на несколько поясьев: редуцирование "
                        "считается по напряжению на уровне центра тяжести каждой строки.\n"
                        "- Вырезы, обрывающиеся связи и надстройки (2.2.32–2.2.34) учитывайте сами — "
                        "вносите только те связи, которые идут в брус.")

    els = beam_mod.elements_from_records(edited_beam.to_dict("records"))
    st.subheader("Нулевое приближение (все φ = 1)")
    zero = None
    if not els:
        st.info("Таблица пуста — нажмите «Пример» или добавьте строки.")
    else:
        try:
            zero = beam_mod.calculate_beam(els, beam_half).zero
        except beam_mod.BeamError as exc:
            st.error(str(exc))
    if zero is not None:
        z1, z2, z3, z4, z5 = st.columns(5)
        z1.metric("A, м²", c(zero.A, 4))
        z2.metric("z0 от ОП, м", c(zero.z0, 3))
        z3.metric("I0, м⁴", c(zero.I, 4))
        z4.metric("W верх. кромки, м³", c(zero.W_deck, 4))
        z5.metric("W днища, м³", c(zero.W_bottom, 4))
        if use_beam_I:
            st.success("Этот I0 используется в расчёте Mдв.")
        elif abs(zero.I - I) / zero.I > 0.005:
            st.info(f"В расчёте Mдв сейчас I = {c(I, 3)} м⁴, а по брусу I0 = {c(zero.I, 4)} м⁴.")
            st.button("Использовать I0 в расчёте Mдв", on_click=on_use_beam_I)

        st.subheader("Напряжения от общего изгиба и редуцирование")
        src = st.radio("Расчётный момент Mр", M_SOURCES, horizontal=True, key=K("beam_m_source"),
                       index=M_SOURCES.index(V["beam_m_source"]) if V["beam_m_source"] in M_SOURCES else 0)
        M_sag = M_hog = None
        if src == M_SOURCES[0]:
            if res.Mdv is not None:
                M_sag = M_hog = abs(res.Mdv)
                st.caption(f"Mр = ±Mдв = ±{core.format_value(res.Mdv)} кН·м — без момента на тихой воде. "
                           "Для настоящей проверки задайте Mтв во вкладке «Расчётный момент Mр».")
            else:
                st.warning("Mдв не определён — выберите ввод Mр вручную.")
        elif src == M_SOURCES[1]:
            if mr_from_tab:
                M_sag, M_hog = mr_from_tab
                st.caption(f"Прогиб: {core.format_value(M_sag)} кН·м, перегиб: "
                           f"{core.format_value(M_hog)} кН·м (экстремумы по длине, знак «+» — перегиб).")
            else:
                st.warning("Во вкладке «Расчётный момент Mр» не заполнена эпюра Mтв.")
        else:
            m1, m2 = st.columns(2)
            M_sag = m1.number_input("|Mр| при прогибе (палуба сжата), кН·м", min_value=0.0,
                                    value=float(V["beam_M_sag"]), step=1000.0, format="%.0f",
                                    key=K("beam_M_sag"))
            M_hog = m2.number_input("|Mр| при перегибе (днище сжато), кН·м", min_value=0.0,
                                    value=float(V["beam_M_hog"]), step=1000.0, format="%.0f",
                                    key=K("beam_M_hog"))
        if M_sag or M_hog:
            try:
                beam_full = beam_mod.calculate_beam(els, beam_half, M_sag, M_hog, basin.value)
            except beam_mod.BeamError as exc:
                st.error(str(exc))
        if beam_full is not None:
            for w in beam_full.warnings:
                st.info(w)
            worst = beam_full.worst
            if worst is not None:
                msg = (f"Наибольшее использование: «{worst.name}», {worst.case}: σ = {c(worst.sigma, 1)} МПа "
                       f"при допускаемом {c(worst.allowable, 1)} МПа ({c(worst.usage * 100, 0)} %).")
                (st.success if worst.ok else st.error)(("Прочность обеспечена. " if worst.ok else
                                                        "Прочность НЕ обеспечена. ") + msg)
            case_names = [cr.case for cr in beam_full.cases]
            ci = None
            if case_names:
                ci = case_names.index(st.radio("Случай", case_names, horizontal=True, key="beam_case"))
            st.pyplot(plots.fig_section(beam_full, ci))
            st.caption("Синим — связи без редуцирования, оранжевым — редуцированные пластины.")
            if ci is not None:
                cr = beam_full.cases[ci]
                st.markdown(f"**Приближения ({cr.case})** — до разницы напряжений в крайних кромках ≤ 5 % "
                            "(п. 2.2.36)")
                st.dataframe(pd.DataFrame([{
                    "№": a.no, "z0, м": round(a.props.z0, 3), "I, м⁴": round(a.props.I, 4),
                    "σ верх, МПа": round(a.sigma_deck, 1), "σ низ, МПа": round(a.sigma_bottom, 1),
                    "Разница, %": None if a.diff is None else round(a.diff * 100, 2)}
                    for a in cr.approximations]), hide_index=True)
                if cr.reductions:
                    st.markdown("**Редукционные коэффициенты гибких пластин** (последнее приближение)")
                    st.dataframe(pd.DataFrame([{
                        "Связь": els[i].name, "σж, МПа": round(r.sigma, 1),
                        "σкр, МПа": None if r.sigma_cr is None else round(r.sigma_cr, 1),
                        "φ": round(r.phi, 3), "φ с жёсткими полосами": round(r.phi_eff, 3),
                        "Правило": r.rule} for i, r in cr.reductions.items()]), hide_index=True)
            st.markdown("**Проверка нормальных напряжений** σ = Mр·z·10⁻³/I (2.2.40-1), "
                        "[σ] = k·kн·ReH (2.2.67, табл. 2.2.68)")
            chk = pd.DataFrame([{
                "Связь": ch.name, "Случай": ch.case, "z волокна, м": round(ch.z, 3),
                "σ, МПа": round(ch.sigma, 1), "σ0, МПа": round(ch.sigma0, 1), "k": ch.factor,
                "[σ], МПа": round(ch.allowable, 1), "Использование, %": round(ch.usage * 100, 1),
                "Вывод": "✓" if ch.ok else "✗ превышено"} for ch in beam_full.checks])
            st.dataframe(chk, hide_index=True)
            st.download_button("Проверка напряжений (CSV)", csv_bytes(chk),
                               file_name="beam_stress.csv", mime="text/csv")
        st.caption("Не входят в расчёт: касательные напряжения (2.2.40-2), предельные моменты (2.2.76–2.2.80), "
                   "местный изгиб перекрытий.")
    st.download_button("Таблица связей (CSV)", csv_bytes(edited_beam), file_name="beam_elements.csv",
                       mime="text/csv")

# ----------------------------------------------------------------------------
# Протокол
# ----------------------------------------------------------------------------
proto = protocol_table(res)
with tab_protocol:
    st.dataframe(proto, hide_index=True)
    st.download_button("Протокол (CSV)", csv_bytes(proto),
                       file_name="protocol_2_2_10.csv", mime="text/csv")
    with st.expander("Сверка с таблицами Правил"):
        st.markdown(
            f"Параметры для табл. 2.2.10-2…2.2.10-4: **δ = {Cb:.3f}**, "
            f"**a = {res.a:.3f} м** (табл. 2.2.10-3), **b = {res.b:.3f} м** (табл. 2.2.10-4). "
            "Коэффициенты kδ, kт, kв в протоколе считаются по формулам; "
            "значения из таблиц должны совпасть с ними с точностью до интерполяции.")

# ----------------------------------------------------------------------------
# Справка
# ----------------------------------------------------------------------------
with tab_help:
    st.markdown("""
**Что считается.** Дополнительный волновой момент на миделе по п. 2.2.10:

`Mдв = ±(kр·Mв + Mу)`, `Nдв = 4·Mдв / L`.

- `Mв = 0,255·ε·kδ·kт·kв·B·L²·h` — прямое действие волны;
- `kр` — влияние волновой вибрации (через ψ, ωк, σ, kμ);
- `Mу = kу·φ1·D·L` — удар волн в носовую оконечность, возникает при малой осадке носом.

**Эквивалентный брус.** Таблица связей миделевого сечения → нейтральная ось, I, W (нулевое
приближение, φ = 1 — его I идёт в п. 2.2.10). Затем последовательные приближения с редуцированием
гибких пластин (2.2.36–2.2.39) для прогиба и перегиба и проверка σ = Mр·z·10⁻³/I по допускаемым
напряжениям (2.2.67, табл. 2.2.68).

**Проект.** «Сохранить проект» в боковой панели скачивает файл .json со всеми данными;
«Открыть проект» восстанавливает их. Чертёж PDF в файл не входит — загрузите его снова.

**Отчёт.** Кнопка «Экспорт отчёта» над вкладками: отметьте разделы и формат (PDF, Word).

**Единицы.** L, B, H, T, Tн — м; скорость V₀ — км/ч (можно вводить в м/с и узлах, пересчёт автоматический);
I — м⁴; D — кН; моменты — кН·м; напряжения — МПа.

**Высота борта H.** В формулы п. 2.2.10 не входит. Используется для 3D-модели и для проверки
пропорций L/H и B/H по табл. 2.1.3-1: если они превышены, раздел 2 Правил на судно не распространяется
(п. 2.1.5) — программа покажет предупреждение.

**Допущения этой версии.**
- Если D не задано, оно оценивается как 9,81·ρ·Cb·L·B·T. Это приближение.
- Эпюра Nдв построена по рис. 2.2.10 (узлы 0 / 0,15L / 0,30L / 0,40L / 0,60L / 0,70L / 0,85L / 1,0L).
- Если ωк/σ > 1,4, Правила не дают ψ: Mв и Mу считаются, а kр, Mдв и Nдв — нет.

**Пока не реализовано.** Классы «М-СП», «М-ПР», «О-ПР» (п. 2.2.11), минимум Mдв для
грузовых операций (п. 2.2.15), расчёт Mтв, касательные напряжения и предельные моменты.

Расчёт справочный: соответствие требованиям Правил подтверждает расчётчик и РКО.
""")


# ----------------------------------------------------------------------------
# Отчёт (кнопка над вкладками)
# ----------------------------------------------------------------------------
def report_data() -> report.ReportData:
    inputs = [
        ("Класс судна", f"«{basin.value}»"), ("Тип судна", ship_type.value),
        ("L — длина, м", c(L, 2)), ("B — ширина, м", c(B, 2)), ("H — высота борта, м", c(H, 2)),
        ("T — осадка, м", c(T, 2)), ("Tн — осадка носом, м", c(Tn, 2)), ("Cb (δ)", c(Cb, 3)),
        ("V₀ — скорость, км/ч", c(V0_kmh, 2)),
        ("I — момент инерции, м⁴", c(I, 4) + (" (из эквивалентного бруса)"
                                               if use_beam_I else "")),
        ("D — водоизмещение, кН", core.format_value(res.D) + ("" if manual_D else " (оценка по Cb)")),
        ("Постоянный участок эпюры", f"{c(plateau, 2)}L"),
    ]
    results = [("Mв — волновой, кН·м", core.format_value(res.Mv)), ("kр", core.format_value(res.kp)),
               ("Mу — ударный, кН·м", core.format_value(res.My)),
               ("Mдв = kр·Mв + Mу, кН·м", core.format_value(res.Mdv)),
               ("Nдв = 4Mдв/L, кН", core.format_value(res.Ndv))]
    d = report.ReportData(title="Дополнительный волновой изгибающий момент", project=project_name,
                          inputs=inputs, results=results, warnings=list(res.warnings), sections=sections,
                          total_rows=total_rows,
                          protocol=[list(proto.columns)] + proto.astype(str).values.tolist())
    if fig_m is not None:
        d.epure_png = [plots.fig_to_png(fig_m, 170), plots.fig_to_png(fig_n, 170)]
    if fig_t is not None:
        d.total_png = plots.fig_to_png(fig_t, 170)
    if hull3d is not None:
        d.model_png = report.hull_png(hull3d)
    bres = beam_full
    if bres is None and els:
        try:
            bres = beam_mod.calculate_beam(els, beam_half)
        except beam_mod.BeamError:
            bres = None
    if bres is not None:
        z = bres.zero
        d.beam_props = [("Сечение", "полусечение ×2" if bres.half else "полное"),
                        ("A, м²", c(z.A, 4)), ("z0 от ОП, м", c(z.z0, 3)), ("I0 (φ = 1), м⁴", c(z.I, 4)),
                        ("W верхней кромки, м³", c(z.W_deck, 4)), ("W днища, м³", c(z.W_bottom, 4))]
        for cr in bres.cases:
            f = cr.final
            d.beam_props += [(f"{cr.case}: |Mр|, кН·м", core.format_value(abs(cr.M))),
                             (f"{cr.case}: I после редуцирования ({f.no}-е прибл.), м⁴", c(f.props.I, 4)),
                             (f"{cr.case}: σ верх / низ, МПа", f"{c(f.sigma_deck, 1)} / {c(f.sigma_bottom, 1)}")]
        if bres.worst is not None:
            w = bres.worst
            d.beam_notes.append(("Прочность обеспечена. " if w.ok else "Прочность НЕ обеспечена. ") +
                                f"Наибольшее использование — «{w.name}», {w.case}: {c(w.usage * 100, 0)} %.")
        idx = list(range(len(bres.cases))) or [None]
        d.beam_png = [plots.fig_to_png(plots.fig_section(bres, i), 150) for i in idx]

        def dims(e) -> str:
            if e.kind == beam_mod.T_ANY:
                return "—"
            return f"{e.b:g}×{e.t:g}" + (f" + {e.bf:g}×{e.tf:g}" if e.bf and e.kind in (beam_mod.T_TEE, beam_mod.T_ANGLE) else "")

        d.beam_elements = [["Связь", "Тип", "n", "b×t, мм", "z, м", "A, см²", "Гибкая"]] + [
            [e.name, e.kind, str(e.n), dims(e), c(e.z, 3), c(e.area() * 1e4, 2), "да" if e.flexible else ""]
            for e in bres.elements]
        if bres.checks:
            d.beam_checks = [["Связь", "Случай", "σ, МПа", "[σ], МПа", "Использ., %", "Вывод"]] + [
                [ch.name, ch.case, c(ch.sigma, 1), c(ch.allowable, 1), c(ch.usage * 100, 1),
                 "✓" if ch.ok else "превышено"] for ch in bres.checks]
    return d


with export_slot:
    box = st.popover("📄 Экспорт отчёта") if hasattr(st, "popover") else st.expander("📄 Экспорт отчёта")
    with box:
        st.markdown("**Что включить в отчёт**")
        chosen = [k for k, name in report.SECTIONS.items()
                  if st.checkbox(name, value=True, key=f"rep_{k}")]
        st.markdown("**Формат**")
        f1, f2 = st.columns(2)
        want_pdf = f1.checkbox("PDF", value=True, key="rep_pdf")
        want_docx = f2.checkbox("Word (.docx)", value=True, key="rep_docx")
        state_bytes = project.dump(values_now, current_beam_rows(),
                                   current_mtv_rows(), with_time=False)
        sig = hashlib.sha1(state_bytes + repr((chosen, want_pdf, want_docx)).encode()).hexdigest()
        if st.button("Сформировать отчёт", type="primary", disabled=not chosen or not (want_pdf or want_docx),
                     use_container_width=True):
            with st.spinner("Собираю отчёт…"):
                rd = report_data()
                ss["_report"] = {"sig": sig,
                                 "pdf": report.build_pdf(rd, chosen) if want_pdf else None,
                                 "docx": report.build_docx(rd, chosen) if want_docx else None}
        rep = ss.get("_report")
        if rep and rep["sig"] == sig:
            base = project.file_name(project_name).replace(".pkps.json", "")
            if rep.get("pdf"):
                st.download_button("⬇ Скачать PDF", rep["pdf"], file_name=f"otchet_{base}.pdf",
                                   mime="application/pdf", use_container_width=True)
            if rep.get("docx"):
                st.download_button("⬇ Скачать Word", rep["docx"], file_name=f"otchet_{base}.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   use_container_width=True)
        elif rep:
            st.caption("Данные изменились — сформируйте отчёт заново.")
