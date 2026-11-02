"""
pkps_theme.py — оформление приложения: фон «Верфь на закате» и стили панелей.

Фон — SVG из assets/bg_shipyard.svg, вшивается в CSS как data-URI (отдельных запросов нет).
Если файла нет, приложение просто остаётся на однотонном тёмном фоне.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

ASSET = Path(__file__).with_name("assets") / "bg_shipyard.svg"

PANEL = "rgba(11, 16, 34, 0.72)"          # полупрозрачная «стеклянная» панель
PANEL_SOFT = "rgba(11, 16, 34, 0.42)"  # карточки внутри панели
LINE = "rgba(255, 255, 255, 0.12)"
ACCENT = "#f6a04d"


def _bg_data_uri() -> str | None:
    try:
        text = ASSET.read_text(encoding="utf-8")
    except OSError:
        return None
    # служебные метаданные (если файл их получил при копировании) фону не нужны
    text = re.sub(r"<metadata>.*?</metadata>", "", text, flags=re.S)
    raw = text.encode("utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


def build_css() -> str:
    uri = _bg_data_uri()
    bg = (f'background: #0b1022 url("{uri}") center bottom / cover no-repeat;' if uri
          else "background: #0b1022;")
    return f"""
<style>
/* Фон — только картинка на корневом контейнере. Позиционирование, z-index и прокрутку
   элементов Streamlit не трогаем: боковая панель прокручивается колёсиком как обычно. */
[data-testid="stApp"], .stApp {{ {bg} background-attachment: fixed; }}
[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {{
  background: transparent !important;
}}

/* Основная колонка — полупрозрачная панель */
[data-testid="stMainBlockContainer"], .block-container {{
  background: {PANEL};
  border: 1px solid {LINE}; border-radius: 18px;
  padding-top: 1.6rem !important; padding-bottom: 2.2rem !important;
  margin-top: 1rem; margin-bottom: 2rem;
  box-shadow: 0 12px 40px rgba(0, 0, 0, .35);
}}

/* Боковая панель — только цвет */
[data-testid="stSidebar"] {{
  background-color: rgba(11, 16, 34, 0.86) !important;
  border-right: 1px solid {LINE};
}}
[data-testid="stSidebar"] h2 {{ margin-top: .2rem; }}
[data-testid="stSidebar"] h3 {{
  margin-top: 1.1rem; padding-top: .6rem; border-top: 1px solid {LINE}; font-size: 1.02rem;
}}

/* Заголовок */
h1 {{ letter-spacing: .2px; text-shadow: 0 2px 14px rgba(0, 0, 0, .45); }}
h1::after {{
  content: ""; display: block; width: 84px; height: 3px; margin-top: .55rem;
  border-radius: 2px; background: linear-gradient(90deg, {ACCENT}, #d9708a);
}}

/* Карточки показателей */
[data-testid="stMetric"] {{
  background: {PANEL_SOFT}; border: 1px solid {LINE}; border-radius: 12px; padding: .7rem .9rem;
}}
[data-testid="stMetricLabel"] {{ opacity: .8; }}

/* Вкладки */
[data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid {LINE}; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: {ACCENT} !important; }}

/* Раскрывающиеся блоки «Хода решения» */
[data-testid="stExpander"] {{
  background: {PANEL_SOFT}; border: 1px solid {LINE} !important; border-radius: 12px;
}}

/* Эпюры (matplotlib, белый фон) — на светлой «карточке» с рамкой */
[data-testid="stImage"] img {{
  background: #ffffff; border-radius: 10px; padding: 6px; box-shadow: 0 4px 18px rgba(0, 0, 0, .35);
}}

/* 3D-модель */
iframe {{ border-radius: 12px; }}

/* Таблицы и кнопки */
[data-testid="stDataFrame"] {{ border-radius: 10px; }}
.stDownloadButton button, .stButton button {{
  border: 1px solid {ACCENT}; border-radius: 10px; background: rgba(246, 160, 77, .12);
}}
.stDownloadButton button:hover, .stButton button:hover {{ background: rgba(246, 160, 77, .28); }}

/* Телефон: панели шире, отступы меньше, фон не «режется» */
@media (max-width: 760px) {{
  [data-testid="stMainBlockContainer"], .block-container {{
    padding-top: 1rem !important; margin-top: .3rem; border-radius: 12px;
  }}
  [data-testid="stMetric"] {{ padding: .5rem .6rem; }}
}}
</style>
"""


# Запрет автоперевода. Streamlit по умолчанию помечает страницу как английскую (lang="en"),
# и Chrome предлагает «перевести» русский текст. Переводчик подменяет текст на странице,
# портит формулы и, главное, мешает Streamlit обновлять числа после пересчёта.
NO_TRANSLATE_JS = """
<script>
(function () {
  try {
    var d = window.parent.document, h = d.documentElement;
    h.setAttribute("lang", "ru");
    h.setAttribute("translate", "no");
    h.classList.add("notranslate");
    if (d.body) { d.body.setAttribute("translate", "no"); d.body.classList.add("notranslate"); }
    if (!d.querySelector('meta[name="google"][content="notranslate"]')) {
      var m = d.createElement("meta");
      m.name = "google"; m.content = "notranslate";
      d.head.appendChild(m);
    }
  } catch (e) { /* без доступа к странице — просто ничего не делаем */ }
})();
</script>
"""

# Нулевая высота служебного блока со скриптом
HIDE_HELPER_CSS = """
<style>
iframe[height="0"] { display: block; }
div[data-testid="stIFrame"]:has(iframe[height="0"]),
div.element-container:has(iframe[height="0"]) { height: 0 !important; margin: 0 !important; overflow: hidden; }
</style>
"""


def apply_theme() -> None:
    """Подключить оформление и запрет автоперевода. Вызывать сразу после st.set_page_config."""
    import streamlit as st
    import streamlit.components.v1 as components
    st.markdown(build_css() + HIDE_HELPER_CSS, unsafe_allow_html=True)
    components.html(NO_TRANSLATE_JS, height=0)
