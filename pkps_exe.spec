# -*- mode: python ; coding: utf-8 -*-
# Сборка: pyinstaller --noconfirm pkps_exe.spec   (на Windows → dist/Volnovoy_moment.exe)
import glob

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []

# Пакеты, которым нужны свои файлы данных (веб-интерфейс Streamlit, dll pdfium, шаблоны Word…)
for pkg in ("streamlit", "pypdfium2", "pypdfium2_raw", "docx", "reportlab", "altair", "pyarrow"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h

# Streamlit проверяет свою версию через метаданные пакетов
for pkg in ("streamlit", "numpy", "pandas", "matplotlib", "python-docx", "reportlab", "pypdfium2"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Файлы приложения: app.py запускается Streamlit'ом как обычный файл, поэтому кладём исходники рядом
app_modules = sorted(p for p in glob.glob("pkps_*.py"))
datas += [("app.py", ".")] + [(p, ".") for p in app_modules]
datas += [("assets", "assets"), (".streamlit", ".streamlit")]

# Импорты app.py PyInstaller сам не видит (файл запускается во время работы) — перечисляем явно
hiddenimports += [p[:-3] for p in app_modules]
hiddenimports += [
    "streamlit.components.v1",
    "matplotlib.backends.backend_agg",
    "mpl_toolkits.mplot3d",
    "PIL.ImageChops",
    "pandas", "numpy",
]
hiddenimports += collect_submodules("streamlit.runtime")

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "IPython", "jupyter", "notebook", "pytest", "scipy"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Volnovoy_moment",
    console=True,            # окно консоли: видно, что программа работает; закрыли — остановилась
    upx=False,
    icon=None,
)
