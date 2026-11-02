"""
launcher.py — точка входа для .exe (PyInstaller).

Двойной щелчок по Volnovoy_moment.exe:
  1) запускает Streamlit-сервер с приложением app.py на свободном порту;
  2) сам открывает браузер на http://localhost:<порт>;
  3) работает, пока открыто чёрное окно консоли (закрыли окно — программа остановилась).

Переменные окружения (для проверки сборки на сервере):
  PKPS_PORT=8501        — фиксированный порт;
  PKPS_NO_BROWSER=1     — не открывать браузер.
"""
from __future__ import annotations

import multiprocessing
import os
import socket
import sys
import threading
import webbrowser
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



def base_dir() -> str:
    """Папка с app.py: внутри exe — временная папка распаковки, иначе — папка скрипта."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def free_port(preferred: int = 8501) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    base = base_dir()
    os.chdir(base)                       # здесь лежит .streamlit/config.toml (тёмная тема)
    if base not in sys.path:
        sys.path.insert(0, base)

    port = int(os.environ.get("PKPS_PORT") or free_port())
    url = f"http://localhost:{port}"

    print("=" * 64)
    print("  Дополнительный волновой изгибающий момент — ПКПС РКО, п. 2.2.10")
    print(f"  Программа запускается, браузер откроется сам: {url}")
    print("  Если браузер не открылся — скопируйте адрес в адресную строку.")
    print("  НЕ закрывайте это окно, пока работаете. Закрыли — программа остановилась.")
    print("=" * 64, flush=True)

    if not os.environ.get("PKPS_NO_BROWSER"):
        threading.Timer(4.0, lambda: webbrowser.open(url)).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", os.path.join(base, "app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=localhost",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--server.maxUploadSize=50",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
