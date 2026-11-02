"""
pkps_project.py — сохранение и загрузка проекта (файл .json).

В проект попадают все исходные данные боковой панели, таблица эквивалентного бруса,
настройки проверки напряжений и таблица Mтв. Формат — обычный JSON в UTF-8,
его можно открыть и поправить в любом редакторе.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from pkps_core import BasinClass, ShipType, SPEED_UNITS

FORMAT = "pkps-wave-moment"
VERSION = 1

# ключ виджета → значение по умолчанию
DEFAULTS: dict[str, Any] = {
    "project_name": "Новый проект",
    "basin": BasinClass.M,
    "ship_type": list(ShipType)[0],
    "L": 110.0, "B": 16.5, "H": 5.0, "T": 3.5, "Tn": 3.4,
    "Cb": 0.85,
    "v_in": 18.0, "v_unit": list(SPEED_UNITS)[0],
    "I": 5.0, "I_from_beam": False,
    "manual_D": False, "D": 52000.0, "rho": 1.0,
    "plateau": 0.5,
    # эквивалентный брус
    "beam_half": True,
    "beam_m_source": "Mдв (Mтв = 0)",
    "beam_M_sag": 0.0, "beam_M_hog": 0.0,
}

TABLE_KEYS = ("beam_rows", "mtv_rows")


def _enc(v: Any) -> Any:
    if isinstance(v, (BasinClass, ShipType)):
        return v.name
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _clean_rows(rows: list[dict]) -> list[dict]:
    return [{k: _enc(v) for k, v in r.items()} for r in rows]


def dump(values: dict[str, Any], beam_rows: list[dict], mtv_rows: list[dict],
         with_time: bool = True) -> bytes:
    """Проект → байты JSON. with_time=False — без отметки времени (для сравнения состояний)."""
    data = {
        "format": FORMAT,
        "version": VERSION,
        "saved": datetime.now().isoformat(timespec="seconds") if with_time else "",
        "inputs": {k: _enc(values.get(k, d)) for k, d in DEFAULTS.items()},
        "beam_rows": _clean_rows(beam_rows),
        "mtv_rows": _clean_rows(mtv_rows),
    }
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


class ProjectError(ValueError):
    pass


def _dec(key: str, v: Any) -> Any:
    d = DEFAULTS[key]
    if v is None:
        return d
    if isinstance(d, BasinClass):
        try:
            return BasinClass[v]
        except KeyError:
            try:
                return BasinClass(v)
            except ValueError:
                raise ProjectError(f"Неизвестный класс судна «{v}»")
    if isinstance(d, ShipType):
        try:
            return ShipType[v]
        except KeyError:
            raise ProjectError(f"Неизвестный тип судна «{v}»")
    if isinstance(d, bool):
        return bool(v)
    if isinstance(d, float):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ProjectError(f"Поле {key}: ожидалось число, получено «{v}»")
    if key == "v_unit" and v not in SPEED_UNITS:
        raise ProjectError(f"Неизвестная единица скорости «{v}»")
    return v


def load(raw: bytes) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Байты JSON → (значения виджетов, строки бруса, строки Mтв). Неизвестные поля игнорируются,
    отсутствующие берутся по умолчанию."""
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Файл не похож на проект: {exc}")
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ProjectError("Это не файл проекта «Волновой момент» (нет отметки формата).")
    if int(data.get("version", 0)) > VERSION:
        raise ProjectError("Проект сохранён более новой версией программы — обновите программу.")
    inputs = data.get("inputs") or {}
    values = {k: _dec(k, inputs.get(k)) for k in DEFAULTS}
    rows = {}
    for k in TABLE_KEYS:
        r = data.get(k) or []
        if not isinstance(r, list) or not all(isinstance(x, dict) for x in r):
            raise ProjectError(f"Раздел {k} повреждён.")
        rows[k] = r
    return values, rows["beam_rows"], rows["mtv_rows"]


def file_name(project_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in project_name).strip() or "proekt"
    return safe.replace(" ", "_") + ".pkps.json"


def apply_editor_state(rows: list[dict], state: Any, columns: list[str], defaults: dict) -> list[dict]:
    """Повторяет то, что st.data_editor делает с исходной таблицей: правки, удаление и
    добавление строк из st.session_state[key]. Нужен, чтобы использовать свежую таблицу
    ещё до того, как сам редактор отрисован (например, в боковой панели)."""
    out = [dict(r) for r in rows]
    if not isinstance(state, dict):
        return out
    for idx, changes in (state.get("edited_rows") or {}).items():
        i = int(idx)
        if 0 <= i < len(out):
            out[i].update(changes)
    deleted = {int(i) for i in (state.get("deleted_rows") or [])}
    out = [r for i, r in enumerate(out) if i not in deleted]
    for added in state.get("added_rows") or []:
        r = {c: defaults.get(c) for c in columns}
        r.update({k: v for k, v in added.items() if not str(k).startswith("_")})
        out.append(r)
    return out
