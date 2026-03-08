#!/usr/bin/env python3
"""
main_dock.py — Field Viewer  [Docking Edition]
================================================
Панели:
  Left   — Cell List (таблица ячеек, живые значения + цвет)
  Center — Field View (поле, 100 ячеек в круге)
  Right  — Antenna Detail (256 кружков) + Antenna Table (сортируемая)
  Bottom — Log
  Controls — карта цветов, lerp, Settings (форма/кол-во/пороги)

Запуск:
    python udp_server_test.py --hz 1 --cells 100 --elems 256
    python main_dock.py
"""
import math, queue, socket, threading, time, json, argparse
from typing import Optional, List, Tuple

import dearpygui.dearpygui as dpg

from geometry    import (rect_points, circle_points, hex_points,
                          circle_grid_centers, point_in_polygon, poly_centroid)
from color_map   import value_to_color, lerp, CMAP_NAMES
from data_models import Field, Cell, Element

# ════════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ════════════════════════════════════════════════════════════════════════════
CFG = dict(
    udp_host    = "0.0.0.0",
    udp_port    = 5005,
    n_cells     = 100,
    n_elems     = 256,
    cell_side   = 44.0,
    cell_shape  = "square",   # square | hexagon | circle
    field_r     = 310.0,
    field_cx    = 400.0,
    field_cy    = 300.0,
    cmap        = "heat",
    lerp_speed  = 0.08,
    vmin        = 0.0,
    vmax        = 100.0,
    thr_red     = 30.0,       # ниже — красный (плохо)
    thr_yellow  = 70.0,       # ниже — жёлтый (внимание), выше — зелёный
    target_fps  = 60,
)

# ── Теги окон ────────────────────────────────────────────────────────────────
W_FIELD    = "w_field"
W_CELLS    = "w_cells"
W_DETAIL   = "w_detail"
W_ANT_TBL  = "w_ant_tbl"   # таблица антенн (новое)
W_CTRL     = "w_ctrl"
W_LOG      = "w_log"

FIELD_LAYER  = "field_layer"
DETAIL_LAYER = "detail_layer"
DETAIL_W     = 500.0
DETAIL_H     = 500.0

# ── Глобальное состояние сортировки таблицы антенн ───────────────────────────
# _order: list[elem_id] — порядок отображения строк
_order: List[int] = []
_sort_col = 2        # 0=N  1=Value  2=Status (red вверху)
_sort_dir = 1        # 1=asc  -1=desc

# UUID теги строк таблицы антенн (tag_v, tag_s) для каждой строки
# Используем UUID вместо строковых тегов — DPG не освобождает строковые теги
# при delete_item, что приводит к конфликту при перестройке таблицы
_row_tags: List[Tuple[int, int]] = []

# UUID самих строк (mvTableRow) — чтобы удалять ТОЛЬКО строки, не колонки.
# ВАЖНО: delete_item(ANT_TBL, children_only=True) убивает колонки тоже!
# После удаления колонок таблица рендерится как чёрный прямоугольник.
_row_items: List[int] = []

# UUID числовых ID колонок для sort_cb (Antenna Table)
_col_n_uid: int = 0
_col_v_uid: int = 0
_col_s_uid: int = 0

# ── Глобальное состояние таблицы ячеек (Cell List) ────────────────────────────
_cell_row_items: List[int] = []
_cell_row_tags:  List[Tuple[int, int]] = []   # (tag_v, tag_s) per row
_cell_order:     List[int] = []
_cell_sort_col:  int = 2     # 0=N  1=Value  2=Status
_cell_sort_dir:  int = 1     # 1=asc (BAD first)
_ccol_n_uid: int = 0
_ccol_v_uid: int = 0
_ccol_s_uid: int = 0

# Двойной клик по строке Cell List
_dblclk_cid:  int   = -1
_dblclk_t:    float = 0.0

# Ссылка на fld_ref из main() (для callback-ов)
_fld_ref_global = None

# ════════════════════════════════════════════════════════════════════════════
# ГЕОМЕТРИЯ ЯЧЕЕК
# ════════════════════════════════════════════════════════════════════════════

def make_cell_poly(px: float, py: float) -> list:
    cs    = CFG["cell_side"]
    shape = CFG["cell_shape"]
    if shape == "hexagon":
        return hex_points(px, py, cs * 0.55)
    elif shape == "circle":
        return circle_points(px, py, cs * 0.48, n=20)
    else:  # square
        return rect_points(px, py, cs * 0.90, cs * 0.90)


def build_field() -> Field:
    cx0, cy0 = CFG["field_cx"], CFG["field_cy"]
    fr, cs   = CFG["field_r"],  CFG["cell_side"]
    centers  = circle_grid_centers(CFG["n_cells"], cx0, cy0, fr, cs, gap=4)
    cells = []
    for idx, (px, py) in enumerate(centers):
        pts   = make_cell_poly(px, py)
        elems = [Element(id=eid, label=f"a{eid}") for eid in range(CFG["n_elems"])]
        for e in elems:
            e.current_value = e.target_value = 50.0
        cell = Cell(id=idx, label=f"C{idx}", points=pts,
                    target_value=50.0, current_value=50.0, elements=elems)
        cell.pulse_phase = idx * 0.41
        cells.append(cell)
    border = circle_points(cx0, cy0, fr, n=80)
    return Field(cells=cells, border_poly=border, name="Antenna Panel")


def build_detail_layout(cell: Cell) -> None:
    n = len(cell.elements)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw, ch = DETAIL_W / cols, DETAIL_H / rows
    r = min(cw, ch) * 0.42
    for idx, elem in enumerate(cell.elements):
        row, col = divmod(idx, cols)
        elem._cx = 6 + col * cw + cw / 2
        elem._cy = 6 + row * ch + ch / 2
        elem._r  = r
        elem.points = []


# ════════════════════════════════════════════════════════════════════════════
# СТАТУС АНТЕННЫ
# ════════════════════════════════════════════════════════════════════════════

def antenna_status(v: float) -> Tuple[int, Tuple[int,int,int,int]]:
    """Возвращает (sort_key, color). 0=red, 1=yellow, 2=green."""
    if v < CFG["thr_red"]:
        return 0, (220, 65,  65,  255)
    if v < CFG["thr_yellow"]:
        return 1, (220, 175, 45,  255)
    return     2, (60,  210, 70,  255)


# ════════════════════════════════════════════════════════════════════════════
# РЕНДЕР ПОЛЯ
# ════════════════════════════════════════════════════════════════════════════

def draw_field(fld: Field):
    dpg.delete_item(FIELD_LAYER, children_only=True)
    if fld.border_poly:
        dpg.draw_polygon(fld.border_poly,
                         color=(70,130,210,200), fill=(12,25,55,55),
                         thickness=2, parent=FIELD_LAYER)
    cmap = CFG["cmap"]
    vmin, vmax = CFG["vmin"], CFG["vmax"]
    for cell in fld.cells:
        v  = cell.current_value
        bc = value_to_color(v, vmin, vmax, cmap, alpha=215)
        fc, oc, ot = bc, (175,175,175,130), 1
        if cell.is_flashing:
            ft = min(1.0, cell.flash_timer / 0.4)
            fw = int(200*ft)
            fc = (min(255,bc[0]+fw), min(255,bc[1]+fw), min(255,bc[2]+fw), bc[3])
            oc = (255,255,255,int(200*ft)); ot = 2
        dpg.draw_polygon(cell.points, color=oc, fill=fc,
                         thickness=ot, parent=FIELD_LAYER)
        px, py = poly_centroid(cell.points)
        dpg.draw_text((px-13,py-8), cell.label,
                      color=(255,255,255,215), size=11, parent=FIELD_LAYER)
        dpg.draw_text((px-13,py+2), f"{v:.1f}",
                      color=(215,215,215,175), size=10, parent=FIELD_LAYER)


# ════════════════════════════════════════════════════════════════════════════
# РЕНДЕР АНТЕНН (кружки в W_DETAIL)
# ════════════════════════════════════════════════════════════════════════════

def draw_detail(cell: Cell):
    if not dpg.does_item_exist(DETAIL_LAYER):
        return
    dpg.delete_item(DETAIL_LAYER, children_only=True)
    cmap = CFG["cmap"]
    vmin, vmax = CFG["vmin"], CFG["vmax"]
    for elem in cell.elements:
        cx = getattr(elem, "_cx", None)
        if cx is None:
            continue
        cy, r = elem._cy, elem._r
        v    = elem.current_value
        fill = value_to_color(v, vmin, vmax, cmap, alpha=235)
        oc   = (min(255,fill[0]+40), min(255,fill[1]+40),
                min(255,fill[2]+40), 185)
        dpg.draw_circle((cx,cy), r,   color=oc, fill=fill,
                        thickness=1, parent=DETAIL_LAYER)
        if r > 7:
            dpg.draw_circle((cx,cy), 2, color=(255,255,255,140),
                            fill=(255,255,255,100), thickness=0,
                            parent=DETAIL_LAYER)


# ════════════════════════════════════════════════════════════════════════════
# СОРТИРУЕМАЯ ТАБЛИЦА АНТЕНН (W_ANT_TBL)
# ════════════════════════════════════════════════════════════════════════════

ANT_TBL = "ant_tbl"

def _sorted_order(cell: Cell) -> List[int]:
    """Возвращает список elem.id в порядке текущей сортировки."""
    global _sort_col, _sort_dir
    def key(eid):
        v = cell.elements[eid].current_value
        if _sort_col == 0:   return eid
        elif _sort_col == 1: return v
        else:                return antenna_status(v)[0]
    ids = list(range(len(cell.elements)))
    ids.sort(key=key, reverse=(_sort_dir == -1))
    return ids


def _clear_table_rows():
    """Удаляет только строки таблицы, НЕ трогая колонки."""
    global _row_items, _row_tags
    for row_id in _row_items:
        if dpg.does_item_exist(row_id):
            dpg.delete_item(row_id)
    _row_items = []
    _row_tags  = []


def build_antenna_table_rows(cell: Cell):
    """
    Полностью перестраивает строки таблицы.
    НЕ использует delete_item(children_only=True) — он удаляет колонки тоже,
    после чего таблица выглядит как чёрный прямоугольник.
    Вместо этого удаляем только tracked row-элементы (_row_items).
    """
    global _order, _row_tags, _row_items
    _order = _sorted_order(cell)
    if not dpg.does_item_exist(ANT_TBL):
        return
    _clear_table_rows()            # удалить только строки, колонки целы
    for eid in _order:
        elem  = cell.elements[eid]
        v     = elem.current_value
        st, color = antenna_status(v)
        st_label  = {0: "BAD", 1: "WARN", 2: "OK"}[st]
        tag_v = dpg.generate_uuid()
        tag_s = dpg.generate_uuid()
        _row_tags.append((tag_v, tag_s))
        row_id = dpg.add_table_row(parent=ANT_TBL)   # получаем UUID строки
        _row_items.append(row_id)
        dpg.add_text(str(eid),      parent=row_id)
        dpg.add_text(f"{v:.1f}",   parent=row_id, tag=tag_v,
                     color=(210, 210, 210, 255))
        dpg.add_text(st_label,     parent=row_id, tag=tag_s,
                     color=list(color))


def update_antenna_values(cell: Cell):
    """Обновляет значения в строках без перестройки (вызывать каждый кадр)."""
    if not dpg.does_item_exist(ANT_TBL) or not _order or not _row_tags:
        return
    for row_idx, eid in enumerate(_order):
        if row_idx >= len(_row_tags) or eid >= len(cell.elements):
            break
        elem = cell.elements[eid]
        v    = elem.current_value
        st, color = antenna_status(v)
        st_label  = {0: "BAD", 1: "WARN", 2: "OK"}[st]
        tag_v, tag_s = _row_tags[row_idx]
        if dpg.does_item_exist(tag_v):
            dpg.set_value(tag_v, f"{v:.1f}")
        if dpg.does_item_exist(tag_s):
            dpg.set_value(tag_s, st_label)
            dpg.configure_item(tag_s, color=list(color))


def build_antenna_table_panel(cell: Optional[Cell] = None):
    """Строит панель W_ANT_TBL с сортируемой таблицей."""
    global _col_n_uid, _col_v_uid, _col_s_uid

    # Генерируем UUID для колонок заранее, чтобы sort_cb мог их сравнивать
    _col_n_uid = dpg.generate_uuid()
    _col_v_uid = dpg.generate_uuid()
    _col_s_uid = dpg.generate_uuid()

    def sort_cb(sender, sort_specs):
        global _sort_col, _sort_dir, _current_cell
        if sort_specs is None or _current_cell is None:
            return
        col_id    = sort_specs[0][0]   # числовой UUID колонки
        _sort_dir = sort_specs[0][1]   # 1=asc / -1=desc
        if   col_id == _col_n_uid: _sort_col = 0
        elif col_id == _col_v_uid: _sort_col = 1
        else:                      _sort_col = 2
        build_antenna_table_rows(_current_cell)

    with dpg.window(label="  Antenna Table", tag=W_ANT_TBL,
                    width=265, height=580, pos=(1355, 20), no_close=False):

        dpg.add_text("Sort: click column header",
                     color=(120, 140, 170, 200))
        dpg.add_separator()

        # Счётчики статусов (ASCII — Unicode "●" не рендерится в дефолтном шрифте)
        with dpg.group(horizontal=True):
            dpg.add_text("BAD:",    color=(220, 65,  65,  255))
            dpg.add_text("0", tag="cnt_red",    color=(220, 65,  65,  255))
            dpg.add_spacer(width=8)
            dpg.add_text("WARN:",   color=(220, 175, 45,  255))
            dpg.add_text("0", tag="cnt_yellow", color=(220, 175, 45,  255))
            dpg.add_spacer(width=8)
            dpg.add_text("OK:",     color=(60,  210, 70,  255))
            dpg.add_text("0", tag="cnt_green",  color=(60,  210, 70,  255))

        dpg.add_separator()

        with dpg.table(
            tag=ANT_TBL,
            header_row=True,
            sortable=True,
            sort_multi=False,
            sort_tristate=False,
            callback=sort_cb,
            resizable=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            scrollY=True,
            freeze_rows=1,
            height=480,    # явная высота — height=-1 иногда даёт 0 в DPG 2.x
            width=-1,
        ):
            dpg.add_table_column(label="N",      tag=_col_n_uid,
                                 width_fixed=True, init_width_or_weight=45)
            dpg.add_table_column(label="Value",  tag=_col_v_uid,
                                 width_fixed=True, init_width_or_weight=65)
            dpg.add_table_column(label="Status", tag=_col_s_uid,
                                 width_stretch=True,
                                 default_sort=True,
                                 prefer_sort_ascending=True)

    if cell is not None:
        build_antenna_table_rows(cell)


def update_status_counters(cell: Cell):
    """Обновляет счётчики red/yellow/green над таблицей."""
    if not dpg.does_item_exist("cnt_red"):
        return
    r = y = g = 0
    for elem in cell.elements:
        st, _ = antenna_status(elem.current_value)
        if st == 0:   r += 1
        elif st == 1: y += 1
        else:         g += 1
    dpg.set_value("cnt_red",    str(r))
    dpg.set_value("cnt_yellow", str(y))
    dpg.set_value("cnt_green",  str(g))


# ════════════════════════════════════════════════════════════════════════════
# СПИСОК ЯЧЕЕК (W_CELLS) — сортируемая таблица N/Value/Status + двойной клик
# ════════════════════════════════════════════════════════════════════════════

CELL_TBL = "cell_tbl"


def _open_cell_detail(cell: Cell):
    """Открывает детальный вид ячейки (антенны + таблица антенн)."""
    global _current_cell
    _current_cell = cell
    build_detail_layout(cell)
    dpg.configure_item("detail_hint", show=False)
    dpg.configure_item("detail_draw", show=True)
    dpg.configure_item(W_DETAIL,
        label=f"  Cell {cell.id} · {len(cell.elements)} antennas"
              f"  [val={cell.current_value:.1f}]")
    build_antenna_table_rows(cell)
    log(f"Cell {cell.id} selected, val={cell.current_value:.1f}")


def _on_cell_row_click(sender, app_data, user_data):
    """Обработчик клика по строке Cell List. Двойной клик → открыть ячейку."""
    global _dblclk_cid, _dblclk_t
    cid = user_data
    now = time.perf_counter()
    if cid == _dblclk_cid and (now - _dblclk_t) < 0.4:
        # Двойной клик — открыть ячейку
        fld = _fld_ref_global[0] if _fld_ref_global else None
        if fld and cid < len(fld.cells):
            _open_cell_detail(fld.cells[cid])
        _dblclk_cid = -1
    else:
        _dblclk_cid = cid
        _dblclk_t   = now


def _cell_sorted_order(fld: Field) -> List[int]:
    """Отсортированные ID ячеек."""
    def key(cid):
        v = fld.cells[cid].current_value
        if _cell_sort_col == 0:   return cid
        elif _cell_sort_col == 1: return v
        else:                     return antenna_status(v)[0]
    ids = list(range(len(fld.cells)))
    ids.sort(key=key, reverse=(_cell_sort_dir == -1))
    return ids


def _clear_cell_rows():
    """Удаляет только строки Cell Table, не колонки."""
    global _cell_row_items, _cell_row_tags
    for row_id in _cell_row_items:
        if dpg.does_item_exist(row_id):
            dpg.delete_item(row_id)
    _cell_row_items = []
    _cell_row_tags  = []


def build_cell_list_rows(fld: Field):
    """Полностью перестраивает строки таблицы ячеек."""
    global _cell_order, _cell_row_tags, _cell_row_items
    _cell_order = _cell_sorted_order(fld)
    if not dpg.does_item_exist(CELL_TBL):
        return
    _clear_cell_rows()
    for cid in _cell_order:
        cell = fld.cells[cid]
        v    = cell.current_value
        st, color = antenna_status(v)
        st_label  = {0: "BAD", 1: "WARN", 2: "OK"}[st]
        tag_v = dpg.generate_uuid()
        tag_s = dpg.generate_uuid()
        _cell_row_tags.append((tag_v, tag_s))
        row_id = dpg.add_table_row(parent=CELL_TBL)
        _cell_row_items.append(row_id)
        # N — selectable (кликабельный для двойного клика)
        dpg.add_selectable(label=str(cid), parent=row_id,
                           callback=_on_cell_row_click, user_data=cid)
        dpg.add_text(f"{v:.1f}", parent=row_id, tag=tag_v,
                     color=(210, 210, 210, 255))
        dpg.add_text(st_label,   parent=row_id, tag=tag_s,
                     color=list(color))


def update_cell_list_values(fld: Field):
    """Обновляет значения в строках без перестройки."""
    if not _cell_order or not _cell_row_tags:
        return
    for row_idx, cid in enumerate(_cell_order):
        if row_idx >= len(_cell_row_tags) or cid >= len(fld.cells):
            break
        cell = fld.cells[cid]
        v    = cell.current_value
        st, color = antenna_status(v)
        st_label  = {0: "BAD", 1: "WARN", 2: "OK"}[st]
        tag_v, tag_s = _cell_row_tags[row_idx]
        if dpg.does_item_exist(tag_v):
            dpg.set_value(tag_v, f"{v:.1f}")
        if dpg.does_item_exist(tag_s):
            dpg.set_value(tag_s, st_label)
            dpg.configure_item(tag_s, color=list(color))


def update_cell_status_counters(fld: Field):
    """Обновляет счётчики BAD/WARN/OK для ячеек."""
    if not dpg.does_item_exist("ccnt_red"):
        return
    r = y = g = 0
    for cell in fld.cells:
        st, _ = antenna_status(cell.current_value)
        if st == 0:   r += 1
        elif st == 1: y += 1
        else:         g += 1
    dpg.set_value("ccnt_red",    str(r))
    dpg.set_value("ccnt_yellow", str(y))
    dpg.set_value("ccnt_green",  str(g))


def build_cell_list_panel(fld: Field):
    """Создаёт панель W_CELLS с сортируемой таблицей ячеек."""
    global _ccol_n_uid, _ccol_v_uid, _ccol_s_uid
    _ccol_n_uid = dpg.generate_uuid()
    _ccol_v_uid = dpg.generate_uuid()
    _ccol_s_uid = dpg.generate_uuid()

    def cell_sort_cb(sender, sort_specs):
        global _cell_sort_col, _cell_sort_dir
        if sort_specs is None:
            return
        col_id = sort_specs[0][0]
        _cell_sort_dir = sort_specs[0][1]
        if   col_id == _ccol_n_uid: _cell_sort_col = 0
        elif col_id == _ccol_v_uid: _cell_sort_col = 1
        else:                       _cell_sort_col = 2
        if _fld_ref_global:
            build_cell_list_rows(_fld_ref_global[0])

    with dpg.window(label="  Cell List", tag=W_CELLS,
                    width=240, height=640, pos=(0, 20), no_close=False):

        dpg.add_text("Double-click row to open cell",
                     color=(120, 140, 170, 200))
        dpg.add_separator()

        # Счётчики статусов ячеек
        with dpg.group(horizontal=True):
            dpg.add_text("BAD:",  color=(220, 65,  65,  255))
            dpg.add_text("0", tag="ccnt_red",    color=(220, 65,  65,  255))
            dpg.add_spacer(width=6)
            dpg.add_text("WARN:", color=(220, 175, 45,  255))
            dpg.add_text("0", tag="ccnt_yellow", color=(220, 175, 45,  255))
            dpg.add_spacer(width=6)
            dpg.add_text("OK:",   color=(60,  210, 70,  255))
            dpg.add_text("0", tag="ccnt_green",  color=(60,  210, 70,  255))

        dpg.add_separator()

        with dpg.table(
            tag=CELL_TBL,
            header_row=True,
            sortable=True,
            sort_multi=False,
            sort_tristate=False,
            callback=cell_sort_cb,
            resizable=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            scrollY=True,
            freeze_rows=1,
            height=540,
            width=-1,
        ):
            dpg.add_table_column(label="N",      tag=_ccol_n_uid,
                                 width_fixed=True, init_width_or_weight=36)
            dpg.add_table_column(label="Value",  tag=_ccol_v_uid,
                                 width_fixed=True, init_width_or_weight=55)
            dpg.add_table_column(label="Status", tag=_ccol_s_uid,
                                 width_stretch=True,
                                 default_sort=True,
                                 prefer_sort_ascending=True)

    build_cell_list_rows(fld)


# ════════════════════════════════════════════════════════════════════════════
# ЛОГ (W_LOG)
# ════════════════════════════════════════════════════════════════════════════

LOG_TAG = "log_lb"
_log: List[str] = []

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    _log.append(f"[{ts}] {msg}")
    if len(_log) > 200:
        _log.pop(0)
    if dpg.does_item_exist(LOG_TAG):
        dpg.set_value(LOG_TAG, "\n".join(_log))


# ════════════════════════════════════════════════════════════════════════════
# ЛЕГЕНДА (в Controls)
# ════════════════════════════════════════════════════════════════════════════

LEGEND_DL = "ctrl_legend"
LGW, LGH, LGST = 185, 22, 74

def draw_legend():
    if not dpg.does_item_exist(LEGEND_DL):
        return
    dpg.delete_item(LEGEND_DL, children_only=True)
    cmap = CFG["cmap"]; vmin, vmax = CFG["vmin"], CFG["vmax"]
    for i in range(LGST):
        x0 = int(i * LGW / LGST); x1 = int((i+1)*LGW/LGST)
        t  = i / (LGST-1)
        c  = value_to_color(vmin + t*(vmax-vmin), vmin, vmax, cmap)
        dpg.draw_rectangle((x0,0),(x1,LGH), fill=c, color=(0,0,0,0),
                           parent=LEGEND_DL)


# ════════════════════════════════════════════════════════════════════════════
# REBUILD SCENE (при изменении Settings)
# ════════════════════════════════════════════════════════════════════════════

def rebuild_scene(fld_ref: list, data_q: queue.Queue):
    """
    Пересоздаёт поле (field) после изменения настроек.
    fld_ref — список [Field], чтобы обновить ссылку из замыкания.
    """
    global _order, _current_cell
    new_fld = build_field()
    fld_ref[0] = new_fld

    # Пересобрать таблицу ячеек
    build_cell_list_rows(new_fld)

    # Закрыть detail и таблицу антенн
    _current_cell = None
    _order = []
    _clear_table_rows()            # удалить только строки, не колонки!
    if dpg.does_item_exist("detail_draw"):
        dpg.configure_item("detail_draw", show=False)
    if dpg.does_item_exist("detail_hint"):
        dpg.configure_item("detail_hint", show=True)

    # Сбросить очередь UDP
    while not data_q.empty():
        try: data_q.get_nowait()
        except: pass

    log(f"Scene rebuilt: {len(new_fld.cells)} cells, "
        f"{CFG['n_elems']} antennas, shape={CFG['cell_shape']}")


# ════════════════════════════════════════════════════════════════════════════
# UDP КЛИЕНТ
# ════════════════════════════════════════════════════════════════════════════

class UDPClient:
    def __init__(self, host, port, dq):
        self.host=host; self.port=port; self.q=dq
        self._stop=threading.Event()
        self.pkt_count=0; self.last_pkt_t=0.0
        self.connected=False; self.error_msg=""

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="UDP").start()

    def stop(self): self._stop.set()

    def _run(self):
        try:
            sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.host,self.port)); sock.settimeout(1.0)
            self.connected=True; log(f"UDP {self.host}:{self.port}")
        except OSError as e:
            self.error_msg=str(e); log(f"UDP ERROR: {e}"); return
        while not self._stop.is_set():
            try:
                raw,_=sock.recvfrom(65536)
                data=json.loads(raw.decode())
                self.last_pkt_t=time.perf_counter(); self.pkt_count+=1
                try: self.q.put_nowait(data)
                except queue.Full:
                    try: self.q.get_nowait()
                    except: pass
                    self.q.put_nowait(data)
            except socket.timeout: continue
            except Exception as e: log(f"UDP: {e}")
        sock.close()


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════════════════════════════
_current_cell: Optional[Cell] = None


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    global _current_cell, _sort_col, _sort_dir, _fld_ref_global

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=CFG["udp_port"])
    parser.add_argument("--cmap", default=CFG["cmap"], choices=CMAP_NAMES)
    args = parser.parse_args()
    CFG["udp_port"] = args.port
    CFG["cmap"]     = args.cmap

    fld_ref = [build_field()]   # список — для обновления из closure
    _fld_ref_global = fld_ref   # для callback-ов cell list
    data_q: queue.Queue = queue.Queue(maxsize=8)

    # ── Dear PyGui ────────────────────────────────────────────────────────
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True)

    # ── Тема ─────────────────────────────────────────────────────────────
    with dpg.theme() as dark:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,      (16, 20, 32, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,       (26, 36, 62, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  (40, 58,105, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border,        (48, 72,128, 175))
            dpg.add_theme_color(dpg.mvThemeCol_Text,          (210,220,235, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,       (28, 36, 58, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,(40, 55, 90, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header,        (40, 65,120, 200))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (55, 85,150, 220))
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (38, 60,110, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 85,150, 240))
            dpg.add_theme_color(dpg.mvThemeCol_Tab,           (26, 36, 62, 220))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive,     (45, 70,125, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered,    (55, 85,150, 230))
            dpg.add_theme_color(dpg.mvThemeCol_DockingPreview,(55,120,220, 160))
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (32, 45, 78, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong,(55,80,130,200))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight,(40,60,100,150))
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg,    (20, 26, 42, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (26, 33, 52, 255))

    # ════════════════════════════════════════════════════════════════════
    # MENU BAR
    # ════════════════════════════════════════════════════════════════════
    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Exit", callback=dpg.stop_dearpygui)

        with dpg.menu(label="View"):
            def tog(tag): dpg.configure_item(tag, show=not dpg.is_item_shown(tag))
            dpg.add_menu_item(label="Cell List",
                              callback=lambda: tog(W_CELLS))
            dpg.add_menu_item(label="Controls",
                              callback=lambda: tog(W_CTRL))
            dpg.add_menu_item(label="Antenna Detail",
                              callback=lambda: tog(W_DETAIL))
            dpg.add_menu_item(label="Antenna Table",
                              callback=lambda: tog(W_ANT_TBL))
            dpg.add_menu_item(label="Log",
                              callback=lambda: tog(W_LOG))

        with dpg.menu(label="Sort Antennas"):
            def set_sort(col, direc):
                global _sort_col, _sort_dir
                _sort_col = col; _sort_dir = direc
                if _current_cell:
                    build_antenna_table_rows(_current_cell)
            dpg.add_menu_item(label="N (asc)",
                callback=lambda: set_sort(0, 1))
            dpg.add_menu_item(label="Value (asc)",
                callback=lambda: set_sort(1, 1))
            dpg.add_menu_item(label="[BAD] Bad first (default)",
                callback=lambda: set_sort(2, 1))
            dpg.add_menu_item(label="[OK]  Good first",
                callback=lambda: set_sort(2, -1))

    # ════════════════════════════════════════════════════════════════════
    # LEFT — Cell List (сортируемая таблица с двойным кликом)
    # ════════════════════════════════════════════════════════════════════
    build_cell_list_panel(fld_ref[0])

    # ════════════════════════════════════════════════════════════════════
    # CENTER — Field View
    # ════════════════════════════════════════════════════════════════════
    FW = int(CFG["field_r"] * 2 + 90)
    FH = int(CFG["field_r"] * 2 + 90)

    with dpg.window(label="  Field View", tag=W_FIELD,
                    width=FW, height=FH+22, pos=(196, 20), no_close=True):
        with dpg.drawlist(width=FW-8, height=FH, tag="field_draw"):
            with dpg.draw_layer(tag=FIELD_LAYER):
                pass

    # ════════════════════════════════════════════════════════════════════
    # RIGHT — Antenna Detail (кружки)
    # ════════════════════════════════════════════════════════════════════
    DW = int(DETAIL_W) + 28

    with dpg.window(label="  Antenna Detail", tag=W_DETAIL,
                    width=DW, height=int(DETAIL_H)+52,
                    pos=(196+FW+4, 20), no_close=False):
        dpg.add_text("← Click a cell to view antennas",
                     tag="detail_hint", color=(130,145,170,255))
        with dpg.drawlist(width=int(DETAIL_W)+10, height=int(DETAIL_H)+10,
                          tag="detail_draw", show=False):
            with dpg.draw_layer(tag=DETAIL_LAYER):
                pass

    # ════════════════════════════════════════════════════════════════════
    # RIGHT — Antenna Table (сортируемая)
    # ════════════════════════════════════════════════════════════════════
    build_antenna_table_panel()

    # ════════════════════════════════════════════════════════════════════
    # CONTROLS
    # ════════════════════════════════════════════════════════════════════
    with dpg.window(label="  Controls", tag=W_CTRL,
                    width=230, height=620,
                    pos=(196+FW+DW+270, 20), no_close=False):

        # ── Color Map ────────────────────────────────────────────────
        dpg.add_text("Color Map", color=(130,175,255,255))
        def on_cmap(s,v):
            CFG["cmap"]=v; draw_legend(); log(f"cmap → {v}")
        dpg.add_combo(CMAP_NAMES, default_value=CFG["cmap"],
                      width=-1, callback=on_cmap)
        with dpg.drawlist(width=LGW, height=LGH, tag=LEGEND_DL):
            pass
        dpg.add_text(f"  {CFG['vmin']:.0f}" + " "*10
                     + f"{CFG['vmax']:.0f}", color=(150,150,155,200))
        draw_legend()

        dpg.add_separator()

        # ── Lerp ─────────────────────────────────────────────────────
        dpg.add_text("Animation speed", color=(130,175,255,255))
        def on_lerp(s,v): CFG["lerp_speed"]=v
        dpg.add_slider_float(min_value=0.01, max_value=1.0,
                             default_value=CFG["lerp_speed"],
                             width=-1, callback=on_lerp)
        dpg.add_separator()

        # ── UDP Status ───────────────────────────────────────────────
        dpg.add_text("UDP", color=(130,175,255,255))
        dpg.add_text("waiting...", tag="udp_st", color=(200,200,50,255))
        dpg.add_text("FPS: --",    tag="fps_st",  color=(90,220,90,255))
        dpg.add_separator()

        # ════════════════════════════════════════════════════════════
        # SETTINGS — форма, кол-во, пороги
        # ════════════════════════════════════════════════════════════
        dpg.add_text("⚙  Settings", color=(180,200,255,255))

        dpg.add_text("Cell shape", color=(140,160,200,220))
        _shapes = ["square", "hexagon", "circle"]
        def on_shape(s,v): CFG["cell_shape"] = v
        dpg.add_combo(_shapes, default_value=CFG["cell_shape"],
                      width=-1, callback=on_shape, tag="cfg_shape")

        dpg.add_text("N cells", color=(140,160,200,220))
        def on_ncells(s,v): CFG["n_cells"] = max(1, v)
        dpg.add_input_int(default_value=CFG["n_cells"], min_value=1,
                          max_value=200, width=-1,
                          callback=on_ncells, tag="cfg_ncells")

        dpg.add_text("N antennas", color=(140,160,200,220))
        def on_nelems(s,v):
            CFG["n_elems"] = max(1, v)
        dpg.add_input_int(default_value=CFG["n_elems"], min_value=1,
                          max_value=4096, width=-1,
                          callback=on_nelems, tag="cfg_nelems")

        dpg.add_separator()
        dpg.add_text("Status thresholds", color=(140,160,200,220))

        with dpg.group(horizontal=True):
            dpg.add_text("● Red  <", color=(220,65,65,255))
            def on_thr_red(s,v): CFG["thr_red"] = v
            dpg.add_input_float(default_value=CFG["thr_red"], width=70,
                                min_value=0, max_value=100,
                                callback=on_thr_red, tag="cfg_thr_red",
                                step=0, format="%.0f")

        with dpg.group(horizontal=True):
            dpg.add_text("● Yel <", color=(220,175,45,255))
            def on_thr_yel(s,v): CFG["thr_yellow"] = v
            dpg.add_input_float(default_value=CFG["thr_yellow"], width=70,
                                min_value=0, max_value=100,
                                callback=on_thr_yel, tag="cfg_thr_yel",
                                step=0, format="%.0f")

        dpg.add_separator()

        def on_apply():
            rebuild_scene(fld_ref, data_q)
        dpg.add_button(label="  ▶  Apply & Rebuild  ",
                       callback=on_apply, width=-1)
        dpg.add_text("(Apply rebuilds field)", color=(110,120,140,200))

    # ════════════════════════════════════════════════════════════════════
    # BOTTOM — Log
    # ════════════════════════════════════════════════════════════════════
    with dpg.window(label="  Log", tag=W_LOG,
                    width=1600, height=110, pos=(0, 670), no_close=False):
        dpg.add_input_text(tag=LOG_TAG, multiline=True, readonly=True,
                           width=-1, height=-1, tab_input=False,
                           default_value="")

    # ── Мышь — клик по полю ──────────────────────────────────────────────
    with dpg.handler_registry():
        def on_click(sender, app_data):
            btn = app_data[0] if isinstance(app_data,(list,tuple)) else app_data
            if btn != 0:
                return
            mx, my = dpg.get_mouse_pos(local=False)
            wp = dpg.get_item_pos(W_FIELD)
            lx = mx - wp[0] - 8
            ly = my - wp[1] - 43
            fld = fld_ref[0]
            for cell in fld.cells:
                if point_in_polygon(lx, ly, cell.points):
                    _open_cell_detail(cell)
                    break

        dpg.add_mouse_click_handler(callback=on_click)

    dpg.bind_theme(dark)

    dpg.create_viewport(
        title="GPUWorkLib — Field Viewer  [Docking]",
        width=1640, height=800,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()

    log("Field Viewer started")
    log(f"Cells: {len(fld_ref[0].cells)}, Antennas/cell: {CFG['n_elems']}")
    log("Default sort: ● BAD first | Click column header to resort")

    # ── UDP ──────────────────────────────────────────────────────────────
    udp = UDPClient(CFG["udp_host"], CFG["udp_port"], data_q)
    udp.start()

    # ── Render loop ──────────────────────────────────────────────────────
    frame_int   = 1.0 / CFG["target_fps"]
    t_last      = time.perf_counter()
    fps_frames  = 0; fps_t = time.perf_counter()
    udp_prev    = 0; udp_t = time.perf_counter()
    cell_upd_t  = 0.0   # таблица ячеек — раз в 0.3с
    ant_upd_t   = 0.0   # таблица антенн — раз в 0.25с

    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt  = now - t_last; t_last = now
        fld = fld_ref[0]

        # Данные UDP
        latest = None
        while not data_q.empty():
            try: latest = data_q.get_nowait()
            except: break
        if latest and "cells" in latest:
            fld.apply_update(latest["cells"])

        # Анимация
        for cell in fld.cells:
            cell.update_anim(lerp_speed=CFG["lerp_speed"], dt=dt)

        # ── Рендер поля ──────────────────────────────────────────────
        draw_field(fld)

        # ── Рендер антенн ─────────────────────────────────────────────
        if _current_cell and dpg.is_item_shown("detail_draw"):
            draw_detail(_current_cell)
            dpg.configure_item(W_DETAIL,
                label=f"  Cell {_current_cell.id} · "
                      f"{len(_current_cell.elements)} antennas  "
                      f"[val={_current_cell.current_value:.1f}]")

        # ── Таблица антенн (значения, раз в 0.25с) ────────────────────
        if _current_cell and now - ant_upd_t >= 0.25:
            update_antenna_values(_current_cell)
            update_status_counters(_current_cell)
            ant_upd_t = now

        # ── Таблица ячеек (раз в 0.3с) ────────────────────────────────
        if now - cell_upd_t >= 0.3:
            update_cell_list_values(fld)
            update_cell_status_counters(fld)
            cell_upd_t = now

        # ── FPS ──────────────────────────────────────────────────────
        fps_frames += 1
        if now - fps_t >= 0.5:
            fps = fps_frames / (now - fps_t)
            if dpg.does_item_exist("fps_st"):
                dpg.set_value("fps_st", f"FPS: {fps:.0f}")
            fps_frames = 0; fps_t = now

        # ── UDP статус ───────────────────────────────────────────────
        if now - udp_t >= 1.0:
            udp_t = now
            pkts = udp.pkt_count - udp_prev; udp_prev = udp.pkt_count
            age  = now - udp.last_pkt_t if udp.last_pkt_t > 0 else 9999
            if dpg.does_item_exist("udp_st"):
                if age < 3.0:
                    dpg.set_value("udp_st", f"OK  {pkts} pkt/s")
                    dpg.configure_item("udp_st", color=(90,220,90,255))
                elif udp.connected:
                    dpg.set_value("udp_st", f"idle ({age:.0f}s)")
                    dpg.configure_item("udp_st", color=(210,170,50,255))
                else:
                    dpg.set_value("udp_st", f"ERR: {udp.error_msg}")
                    dpg.configure_item("udp_st", color=(220,70,70,255))

        dpg.render_dearpygui_frame()
        sleep_t = frame_int - (time.perf_counter() - now)
        if sleep_t > 0.001:
            time.sleep(sleep_t)

    udp.stop()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
