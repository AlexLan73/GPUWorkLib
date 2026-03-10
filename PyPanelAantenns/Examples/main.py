#!/usr/bin/env python3
"""
main.py — Field Viewer  (Dear PyGui + UDP + анимация)
Поле: круг | Ячейки: квадрат x26 | 8 цветовых карт | lerp-анимация
"""
import argparse, math, queue, socket, threading, time, json
from typing import Optional

import dearpygui.dearpygui as dpg

from geometry   import (rect_points, circle_points, circle_grid_centers,
                         point_in_polygon, poly_centroid, scale_polygon,
                         hex_points)
from color_map  import value_to_color, lerp, CMAP_NAMES
from data_models import Field, Cell, Element

# ════════════════════════════════════════════════════════════════════════════
# КОНФИГ
# ════════════════════════════════════════════════════════════════════════════
CFG = dict(
    udp_host   = "0.0.0.0",
    udp_port   = 5005,
    n_cells    = 100,
    n_elems    = 256,          # антенн в ячейке
    cell_side  = 44.0,       # сторона квадратной ячейки (шаг=48, R=330 → 121 мест → берём 100)
    field_r    = 330.0,      # радиус поля
    field_cx   = 395.0,      # центр поля X в drawlist
    field_cy   = 308.0,      # центр поля Y в drawlist
    cmap       = "heat",
    lerp_speed = 2.16,       # ×2 быстрее анимация
    vmin       = 0.0,
    vmax       = 100.0,
    win_w      = 810,
    win_h      = 660,
    target_fps = 60,
)

FIELD_WIN   = "field_window"
FIELD_DRAW  = "field_draw"
FIELD_LAYER = "field_layer"
DETAIL_WIN  = "detail_window"
DETAIL_DRAW = "detail_draw"
DETAIL_LAYER= "detail_layer"
UDP_TAG     = "udp_text"
FPS_TAG     = "fps_text"
STATUS_TAG  = "status_text"
CMAP_COMBO  = "cmap_combo"

# ════════════════════════════════════════════════════════════════════════════
# СБОРКА СЦЕНЫ
# ════════════════════════════════════════════════════════════════════════════

def build_field() -> Field:
    cx0 = CFG["field_cx"]
    cy0 = CFG["field_cy"]
    fr  = CFG["field_r"]
    cs  = CFG["cell_side"]
    n   = CFG["n_cells"]
    ne  = CFG["n_elems"]

    centers = circle_grid_centers(n, cx0, cy0, fr, cs, gap=8.0)

    # Если вдруг поместилось меньше n — добавим ещё по концентрической сетке
    # (фолбэк: просто берём сколько есть)
    cells = []
    for idx, (px, py) in enumerate(centers):
        pts = rect_points(px, py, cs * 0.92, cs * 0.92)
        elems = [Element(id=eid, label=f"e{eid}") for eid in range(ne)]
        for e in elems:
            e.current_value = e.target_value = 50.0
        cell = Cell(id=idx, label=f"C{idx}", points=pts,
                    target_value=50.0, current_value=50.0, elements=elems)
        cell.pulse_phase = idx * 0.41
        cells.append(cell)

    # Граница поля = круг (аппроксимация полигоном)
    border = circle_points(cx0, cy0, fr, n=80)
    return Field(cells=cells, border_poly=border, name="Antenna Panel")


DETAIL_AREA_W = 510.0
DETAIL_AREA_H = 510.0

def build_detail_layout(cell: Cell,
                         area_w: float = DETAIL_AREA_W,
                         area_h: float = DETAIL_AREA_H) -> None:
    """
    Раскладывает элементы ячейки — круглые антенны в detail-окне.
    Для 256 штук — 16×16 сетка, draw_circle через cx/cy/r.
    """
    n = len(cell.elements)
    if n == 0:
        return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw   = area_w / cols
    ch   = area_h / rows
    r    = min(cw, ch) * 0.42        # радиус кружка-антенны

    for idx, elem in enumerate(cell.elements):
        row, col = divmod(idx, cols)
        px = 6 + col * cw + cw / 2
        py = 6 + row * ch + ch / 2
        # Сохраняем центр и радиус (используется draw_circle, не draw_polygon)
        elem._cx = px
        elem._cy = py
        elem._r  = r
        elem.points = []             # не используем полигон для антенн


# ════════════════════════════════════════════════════════════════════════════
# РЕНДЕР ПОЛЯ
# ════════════════════════════════════════════════════════════════════════════

def draw_field(fld: Field, _anim_time: float):
    dpg.delete_item(FIELD_LAYER, children_only=True)

    # Круговая граница поля
    if fld.border_poly:
        dpg.draw_polygon(fld.border_poly,
                         color=(70, 130, 210, 200),
                         fill=(12, 25, 55, 60),
                         thickness=2, parent=FIELD_LAYER)

    cmap = CFG["cmap"]
    vmin, vmax = CFG["vmin"], CFG["vmax"]

    for cell in fld.cells:
        v  = cell.current_value
        bc = value_to_color(v, vmin, vmax, cmap, alpha=215)

        fc = bc
        oc = (180, 180, 180, 140)
        ot = 1

        if cell.is_flashing:
            ft = min(1.0, cell.flash_timer / 0.4)
            fw = int(220 * ft)
            fc = (min(255, bc[0] + fw), min(255, bc[1] + fw),
                  min(255, bc[2] + fw), bc[3])
            oc = (255, 255, 255, int(220 * ft))
            ot = 2

        dpg.draw_polygon(cell.points, color=oc, fill=fc,
                         thickness=ot, parent=FIELD_LAYER)

        px, py = poly_centroid(cell.points)
        dpg.draw_text((px - 14, py - 9), cell.label,
                      color=(255, 255, 255, 220), size=12, parent=FIELD_LAYER)
        dpg.draw_text((px - 14, py + 2), f"{v:.1f}",
                      color=(225, 225, 225, 185), size=11, parent=FIELD_LAYER)


# ════════════════════════════════════════════════════════════════════════════
# ДЕТАЛЬНЫЙ ВИД
# ════════════════════════════════════════════════════════════════════════════

def _draw_elements(cell: Cell):
    """Рисует антенны ячейки кружками. При 256 штуках — без подписей."""
    dpg.delete_item(DETAIL_LAYER, children_only=True)
    cmap = CFG["cmap"]
    vmin, vmax = CFG["vmin"], CFG["vmax"]
    n = len(cell.elements)
    show_label = n <= 36          # подписи только если мало антенн

    for elem in cell.elements:
        cx = getattr(elem, "_cx", None)
        cy = getattr(elem, "_cy", None)
        r  = getattr(elem, "_r",  None)
        if cx is None:
            continue
        v    = elem.current_value
        fill = value_to_color(v, vmin, vmax, cmap, alpha=230)
        # Яркая обводка — чуть светлее заливки
        outline = (min(255, fill[0] + 50),
                   min(255, fill[1] + 50),
                   min(255, fill[2] + 50), 200)

        dpg.draw_circle((cx, cy), r,
                        color=outline, fill=fill,
                        thickness=1, parent=DETAIL_LAYER)

        # Маленькая точка в центре — маркер
        if r > 8:
            dpg.draw_circle((cx, cy), 2,
                            color=(255, 255, 255, 160), fill=(255, 255, 255, 120),
                            thickness=0, parent=DETAIL_LAYER)

        if show_label:
            dpg.draw_text((cx - 10, cy - 7), elem.label,
                          color=(255, 255, 255, 210), size=10, parent=DETAIL_LAYER)
            dpg.draw_text((cx - 12, cy + 1), f"{v:.1f}",
                          color=(225, 225, 225, 175), size=9, parent=DETAIL_LAYER)


def open_detail(cell: Cell):
    build_detail_layout(cell)
    if dpg.does_item_exist(DETAIL_WIN):
        dpg.delete_item(DETAIL_WIN)
    dw = int(DETAIL_AREA_W) + 30
    dh = int(DETAIL_AREA_H) + 50
    with dpg.window(label=f"Cell {cell.id} — {cell.label}  "
                          f"[{cell.current_value:.1f}]  ·  {len(cell.elements)} antennas",
                    tag=DETAIL_WIN, width=dw, height=dh, pos=(830, 60)):
        with dpg.drawlist(width=int(DETAIL_AREA_W) + 12,
                          height=int(DETAIL_AREA_H) + 12,
                          tag=DETAIL_DRAW):
            with dpg.draw_layer(tag=DETAIL_LAYER):
                pass


def refresh_detail(cell: Cell):
    if dpg.does_item_exist(DETAIL_WIN):
        _draw_elements(cell)
        dpg.configure_item(DETAIL_WIN,
                           label=f"Cell {cell.id} — {cell.label}  [{cell.current_value:.1f}]")


# ════════════════════════════════════════════════════════════════════════════
# ЛЕГЕНДА — 8 карт, градиентная полоска
# ════════════════════════════════════════════════════════════════════════════

LEGEND_DRAW = "legend_bar"
LEGEND_W = 210
LEGEND_H = 26
LEGEND_STEPS = 90

def rebuild_legend():
    if not dpg.does_item_exist(LEGEND_DRAW):
        return
    dpg.delete_item(LEGEND_DRAW, children_only=True)
    cmap = CFG["cmap"]
    vmin, vmax = CFG["vmin"], CFG["vmax"]
    for i in range(LEGEND_STEPS):
        x0 = int(i * LEGEND_W / LEGEND_STEPS)
        x1 = int((i + 1) * LEGEND_W / LEGEND_STEPS)
        t  = i / (LEGEND_STEPS - 1)
        v  = vmin + t * (vmax - vmin)
        c  = value_to_color(v, vmin, vmax, cmap)
        dpg.draw_rectangle((x0, 0), (x1, LEGEND_H),
                            fill=c, color=(0, 0, 0, 0), parent=LEGEND_DRAW)


def build_legend_window():
    with dpg.window(label="Color scale", width=260, height=110,
                    pos=(840, 10), no_resize=True):
        with dpg.drawlist(width=LEGEND_W, height=LEGEND_H, tag=LEGEND_DRAW):
            pass   # заполним в rebuild_legend()

        dpg.add_text(f"  {CFG['vmin']:.0f}" + " " * 16
                     + f"{(CFG['vmin']+CFG['vmax'])/2:.0f}"
                     + " " * 14 + f"{CFG['vmax']:.0f}")

        def on_cmap_change(sender, app_data):
            CFG["cmap"] = app_data
            rebuild_legend()

        dpg.add_combo(CMAP_NAMES, default_value=CFG["cmap"],
                      tag=CMAP_COMBO, width=LEGEND_W,
                      label="", callback=on_cmap_change)

    rebuild_legend()


# ════════════════════════════════════════════════════════════════════════════
# UDP КЛИЕНТ
# ════════════════════════════════════════════════════════════════════════════

class UDPClient:
    def __init__(self, host: str, port: int, dq: queue.Queue):
        self.host = host; self.port = port; self.q = dq
        self._stop = threading.Event()
        self.pkt_count = 0; self.last_pkt_t = 0.0
        self.connected = False; self.error_msg = ""

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="UDP").start()

    def stop(self): self._stop.set()

    def _run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.host, self.port))
            sock.settimeout(1.0)
            self.connected = True
            print(f"[UDP] Listening {self.host}:{self.port}")
        except OSError as e:
            self.error_msg = str(e); print(f"[UDP] Error: {e}"); return

        while not self._stop.is_set():
            try:
                raw, _ = sock.recvfrom(65536)
                data = json.loads(raw.decode())
                self.last_pkt_t = time.perf_counter()
                self.pkt_count += 1
                try:
                    self.q.put_nowait(data)
                except queue.Full:
                    try: self.q.get_nowait()
                    except queue.Empty: pass
                    self.q.put_nowait(data)
            except socket.timeout:
                continue
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[UDP] Parse error: {e}")

        sock.close(); self.connected = False


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int,   default=CFG["udp_port"])
    parser.add_argument("--cmap", default=CFG["cmap"], choices=CMAP_NAMES)
    parser.add_argument("--fps",  type=float, default=CFG["target_fps"])
    args = parser.parse_args()
    CFG["udp_port"] = args.port
    CFG["cmap"]     = args.cmap
    CFG["target_fps"] = args.fps

    fld = build_field()
    data_q: queue.Queue = queue.Queue(maxsize=8)
    selected_cell: Optional[Cell] = None

    # ── Dear PyGui ────────────────────────────────────────────────────────
    dpg.create_context()

    with dpg.theme() as dark_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,     (16, 20, 30, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,      (28, 38, 65, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,(42, 60, 105, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border,       (50, 75, 130, 180))
            dpg.add_theme_color(dpg.mvThemeCol_Text,         (210, 220, 235, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,      (30, 38, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,(42,55,90,255))

    # Главное окно поля
    with dpg.window(label="Antenna Panel — Field View", tag=FIELD_WIN,
                    width=CFG["win_w"], height=CFG["win_h"],
                    pos=(0, 0), no_close=True):

        with dpg.group(horizontal=True):
            dpg.add_text("UDP:", color=(130, 180, 255, 255))
            dpg.add_text("waiting...", tag=UDP_TAG, color=(200, 200, 50, 255))
            dpg.add_spacer(width=25)
            dpg.add_text("FPS:", color=(130, 180, 255, 255))
            dpg.add_text("--", tag=FPS_TAG, color=(90, 220, 90, 255))
            dpg.add_spacer(width=25)
            dpg.add_text("", tag=STATUS_TAG, color=(150, 155, 180, 255))

        with dpg.drawlist(width=CFG["win_w"] - 16,
                          height=CFG["win_h"] - 56,
                          tag=FIELD_DRAW):
            with dpg.draw_layer(tag=FIELD_LAYER):
                pass

    build_legend_window()

    # ── Мышь ─────────────────────────────────────────────────────────────
    with dpg.handler_registry():
        def on_click(sender, app_data):
            nonlocal selected_cell
            btn = app_data[0] if isinstance(app_data, (list, tuple)) else app_data
            if btn != 0:
                return
            mx, my = dpg.get_mouse_pos(local=False)
            wp = dpg.get_item_pos(FIELD_WIN)
            lx = mx - wp[0] - 8
            ly = my - wp[1] - 43   # заголовок ~19 + статус-бар ~24
            for cell in fld.cells:
                if point_in_polygon(lx, ly, cell.points):
                    selected_cell = cell
                    open_detail(cell)
                    dpg.set_value(STATUS_TAG,
                                  f"Selected: {cell.label}  val={cell.current_value:.1f}")
                    break

        dpg.add_mouse_click_handler(callback=on_click)

    dpg.bind_theme(dark_theme)
    dpg.create_viewport(
        title=f"GPUWorkLib — Field Viewer  [port:{CFG['udp_port']}]",
        width=1340, height=720,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # ── UDP ───────────────────────────────────────────────────────────────
    udp = UDPClient(CFG["udp_host"], CFG["udp_port"], data_q)
    udp.start()

    # ── Render loop ───────────────────────────────────────────────────────
    frame_int   = 1.0 / CFG["target_fps"]
    t_last      = time.perf_counter()
    anim_time   = 0.0
    fps_frames  = 0
    fps_t       = time.perf_counter()
    udp_prev    = 0
    udp_rep_t   = time.perf_counter()

    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt  = now - t_last
        t_last = now
        anim_time += dt

        # Данные из UDP
        latest = None
        while not data_q.empty():
            try: latest = data_q.get_nowait()
            except queue.Empty: break

        if latest and "cells" in latest:
            fld.apply_update(latest["cells"])

        # Анимация (lerp каждый кадр)
        for cell in fld.cells:
            cell.update_anim(lerp_speed=CFG["lerp_speed"], dt=dt)

        # Рендер
        draw_field(fld, anim_time)
        if selected_cell and dpg.does_item_exist(DETAIL_WIN):
            refresh_detail(selected_cell)

        # FPS
        fps_frames += 1
        if now - fps_t >= 0.5:
            dpg.set_value(FPS_TAG, f"{fps_frames / (now - fps_t):.0f}")
            fps_frames = 0; fps_t = now

        # UDP статус
        if now - udp_rep_t >= 1.0:
            udp_rep_t = now
            new_pkts  = udp.pkt_count - udp_prev
            udp_prev  = udp.pkt_count
            age = now - udp.last_pkt_t if udp.last_pkt_t > 0 else 9999
            if age < 3.0:
                dpg.set_value(UDP_TAG, f"OK  {new_pkts} pkt/s")
                dpg.configure_item(UDP_TAG, color=(90, 220, 90, 255))
            elif udp.connected:
                dpg.set_value(UDP_TAG, f"idle ({age:.0f}s)")
                dpg.configure_item(UDP_TAG, color=(210, 170, 50, 255))
            else:
                dpg.set_value(UDP_TAG, f"ERR: {udp.error_msg or 'no bind'}")
                dpg.configure_item(UDP_TAG, color=(220, 70, 70, 255))

        dpg.render_dearpygui_frame()

        sleep_t = frame_int - (time.perf_counter() - now)
        if sleep_t > 0.001:
            time.sleep(sleep_t)

    udp.stop()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
