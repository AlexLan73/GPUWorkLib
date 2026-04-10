#!/usr/bin/env python3
"""
udp_server_test.py — тестовый UDP-сервер с ДЕМО-СЦЕНАРИЯМИ.

Два сценария:
  «Recovery»    : Всё RED → волна от центра → всё GREEN
  «Random Fill» : Все значения 0, хаотичное заполнение (new > old), все GREEN

Порядок:
  0-40s   : Recovery  (RED → wave → GREEN)
  40-42s  : Сброс в 0 (всё RED мгновенно)
  42-72s  : Random Fill (30s хаотичного заполнения)
  72-85s  : Полное заполнение (всё GREEN, стабильно)
  85s+    : Цикл — снова Recovery

Запуск:
    python udp_server_test.py [--port 5005] [--hz 4] [--cells 100] [--elems 256]
"""
import socket
import json
import time
import math
import argparse
import random
import numpy as np


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


# ════════════════════════════════════════════════════════════════════════════
# СЦЕНАРИЙ 1: Recovery (RED → wave → GREEN)
# ════════════════════════════════════════════════════════════════════════════

def generate_recovery(t: float, n_cells: int, n_elems: int) -> dict:
    """Волна исправления от центра к краям."""
    T_RED_END   = 8.0
    T_WAVE_DUR  = 17.0
    T_GREEN     = T_RED_END + T_WAVE_DUR   # 25s

    RED_BASE, RED_NOISE     = 12.0, 8.0
    GREEN_BASE, GREEN_NOISE = 87.0, 6.0

    cells = {}
    for cid in range(n_cells):
        dist = cid / max(1, n_cells - 1)

        if t < T_RED_END:
            progress = 0.0
        elif t < T_GREEN:
            wave_t = (t - T_RED_END) / T_WAVE_DUR
            wave_front = wave_t * 1.4
            raw = (wave_front - dist) * 5.0
            progress = clamp(1.0 / (1.0 + math.exp(-raw)), 0.0, 1.0)
        else:
            progress = 1.0

        elements = {}
        for eid in range(n_elems):
            ant_off = (eid % 17) * 0.02
            p = clamp(progress - ant_off + random.gauss(0, 0.02))
            red_val   = RED_BASE   + random.gauss(0, RED_NOISE)
            green_val = GREEN_BASE + random.gauss(0, GREEN_NOISE)
            val = red_val + p * (green_val - red_val)
            if t > 35.0:
                val += 3.0 * math.sin(t * 0.7 + cid * 0.3 + eid * 0.1)
            elements[str(eid)] = round(clamp(val), 2)

        cells[str(cid)] = {"elements": elements}
    return {"timestamp": round(t, 4), "cells": cells}


# ════════════════════════════════════════════════════════════════════════════
# СЦЕНАРИЙ 2: Random Fill (0 → хаотичное заполнение → GREEN)
# ════════════════════════════════════════════════════════════════════════════

class RandomFillState:
    """Хранит состояние «Random Fill» — массив значений (monotonic increase)."""

    def __init__(self, n_cells: int, n_elems: int):
        self.n_cells = n_cells
        self.n_elems = n_elems
        self.values = np.zeros((n_cells, n_elems), dtype=np.float64)

    def reset(self):
        self.values[:] = 0.0

    def tick(self, fill_speed: float = 0.15) -> dict:
        """
        Генерирует случайные числа. Обновляет только если new > old.
        fill_speed: доля ячеек, получающих обновление за тик (0..1).
        """
        nc, ne = self.n_cells, self.n_elems

        # Случайные новые значения (0..100)
        new_vals = np.random.uniform(0, 100, size=(nc, ne))

        # Маска: какие ячейки обновляем (не все сразу — для хаотичности)
        update_mask = np.random.random((nc, ne)) < fill_speed

        # Обновляем только если new > old
        improved = (new_vals > self.values) & update_mask
        self.values[improved] = new_vals[improved]

        # Конвертируем в JSON-формат
        cells = {}
        for cid in range(nc):
            elements = {}
            for eid in range(ne):
                elements[str(eid)] = round(float(self.values[cid, eid]), 2)
            cells[str(cid)] = {"elements": elements}

        return cells


def generate_random_fill(state: RandomFillState, t_in_phase: float) -> dict:
    """Random Fill с увеличивающейся скоростью заполнения."""
    # Скорость растёт со временем: 5% → 30% ячеек за тик
    speed = 0.05 + 0.25 * min(1.0, t_in_phase / 25.0)
    cells = state.tick(fill_speed=speed)
    return {"timestamp": round(t_in_phase, 4), "cells": cells}


# ════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР «Всё GREEN» (стабильное)
# ════════════════════════════════════════════════════════════════════════════

def generate_full_green(n_cells: int, n_elems: int, t: float) -> dict:
    cells = {}
    for cid in range(n_cells):
        elements = {}
        for eid in range(n_elems):
            val = 87.0 + random.gauss(0, 4.0)
            val += 2.0 * math.sin(t * 0.5 + cid * 0.2 + eid * 0.05)
            elements[str(eid)] = round(clamp(val), 2)
        cells[str(cid)] = {"elements": elements}
    return {"timestamp": round(t, 4), "cells": cells}


# ════════════════════════════════════════════════════════════════════════════
# MAIN — циклический сценарий
# ════════════════════════════════════════════════════════════════════════════

PHASES = [
    ("RECOVERY",    0,  40),   # 0-40s:  Recovery (red → wave → green)
    ("RANDOM_FILL", 40, 72),   # 40-72s: Random Fill (0 → chaotic → full)
    ("FULL_GREEN",  72, 85),   # 72-85s: Full green (stable)
]
CYCLE_LEN = 85.0  # после 85s → снова с начала


def phase_label(phase_name: str) -> str:
    labels = {
        "RECOVERY":    "Recovery (RED -> wave -> GREEN)",
        "RANDOM_FILL": "Random Fill (0 -> chaotic increase)",
        "FULL_GREEN":  "Full GREEN (stable)",
    }
    return labels.get(phase_name, phase_name)


def main():
    parser = argparse.ArgumentParser(
        description="Demo UDP server: Recovery + Random Fill scenarios")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=5005)
    parser.add_argument("--hz",    type=float, default=4.0)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--elems", type=int, default=256)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / args.hz
    t_start = time.perf_counter()

    # Состояние Random Fill
    rf_state = RandomFillState(args.cells, args.elems)

    print(f"[UDP DEMO] Sending to {args.host}:{args.port}")
    print(f"  Rate: {args.hz} pkt/s  |  Cells: {args.cells}  |  Antennas: {args.elems}")
    print()
    print("  SCENARIO CYCLE ({:.0f}s loop):".format(CYCLE_LEN))
    for name, t0, t1 in PHASES:
        print(f"    {t0:3.0f}-{t1:3.0f}s : {phase_label(name)}")
    print()
    print("  Press Ctrl+C to stop\n")

    CHUNK_SIZE = 16
    pkt_count = 0
    t_report = time.perf_counter()
    prev_phase = ""

    try:
        while True:
            raw_t = time.perf_counter() - t_start
            t_cycle = raw_t % CYCLE_LEN         # время внутри цикла
            cycle_num = int(raw_t // CYCLE_LEN)  # номер цикла

            # Определяем текущую фазу
            phase_name = "FULL_GREEN"
            phase_start = 0.0
            for name, t0, t1 in PHASES:
                if t0 <= t_cycle < t1:
                    phase_name = name
                    phase_start = t0
                    break
            t_in_phase = t_cycle - phase_start

            # Сброс Random Fill при входе в фазу
            if phase_name == "RANDOM_FILL" and prev_phase != "RANDOM_FILL":
                rf_state.reset()

            # Генерируем данные
            if phase_name == "RECOVERY":
                data = generate_recovery(t_in_phase, args.cells, args.elems)
            elif phase_name == "RANDOM_FILL":
                data = generate_random_fill(rf_state, t_in_phase)
            else:  # FULL_GREEN
                data = generate_full_green(args.cells, args.elems, t_in_phase)

            # Логируем смену фазы
            cur_label = f"[cycle {cycle_num}] {phase_name}"
            if cur_label != prev_phase:
                print(f"  [{raw_t:7.1f}s] >>> {phase_label(phase_name)}  "
                      f"(cycle {cycle_num})")
                prev_phase = cur_label

            # Отправка чанками
            cells_data = data.get("cells", data)
            cells_list = list(cells_data.items())
            for i in range(0, len(cells_list), CHUNK_SIZE):
                chunk_cells = dict(cells_list[i:i + CHUNK_SIZE])
                chunk = {
                    "timestamp": data.get("timestamp", round(raw_t, 4)),
                    "chunk": i // CHUNK_SIZE,
                    "cells": chunk_cells,
                }
                payload = json.dumps(chunk, separators=(',', ':')).encode()
                sock.sendto(payload, (args.host, args.port))
                pkt_count += 1

            # Статистика
            now = time.perf_counter()
            if now - t_report >= 10.0:
                rate = pkt_count / (now - t_report)
                fill_pct = ""
                if phase_name == "RANDOM_FILL":
                    filled = np.sum(rf_state.values > 70) / rf_state.values.size * 100
                    fill_pct = f"  filled={filled:.0f}%"
                print(f"  [{raw_t:7.1f}s] {pkt_count} pkts  ~{rate:.1f}/s"
                      f"  chunk={len(payload)}B{fill_pct}")
                pkt_count = 0
                t_report = now

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n[UDP DEMO] Stopped at {time.perf_counter()-t_start:.1f}s")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
