# color_map.py — 8 цветовых карт + плавная анимация (lerp)
import math
from typing import Tuple

RGBA = Tuple[int, int, int, int]


# ── 8 цветовых карт ───────────────────────────────────────────────────────────

def cmap_heat(t: float, alpha: int = 220) -> RGBA:
    """Красный → Жёлтый → Зелёный  (low=BAD=red, high=OK=green)."""
    t = max(0.0, min(1.0, t))
    r = int(255 * min(1.0, 2.0 * (1.0 - t)))
    g = int(255 * min(1.0, 2.0 * t))
    return (r, g, 30, alpha)


def cmap_cool(t: float, alpha: int = 220) -> RGBA:
    """Синий → Голубой → Белый."""
    t = max(0.0, min(1.0, t))
    r = int(t * 200)
    g = int(t * 230)
    return (r, g, 255, alpha)


def cmap_plasma(t: float, alpha: int = 220) -> RGBA:
    """Синий → Пурпурный → Жёлтый."""
    t = max(0.0, min(1.0, t))
    r = int(255 * (0.5 + 0.5 * math.sin(math.pi * t - 0.5)))
    g = int(255 * t * t)
    b = int(255 * (1.0 - t))
    return (r, g, b, alpha)


def cmap_radar(t: float, alpha: int = 220) -> RGBA:
    """Чёрный → Зелёный → Белый (радар)."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        v = int(t * 2 * 180)
        return (0, v, 0, alpha)
    v = int((t - 0.5) * 2 * 255)
    return (v, 180 + v // 4, v, alpha)


def cmap_viridis(t: float, alpha: int = 220) -> RGBA:
    """Тёмно-фиолетовый → Зелёный → Жёлтый (viridis)."""
    t = max(0.0, min(1.0, t))
    r = int(255 * (0.267 + 0.699 * t - 0.109 * t * t))
    g = int(255 * (0.005 + 1.086 * t - 0.310 * t * t))
    b = int(255 * (0.329 + 0.175 * t - 0.504 * t * t))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), alpha)


def cmap_magma(t: float, alpha: int = 220) -> RGBA:
    """Чёрный → Пурпурный → Оранжевый → Белый."""
    t = max(0.0, min(1.0, t))
    r = int(255 * min(1.0, 1.6 * t))
    g = int(255 * max(0.0, 2.0 * t - 1.0))
    b = int(255 * (0.3 * math.sin(math.pi * t) + max(0.0, 2.0 * t - 1.5)))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), alpha)


def cmap_ocean(t: float, alpha: int = 220) -> RGBA:
    """Тёмно-синий → Голубой → Морской (ocean)."""
    t = max(0.0, min(1.0, t))
    r = int(t * 80)
    g = int(80 + t * 175)
    b = int(140 + t * 115)
    return (r, g, min(255, b), alpha)


def cmap_inferno(t: float, alpha: int = 220) -> RGBA:
    """Чёрный → Тёмно-красный → Оранжевый → Жёлтый (inferno)."""
    t = max(0.0, min(1.0, t))
    r = int(255 * min(1.0, 2.5 * t * t))
    g = int(255 * max(0.0, 1.7 * t - 0.6))
    b = int(255 * max(0.0, 0.6 * math.sin(math.pi * t * 0.7)))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), alpha)


# ── Реестр карт ──────────────────────────────────────────────────────────────

CMAPS = {
    "heat":    cmap_heat,
    "cool":    cmap_cool,
    "plasma":  cmap_plasma,
    "radar":   cmap_radar,
    "viridis": cmap_viridis,
    "magma":   cmap_magma,
    "ocean":   cmap_ocean,
    "inferno": cmap_inferno,
}

CMAP_NAMES = list(CMAPS.keys())   # для UI-списка


def value_to_color(value: float,
                   vmin: float = 0.0,
                   vmax: float = 100.0,
                   cmap: str = "heat",
                   alpha: int = 220) -> RGBA:
    """Нормализует value → [0,1] → цвет по выбранной карте."""
    t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    return CMAPS.get(cmap, cmap_heat)(t, alpha)


# ── Анимация ─────────────────────────────────────────────────────────────────

def lerp(a: float, b: float, speed: float) -> float:
    """Экспоненциальный lerp — плавное приближение к цели."""
    return a + (b - a) * min(1.0, speed)


def lerp_color(c_from: RGBA, c_to: RGBA, speed: float) -> RGBA:
    return tuple(int(lerp(cf, ct, speed)) for cf, ct in zip(c_from, c_to))


def pulse_alpha(base_alpha: int, phase: float, amplitude: int = 60) -> int:
    return max(0, min(255, base_alpha + int(amplitude * math.sin(phase))))


def glow_color(base_color: RGBA, phase: float) -> RGBA:
    factor = 0.75 + 0.25 * math.sin(phase)
    r = min(255, int(base_color[0] * factor + 60 * (1 - factor)))
    g = min(255, int(base_color[1] * factor + 60 * (1 - factor)))
    b = min(255, int(base_color[2] * factor + 60 * (1 - factor)))
    return (r, g, b, base_color[3])
