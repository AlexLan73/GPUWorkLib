# geometry.py — геометрические примитивы и hit-testing
import math
from typing import List, Tuple, Optional

Point = Tuple[float, float]
Polygon = List[Point]


def hex_points(cx: float, cy: float, r: float) -> Polygon:
    """Правильный шестиугольник с плоским верхом (flat-top)."""
    return [
        (cx + r * math.cos(math.radians(60 * i)),
         cy + r * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def rect_points(cx: float, cy: float, w: float, h: float) -> Polygon:
    """Прямоугольник по центру."""
    hw, hh = w / 2, h / 2
    return [(cx - hw, cy - hh), (cx + hw, cy - hh),
            (cx + hw, cy + hh), (cx - hw, cy + hh)]


def circle_points(cx: float, cy: float, r: float, n: int = 36) -> Polygon:
    """Аппроксимация круга полигоном."""
    return [
        (cx + r * math.cos(2 * math.pi * i / n),
         cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def diamond_points(cx: float, cy: float, r: float) -> Polygon:
    """Ромб."""
    return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]


def point_in_polygon(px: float, py: float, poly: Polygon) -> bool:
    """Ray casting алгоритм — точка внутри полигона."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def poly_centroid(poly: Polygon) -> Point:
    """Центр масс полигона."""
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def hex_grid_centers(cols: int, rows: int, r: float,
                     offset_x: float = 0.0, offset_y: float = 0.0,
                     pointy_top: bool = False) -> List[Point]:
    """
    Генерация центров hex-сетки.
    pointy_top=False → flat-top (шестиугольник "лежит" на боку).
    """
    centers = []
    if not pointy_top:
        # flat-top hex grid
        dx = r * math.sqrt(3)
        dy = r * 1.5
        for row in range(rows):
            for col in range(cols):
                cx = offset_x + col * dx + (dx / 2 if row % 2 else 0.0)
                cy = offset_y + row * dy
                centers.append((cx, cy))
    else:
        # pointy-top hex grid
        dx = r * 1.5
        dy = r * math.sqrt(3)
        for row in range(rows):
            for col in range(cols):
                cx = offset_x + col * dx
                cy = offset_y + row * dy + (dy / 2 if col % 2 else 0.0)
                centers.append((cx, cy))
    return centers


def scale_polygon(poly: Polygon, factor: float) -> Polygon:
    """Масштабирование полигона вокруг его центра."""
    cx, cy = poly_centroid(poly)
    return [(cx + (p[0] - cx) * factor, cy + (p[1] - cy) * factor) for p in poly]


def translate_polygon(poly: Polygon, dx: float, dy: float) -> Polygon:
    return [(p[0] + dx, p[1] + dy) for p in poly]


def circle_grid_centers(n_cells: int,
                         field_cx: float, field_cy: float,
                         field_r: float,
                         cell_side: float,
                         gap: float = 6.0) -> List[Point]:
    """
    Размещает n_cells центров квадратных ячеек внутри круга поля.
    Строит сетку, сортирует по расстоянию от центра, берёт n_cells ближайших.
    """
    step = cell_side + gap
    half = cell_side / 2.0

    # Диапазон сетки с запасом
    n = int(math.ceil(field_r / step)) + 1
    candidates: List[Tuple[float, float, float]] = []  # (dist, cx, cy)

    for row in range(-n, n + 1):
        for col in range(-n, n + 1):
            cx = field_cx + col * step
            cy = field_cy + row * step
            dist = math.hypot(cx - field_cx, cy - field_cy)
            # Ячейка целиком внутри круга
            if dist + math.hypot(half, half) <= field_r - 2:
                candidates.append((dist, cx, cy))

    # Сортируем от центра наружу
    candidates.sort(key=lambda c: c[0])
    return [(c[1], c[2]) for c in candidates[:n_cells]]
