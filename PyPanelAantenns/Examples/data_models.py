# data_models.py — структуры данных для поля, ячеек и элементов
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from color_map import RGBA, lerp

Point = Tuple[float, float]
Polygon = List[Point]


@dataclass
class Element:
    """Один элемент внутри ячейки."""
    id:        int
    label:     str       = ""
    points:    Polygon   = field(default_factory=list)  # координаты в окне деталей

    # Значение и его анимированная версия
    target_value:  float = 0.0
    current_value: float = 0.0   # плавно догоняет target

    def update_anim(self, lerp_speed: float = 0.12):
        self.current_value = lerp(self.current_value, self.target_value, lerp_speed)

    @property
    def value(self) -> float:
        return self.target_value

    @value.setter
    def value(self, v: float):
        self.target_value = v


@dataclass
class Cell:
    """Ячейка на поле."""
    id:       int
    label:    str      = ""
    points:   Polygon  = field(default_factory=list)   # координаты в окне поля

    # Значение и анимация
    target_value:  float = 0.0
    current_value: float = 0.0

    elements: List[Element] = field(default_factory=list)

    # Состояние анимации
    pulse_phase: float = 0.0     # фаза пульсации при обновлении
    flash_timer: float = 0.0     # секунды «вспышки» при резком изменении

    def update_anim(self, lerp_speed: float = 0.10, dt: float = 0.033):
        """Обновить анимацию (вызывать каждый кадр)."""
        prev = self.current_value
        self.current_value = lerp(self.current_value, self.target_value, lerp_speed)

        # Вспышка при резком изменении (>10 единиц за кадр)
        if abs(self.target_value - prev) > 5.0:
            self.flash_timer = 0.4   # 400 мс

        if self.flash_timer > 0:
            self.flash_timer -= dt
        self.pulse_phase += dt * 4.0  # скорость пульсации

        for elem in self.elements:
            elem.update_anim(lerp_speed)

    @property
    def value(self) -> float:
        return self.target_value

    @value.setter
    def value(self, v: float):
        self.target_value = v

    @property
    def is_flashing(self) -> bool:
        return self.flash_timer > 0


@dataclass
class Field:
    """Всё поле целиком."""
    cells:       List[Cell] = field(default_factory=list)
    border_poly: Polygon    = field(default_factory=list)
    name:        str        = "Field"

    def cell_by_id(self, cell_id: int) -> Optional[Cell]:
        for c in self.cells:
            if c.id == cell_id:
                return c
        return None

    def apply_update(self, data: dict):
        """
        data = {
            cell_id(int or str): {
                "value": float,          # опционально — если нет, считаем среднее
                "elements": { elem_id(int or str): float, ... }
            },
            ...
        }
        Значение ячейки (cell.value) = среднее всех антенн.
        Поле "value" из JSON игнорируется — главная панель показывает
        среднее по антеннам, так корректнее.
        """
        for raw_cid, cdata in data.items():
            cell = self.cell_by_id(int(raw_cid))
            if cell is None:
                continue
            # Сначала обновляем антенны
            elem_vals = cdata.get("elements", {})
            for elem in cell.elements:
                if str(elem.id) in elem_vals:
                    elem.value = float(elem_vals[str(elem.id)])
                elif elem.id in elem_vals:
                    elem.value = float(elem_vals[elem.id])
            # Значение ячейки = среднее по всем антеннам
            if cell.elements:
                cell.value = sum(e.target_value for e in cell.elements) / len(cell.elements)
