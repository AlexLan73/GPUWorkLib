"""
llm_parser.py — LLMParser (Strategy pattern)
=============================================

Strategy (GoF) + OCP (SOLID):
  LLMParser   — абстрактный парсер NL → FilterSpec
  GroqParser  — реализация через Groq API
  OllamaParser — реализация через Ollama
  MockParser  — детерминированный парсер (regex), не требует LLM

Позволяет тестировать pipeline без AI-бэкенда:
    parser = MockParser()  ← никаких сетевых запросов
    spec = parser.parse("FIR lowpass 1kHz, fs=50kHz")

Usage:
    # В тестах:
    parser = MockParser()
    spec = parser.parse("butterworth lowpass 1kHz, fs=50kHz", fs=50_000)

    # В production:
    parser = GroqParser(api_key="gsk_...", model="llama-3.3-70b-versatile")
    spec = parser.parse("Нужен полосовой IIR фильтр от 1 до 5 кГц при Fs=50кГц")
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Optional


# ─────────────────────────────────────────────────────────────────────────────
# FilterSpec — результат парсинга (Value Object)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterSpec:
    """Спецификация фильтра, извлечённая из NL-запроса.

    Attributes:
        filter_class:  "fir" | "iir"
        filter_type:   "lowpass" | "highpass" | "bandpass" | "bandstop"
        f_cutoff:      частота среза Гц (или [f_low, f_high] для полосовых)
        fs:            частота дискретизации Гц
        order:         порядок фильтра (0 = авто для FIR)
        window:        окно для FIR
        ripple_db:     затухание в полосе задерживания (дБ)
        description:   текстовое описание от парсера
    """
    filter_class: str = "iir"
    filter_type: str = "lowpass"
    f_cutoff: Union[float, list] = 1000.0
    fs: float = 50_000.0
    order: int = 4
    window: str = "kaiser"
    ripple_db: float = 60.0
    description: str = ""

    def normalized_cutoff(self) -> Union[float, list]:
        """Нормированная частота среза (0..1, Nyquist=1)."""
        nyq = self.fs / 2.0
        if isinstance(self.f_cutoff, list):
            return [f / nyq for f in self.f_cutoff]
        return self.f_cutoff / nyq


# ─────────────────────────────────────────────────────────────────────────────
# Системный промпт для LLM
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a DSP expert. Parse the user's filter description into JSON parameters.
Return ONLY valid JSON, nothing else.

The JSON must have these fields:
{
  "filter_class": "fir" or "iir",
  "filter_type": "lowpass" | "highpass" | "bandpass" | "bandstop",
  "f_cutoff": <Hz, single number for LP/HP, or [f_low, f_high] for BP/BS>,
  "fs": <sample rate Hz>,
  "order": <integer (IIR: 2-10, FIR: 0=auto or 32-512)>,
  "window": "kaiser" | "hamming" | "hann" | "blackman",
  "ripple_db": <stopband attenuation dB, default 60>,
  "description": "<brief description>"
}

Rules:
- "butterworth", "biquad", "IIR", "recursive" -> filter_class = "iir"
- "FIR", "convolution", "taps", "window" -> filter_class = "fir"
- default: "iir" for LP/HP, "fir" for complex shapes
- All frequencies in Hz
- IIR default order=4, FIR order=0 (auto)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Извлечь JSON из ответа LLM (может быть в ```json блоке)."""
    if not text:
        return {}
    m = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Cannot find JSON in LLM response:\n{text[:300]}")


def _dict_to_spec(d: dict, default_fs: float = 50_000.0) -> FilterSpec:
    """Преобразовать dict (от LLM) в FilterSpec."""
    return FilterSpec(
        filter_class=d.get("filter_class", "iir"),
        filter_type=d.get("filter_type", "lowpass"),
        f_cutoff=d.get("f_cutoff", default_fs / 10),
        fs=d.get("fs", default_fs),
        order=d.get("order", 4),
        window=d.get("window", "kaiser"),
        ripple_db=d.get("ripple_db", 60.0),
        description=d.get("description", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLMParser — абстрактный интерфейс (Strategy)
# ─────────────────────────────────────────────────────────────────────────────

class LLMParser(ABC):
    """Абстрактный парсер NL-запроса → FilterSpec.

    Strategy (GoF): подкласс определяет как именно выполнять парсинг.
    OCP (SOLID): добавить новый бэкенд = новый подкласс, без изменения кода.
    """

    @abstractmethod
    def parse(self, text: str, fs: float = 50_000.0) -> FilterSpec:
        """Распарсить NL-описание фильтра.

        Args:
            text: описание фильтра на естественном языке
            fs:   частота дискретизации (Гц) — подсказка для LLM

        Returns:
            FilterSpec с параметрами фильтра
        """
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Название бэкенда для логов."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# MockParser — детерминированный парсер (regex, без LLM)
# ─────────────────────────────────────────────────────────────────────────────

class MockParser(LLMParser):
    """Regex-парсер без LLM — для тестов и демо.

    Понимает ключевые слова: butterworth, fir, lowpass, highpass,
    bandpass, bandstop, частоты в Гц/кГц, порядок.
    """

    @property
    def backend_name(self) -> str:
        return "mock"

    def parse(self, text: str, fs: float = 50_000.0) -> FilterSpec:
        t = text.lower()

        # Тип класса
        if any(w in t for w in ['iir', 'butterworth', 'biquad', 'chebyshev', 'recursive']):
            fclass = "iir"
        elif any(w in t for w in ['fir', 'taps', 'convolution', 'window']):
            fclass = "fir"
        else:
            fclass = "iir"

        # Тип фильтра
        if any(w in t for w in ['bandpass', 'band-pass', 'полосовой пропускания']):
            ftype = "bandpass"
        elif any(w in t for w in ['bandstop', 'band-stop', 'notch', 'режекторный']):
            ftype = "bandstop"
        elif any(w in t for w in ['highpass', 'high-pass', 'фвч', 'высокочастотный']):
            ftype = "highpass"
        else:
            ftype = "lowpass"

        # Частоты
        freqs_khz = re.findall(r'(\d+(?:\.\d+)?)\s*khz', t)
        freqs_hz = re.findall(r'(\d+(?:\.\d+)?)\s*hz', t)
        freqs = [float(f) * 1000 for f in freqs_khz] + [float(f) for f in freqs_hz]

        if ftype in ("bandpass", "bandstop") and len(freqs) >= 2:
            f_cutoff = freqs[:2]
        elif freqs:
            f_cutoff = freqs[0]
        else:
            f_cutoff = fs / 10.0  # default 10% от fs

        # Порядок
        m = re.search(r'(?:order|порядок)\s*[=:]?\s*(\d+)', t)
        order = int(m.group(1)) if m else (4 if fclass == "iir" else 0)

        return FilterSpec(
            filter_class=fclass,
            filter_type=ftype,
            f_cutoff=f_cutoff,
            fs=fs,
            order=order,
            description=f"{fclass.upper()} {ftype} fc={f_cutoff} Hz (mock)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GroqParser — Groq Cloud API
# ─────────────────────────────────────────────────────────────────────────────

class GroqParser(LLMParser):
    """Парсер через Groq Cloud API.

    Требует: pip install groq
    API key: из api_keys.json или переменной окружения GROQ_API_KEY
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "llama-3.3-70b-versatile"):
        """
        Args:
            api_key: Groq API key (None = читать из api_keys.json / env)
            model:   идентификатор модели
        """
        self.model = model
        self._api_key = api_key or self._load_api_key()

    @property
    def backend_name(self) -> str:
        return f"groq/{self.model}"

    @staticmethod
    def _load_api_key() -> str:
        """Загрузить ключ из api_keys.json или переменной окружения."""
        # 1. Переменная окружения
        key = os.environ.get("GROQ_API_KEY", "")
        if key:
            return key
        # 2. api_keys.json в корне проекта
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(root, "api_keys.json")
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("api", "")
        return ""

    def parse(self, text: str, fs: float = 50_000.0) -> FilterSpec:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")

        if not self._api_key:
            raise RuntimeError(
                "Groq API key not set. "
                "Create api_keys.json or set GROQ_API_KEY env variable."
            )

        client = Groq(api_key=self._api_key)
        prompt = f'Filter description: "{text}"\nSample rate: {fs} Hz\nReturn JSON.'
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.05,
            max_tokens=512,
        )
        raw = response.choices[0].message.content
        return _dict_to_spec(_extract_json(raw), default_fs=fs)


# ─────────────────────────────────────────────────────────────────────────────
# OllamaParser — локальный Ollama
# ─────────────────────────────────────────────────────────────────────────────

class OllamaParser(LLMParser):
    """Парсер через локальный Ollama.

    Требует: pip install ollama + запущенный ollama serve
    Рекомендуемая модель: qwen2.5-coder:7b
    """

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model

    @property
    def backend_name(self) -> str:
        return f"ollama/{self.model}"

    def parse(self, text: str, fs: float = 50_000.0) -> FilterSpec:
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")

        prompt = f'Filter description: "{text}"\nSample rate: {fs} Hz\nReturn JSON.'
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.05},
        )
        raw = response["message"]["content"]
        return _dict_to_spec(_extract_json(raw), default_fs=fs)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_parser(mode: str = "mock", **kwargs) -> LLMParser:
    """Фабричная функция — создать парсер по строковому имени.

    Args:
        mode: "mock" | "groq" | "ollama"
        **kwargs: дополнительные аргументы для парсера

    Returns:
        LLMParser подходящего типа
    """
    if mode == "groq":
        return GroqParser(**kwargs)
    if mode == "ollama":
        return OllamaParser(**kwargs)
    return MockParser()
