# 🎨 Mermaid — Стилизация диаграмм для VS Code Dark Theme

> **Проверено**: Mermaid 10.9.5 (VS Code extension "Markdown Preview Mermaid Support")
> **Дата**: 2026-03-21
> **Источник**: Context7 docs + эксперименты GPUWorkLib

---

## ⚠️ Ключевые правила

1. **Frontmatter `---config---` НЕ работает** в Mermaid 10.9.5 → используй `%%{init}%%`
2. **C4 диаграммы** (C4Context, C4Container, C4Component) **игнорируют `themeVariables`** → стилизация ТОЛЬКО через `UpdateElementStyle` + `UpdateRelStyle`
3. **flowchart / graph / classDiagram / sequenceDiagram** — `themeVariables` работают нормально
4. **Связи по умолчанию тёмные** → на dark theme не видны! Всегда задавай `lineColor` или `linkStyle`

---

## 📋 Палитра для Dark Theme

| Элемент | Цвет | Hex | Назначение |
|---------|------|-----|-----------|
| Связи / линии | Светло-голубой | `#90caf9` | Хорошо виден на тёмном фоне |
| Текст на связях | Светло-серый | `#e0e0e0` | Читаемый на тёмном |
| Person (C4) | Тёмно-синий | `#08427b` | Стандарт C4 Simon Brown |
| System (C4) | Синий | `#1168bd` | Стандарт C4 |
| External (C4) | Серый | `#999999` | Стандарт C4 |
| Container (C4) | Голубой | `#438dd5` | Стандарт C4 |
| Component (C4) | Светло-голубой | `#85bbf0` | Стандарт C4 (чёрный текст!) |
| Заметки (notes) | Жёлтый | `#fff9c4` / `#f9a825` | Контраст с синими блоками |
| Текст в блоках | Белый | `#ffffff` | На тёмных заливках |
| Текст в светлых блоках | Чёрный | `#000000` | На светлых заливках |

---

## 1️⃣ C4 диаграммы (C4Context, C4Container)

### Синтаксис стилизации

```
UpdateElementStyle(имя, $bgColor="цвет", $fontColor="цвет", $borderColor="цвет")
UpdateRelStyle(от, до, $textColor="цвет", $lineColor="цвет")
```

### Готовый шаблон — C4Context

```mermaid
C4Context
    title Пример C4Context для Dark Theme

    Person(user, "User", "Описание")
    System(sys, "System", "Основная система")
    System_Ext(ext1, "External 1", "Внешняя система")
    System_Ext(ext2, "External 2", "Ещё одна")

    Rel(user, sys, "использует")
    Rel(sys, ext1, "вызывает API")
    Rel(sys, ext2, "читает данные")

    UpdateElementStyle(user, $bgColor="#08427b", $fontColor="#ffffff", $borderColor="#073b6f")
    UpdateElementStyle(sys, $bgColor="#1168bd", $fontColor="#ffffff", $borderColor="#0b4884")
    UpdateElementStyle(ext1, $bgColor="#999999", $fontColor="#ffffff", $borderColor="#6b6b6b")
    UpdateElementStyle(ext2, $bgColor="#999999", $fontColor="#ffffff", $borderColor="#6b6b6b")

    UpdateRelStyle(user, sys, $textColor="#e0e0e0", $lineColor="#cccccc")
    UpdateRelStyle(sys, ext1, $textColor="#e0e0e0", $lineColor="#cccccc")
    UpdateRelStyle(sys, ext2, $textColor="#e0e0e0", $lineColor="#cccccc")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Готовый шаблон — C4Container

```mermaid
C4Container
    title Пример C4Container для Dark Theme

    Person(user, "User", "")

    System_Boundary(boundary, "My System") {
        Container(web, "Web App", "React", "Frontend")
        Container(api, "API", "Python", "Backend")
        Container(db, "Database", "PostgreSQL", "Хранилище")
    }

    System_Ext(ext, "External API", "Внешний сервис")

    Rel(user, web, "открывает")
    Rel(web, api, "REST API")
    Rel(api, db, "SQL queries")
    Rel(api, ext, "HTTP calls")

    UpdateElementStyle(user, $bgColor="#08427b", $fontColor="#ffffff", $borderColor="#073b6f")
    UpdateElementStyle(web, $bgColor="#438dd5", $fontColor="#ffffff", $borderColor="#2e6295")
    UpdateElementStyle(api, $bgColor="#438dd5", $fontColor="#ffffff", $borderColor="#2e6295")
    UpdateElementStyle(db, $bgColor="#438dd5", $fontColor="#ffffff", $borderColor="#2e6295")
    UpdateElementStyle(ext, $bgColor="#999999", $fontColor="#ffffff", $borderColor="#6b6b6b")

    UpdateRelStyle(user, web, $textColor="#e0e0e0", $lineColor="#cccccc")
    UpdateRelStyle(web, api, $textColor="#e0e0e0", $lineColor="#cccccc")
    UpdateRelStyle(api, db, $textColor="#e0e0e0", $lineColor="#cccccc")
    UpdateRelStyle(api, ext, $textColor="#e0e0e0", $lineColor="#cccccc")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## 2️⃣ flowchart / graph — `%%{init}%%` + `linkStyle`

### Готовый шаблон

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#90caf9', 'textColor': '#e0e0e0'}}}%%
flowchart TB
    subgraph G1["Группа 1"]
        A["Компонент A"]
        B["Компонент B"]
    end

    subgraph G2["Группа 2"]
        C["Компонент C"]
        D["Компонент D"]
    end

    A -->|связь| C
    B -->|связь| D
    A -.->|implements| B

    linkStyle default stroke:#90caf9,stroke-width:2px

    style G1 fill:#0d253f,stroke:#42a5f5,stroke-width:2px,color:#e0e0e0
    style G2 fill:#1a3000,stroke:#66bb6a,stroke-width:2px,color:#e0e0e0
    style A fill:#1565c0,stroke:#42a5f5,color:#ffffff
    style B fill:#1565c0,stroke:#42a5f5,color:#ffffff
    style C fill:#2e7d32,stroke:#66bb6a,color:#ffffff
    style D fill:#2e7d32,stroke:#66bb6a,color:#ffffff
```

### Правила

- `linkStyle default stroke:#90caf9,stroke-width:2px` — делает ВСЕ связи голубыми
- `style NODE fill:...,stroke:...,color:...` — стилизация конкретного узла
- `color` в `style` — цвет ТЕКСТА внутри узла
- Subgraph: тёмная заливка + яркая рамка + светлый текст

---

## 3️⃣ classDiagram — `%%{init}%%`

### Готовый шаблон

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fffde7', 'primaryTextColor': '#1a1a1a', 'primaryBorderColor': '#f9a825', 'lineColor': '#90caf9', 'secondaryColor': '#e3f2fd', 'tertiaryColor': '#fff8e1', 'textColor': '#e0e0e0', 'classText': '#1a1a1a'}}}%%
classDiagram
    direction TB

    class IBase {
        &lt;&lt;ABC&gt;&gt;
        +method() result
    }

    class ConcreteA {
        -field str
        +method() result
    }

    class ConcreteB {
        -field int
        +method() result
    }

    class Factory {
        +create(type) IBase
    }

    IBase <|.. ConcreteA
    IBase <|.. ConcreteB
    Factory ..> IBase : creates
```

### Правила

- `<<ABC>>` → пишем `&lt;&lt;ABC&gt;&gt;` (HTML entities)
- НЕ используй `**kwargs*`, `dict[str, Type]$` — ломает парсер 10.9.5
- `primaryColor` = фон классов (жёлтый `#fffde7` — хорошо виден)
- `lineColor` = цвет связей (`#90caf9` — голубой)
- `classText` = цвет текста внутри классов

---

## 4️⃣ sequenceDiagram — `%%{init}%%`

### Готовый шаблон

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'actorBkg': '#1168bd', 'actorTextColor': '#ffffff', 'actorBorder': '#0b4884', 'actorLineColor': '#90caf9', 'signalColor': '#90caf9', 'signalTextColor': '#e0e0e0', 'labelBoxBkgColor': '#1a3a5c', 'labelBoxBorderColor': '#4a90d9', 'labelTextColor': '#ffffff', 'loopTextColor': '#e0e0e0', 'noteBkgColor': '#fff9c4', 'noteTextColor': '#1a1a1a', 'noteBorderColor': '#f9a825', 'activationBkgColor': '#1a3a5c', 'activationBorderColor': '#4a90d9', 'sequenceNumberColor': '#ffffff'}}}%%
sequenceDiagram
    participant A as Client
    participant B as Server
    participant C as Database

    A->>B: HTTP Request
    Note over B: Process
    B->>C: SQL Query
    C-->>B: Result Set
    B-->>A: JSON Response
```

### Ключевые переменные

| Переменная | Назначение | Значение |
|-----------|-----------|---------|
| `actorBkg` | Фон заголовков участников | `#1168bd` (синий) |
| `actorTextColor` | Текст в заголовках | `#ffffff` |
| `signalColor` | Цвет стрелок (связей!) | `#90caf9` (голубой) |
| `signalTextColor` | Текст над стрелками | `#e0e0e0` |
| `actorLineColor` | Вертикальные линии жизни | `#90caf9` |
| `noteBkgColor` | Фон заметок | `#fff9c4` (жёлтый) |
| `noteTextColor` | Текст заметок | `#1a1a1a` (тёмный) |
| `activationBkgColor` | Фон блоков активации | `#1a3a5c` |

---

## 5️⃣ Чеклист — новая диаграмма

- [ ] Тип диаграммы определён (C4 / flowchart / class / sequence)?
- [ ] C4 → `UpdateElementStyle` + `UpdateRelStyle` для КАЖДОГО элемента
- [ ] Не-C4 → `%%{init: {'theme': 'base', 'themeVariables': {...}}}%%`
- [ ] `lineColor: '#90caf9'` — голубые связи
- [ ] flowchart/graph → `linkStyle default stroke:#90caf9,stroke-width:2px`
- [ ] Блоки: тёмная/яркая заливка + белый текст `color:#ffffff`
- [ ] Нет frontmatter `---config---` (не работает в 10.9.5)
- [ ] Нет спецсимволов `*`, `$`, `**` в classDiagram (ломают парсер)
- [ ] `<<ABC>>` → `&lt;&lt;ABC&gt;&gt;`
- [ ] Проверить в VS Code Preview (Ctrl+Shift+V)

---

## ❌ Что НЕ работает в Mermaid 10.9.5

| Не работает | Альтернатива |
|------------|-------------|
| Frontmatter `---config:theme---` | `%%{init: {...}}%%` |
| `themeVariables` для C4 | `UpdateElementStyle` / `UpdateRelStyle` |
| `dict[str, Type]$` в classDiagram | `dict` (без generic) |
| `+method(**kwargs)*` в classDiagram | `+method(kwargs)` |
| `<<ABC>>` напрямую | `&lt;&lt;ABC&gt;&gt;` |
| `theme: 'default'` в frontmatter | `%%{init: {'theme': 'default'}}%%` |

---

*Создано: 2026-03-21 | Кодо | Проверено на реальных диаграммах GPUWorkLib*
