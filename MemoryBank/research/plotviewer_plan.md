# Plan: PlotViewer — Локальный сайт для просмотра тестовых графиков

> Дата: 2026-04-05
> Цель: Веб-галерея для PNG из Python-тестов — просмотр, описания, удаление
> Технологии: FastAPI + vanilla HTML/JS, Python 3.10+
> Браузер: Chrome (локально, без деплоя)
> Статус: 📋 PLAN — готов к созданию тасков

---

## Концепция

Один сайт на **несколько проектов** (`GPUWorkLib`, `Refactoring`, ...).
Структура: `Проект → Модуль → Список картинок`.
Для каждой картинки можно добавить текстовое описание или удалить её.
Данные о структуре — из `config.json`, описания — в `descriptions.json`.

```
http://localhost:8000
│
├── /                    → index.html (весь UI)
└── /api/...             → FastAPI backend
    ├── GET  /projects              → список проектов
    ├── GET  /images/{proj}/{mod}   → список PNG в папке
    ├── GET  /image/{proj}/{mod}/{file} → отдать файл (StaticFiles)
    ├── DELETE /image               → удалить файл
    ├── POST /description           → сохранить описание
    └── GET  /description/{...}     → получить описание
```

---

## Структура проекта PlotViewer

```
E:\C++\PlotViewer\               ← ОТДЕЛЬНЫЙ проект, НЕ в GPUWorkLib
├── app.py                       ← FastAPI сервер (~150 строк)
├── config.json                  ← пути к проектам
├── descriptions.json            ← авто-создаётся, хранит описания
├── requirements.txt             ← fastapi uvicorn
├── static\
│   └── index.html               ← весь UI (HTML + CSS + JS, ~300 строк)
└── README.md
```

**Почему отдельный проект**: PlotViewer — общая утилита для любых проектов.
Не нужно тащить его в GPUWorkLib или Refactoring.

---

## config.json — добавить проект за 30 секунд

```json
{
  "projects": {
    "GPUWorkLib": {
      "description": "GPU Signal Processing Library",
      "path": "E:/C++/GPUWorkLib/Results/Plots"
    },
    "Refactoring": {
      "description": "Refactoring experiments",
      "path": "E:/C++/Refactoring/Results/Plots"
    }
  }
}
```

Добавить новый проект = добавить 4 строки в этот файл. Всё.
Модули (подпапки) — определяются автоматически по структуре директорий.

---

## app.py — полный код

```python
#!/usr/bin/env python3
"""
PlotViewer — локальный сайт для просмотра PNG из тестов GPU.
Запуск: python app.py
URL: http://localhost:8000
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
DESC_FILE   = BASE_DIR / "descriptions.json"
STATIC_DIR  = BASE_DIR / "static"

def load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)

def load_descriptions() -> dict:
    if not DESC_FILE.exists():
        return {}
    with open(DESC_FILE, encoding="utf-8") as f:
        return json.load(f)

def save_descriptions(data: dict):
    with open(DESC_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────

app = FastAPI(title="PlotViewer", version="1.0")


# ─────────────────────────────────────────────────────────
# Главная страница — отдаём index.html
# ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────────────────
# API: Проекты
# ─────────────────────────────────────────────────────────

@app.get("/api/projects")
def get_projects():
    """Возвращает список проектов из config.json."""
    cfg = load_config()
    result = []
    for name, info in cfg["projects"].items():
        path = Path(info["path"])
        modules = []
        if path.exists():
            modules = sorted([
                d.name for d in path.iterdir()
                if d.is_dir()
            ])
        result.append({
            "name": name,
            "description": info.get("description", ""),
            "path": str(path),
            "modules": modules,
            "exists": path.exists()
        })
    return result


# ─────────────────────────────────────────────────────────
# API: Изображения
# ─────────────────────────────────────────────────────────

@app.get("/api/images/{project}/{module}")
def get_images(project: str, module: str):
    """
    Возвращает список PNG файлов в папке проекта/модуля.
    Поддерживает вложенные подпапки (signal_generators/FormSignal).
    """
    cfg = load_config()
    if project not in cfg["projects"]:
        raise HTTPException(404, f"Project '{project}' not found")

    base = Path(cfg["projects"][project]["path"]) / module
    if not base.exists():
        raise HTTPException(404, f"Module '{module}' not found in '{project}'")

    # Рекурсивно собираем все PNG (включая подпапки)
    images = []
    for f in sorted(base.rglob("*.png")):
        # Относительный путь от base, для красивого отображения
        rel = f.relative_to(base)
        images.append({
            "filename": f.name,
            "relpath":  str(rel).replace("\\", "/"),
            "abspath":  str(f),
            "size_kb":  round(f.stat().st_size / 1024, 1),
            "subdir":   str(rel.parent) if str(rel.parent) != "." else ""
        })
    return images


@app.get("/api/image/{project}/{module}")
def serve_image(project: str, module: str, relpath: str):
    """Отдаёт файл изображения по относительному пути."""
    cfg = load_config()
    if project not in cfg["projects"]:
        raise HTTPException(404, "Project not found")

    base = Path(cfg["projects"][project]["path"]) / module
    file_path = base / relpath

    # Защита от path traversal
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path")

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(str(file_path), media_type="image/png")


# ─────────────────────────────────────────────────────────
# API: Удаление
# ─────────────────────────────────────────────────────────

class DeleteRequest(BaseModel):
    project: str
    module:  str
    relpath: str

@app.delete("/api/image")
def delete_image(req: DeleteRequest):
    """Удаляет PNG файл с диска."""
    cfg = load_config()
    if req.project not in cfg["projects"]:
        raise HTTPException(404, "Project not found")

    base = Path(cfg["projects"][req.project]["path"]) / req.module
    file_path = base / req.relpath

    # Защита от path traversal
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path")

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    file_path.unlink()

    # Удалить описание из descriptions.json
    descs = load_descriptions()
    key = f"{req.project}/{req.module}/{req.relpath}"
    if key in descs:
        del descs[key]
        save_descriptions(descs)

    return {"status": "deleted", "file": req.relpath}


# ─────────────────────────────────────────────────────────
# API: Описания
# ─────────────────────────────────────────────────────────

class DescriptionRequest(BaseModel):
    project:     str
    module:      str
    relpath:     str
    description: str

@app.post("/api/description")
def set_description(req: DescriptionRequest):
    """Сохраняет описание для картинки."""
    descs = load_descriptions()
    key = f"{req.project}/{req.module}/{req.relpath}"
    descs[key] = req.description
    save_descriptions(descs)
    return {"status": "saved", "key": key}

@app.get("/api/description/{project}/{module}")
def get_descriptions_for_module(project: str, module: str):
    """Возвращает все описания для модуля (словарь relpath → text)."""
    descs = load_descriptions()
    prefix = f"{project}/{module}/"
    result = {}
    for key, val in descs.items():
        if key.startswith(prefix):
            relpath = key[len(prefix):]
            result[relpath] = val
    return result


# ─────────────────────────────────────────────────────────
# Запуск
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("🚀 PlotViewer запущен: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
```

---

## static/index.html — полный код UI

```html
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PlotViewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }

  /* ── Шапка ── */
  header {
    background: #16213e; padding: 12px 20px;
    border-bottom: 1px solid #0f3460;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  header h1 { font-size: 1.2rem; color: #e94560; }

  /* Табы проектов */
  .tabs { display: flex; gap: 8px; flex-wrap: wrap; }
  .tab {
    padding: 6px 16px; border-radius: 20px; cursor: pointer;
    background: #0f3460; color: #aaa; border: 1px solid #1a4080;
    font-size: 0.85rem; transition: all 0.2s;
  }
  .tab.active { background: #e94560; color: white; border-color: #e94560; }
  .tab:hover:not(.active) { background: #1a4080; color: #ddd; }

  /* ── Главный layout ── */
  .layout { display: flex; height: calc(100vh - 56px); }

  /* ── Сайдбар — список модулей ── */
  .sidebar {
    width: 200px; min-width: 160px;
    background: #16213e; border-right: 1px solid #0f3460;
    overflow-y: auto; padding: 8px 0;
  }
  .sidebar-title {
    padding: 8px 14px; font-size: 0.7rem; color: #666;
    text-transform: uppercase; letter-spacing: 1px;
  }
  .module-item {
    padding: 8px 14px; cursor: pointer; font-size: 0.85rem;
    border-left: 3px solid transparent; transition: all 0.15s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .module-item:hover { background: #1a2a4a; color: #cce; }
  .module-item.active { border-left-color: #e94560; background: #1a2a4a; color: #fff; }
  .module-count { float: right; color: #666; font-size: 0.75rem; }

  /* ── Контент — галерея ── */
  .content { flex: 1; overflow-y: auto; padding: 16px; }

  .content-header {
    margin-bottom: 16px; display: flex;
    align-items: center; gap: 12px;
  }
  .content-header h2 { font-size: 1rem; color: #ccc; }
  .content-header .count { color: #666; font-size: 0.85rem; }

  /* Подраздел (subdir) */
  .subdir-title {
    font-size: 0.8rem; color: #888; margin: 20px 0 8px;
    padding-bottom: 4px; border-bottom: 1px solid #2a2a4a;
  }

  /* ── Грид картинок ── */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }
  .card {
    background: #16213e; border: 1px solid #1a3060;
    border-radius: 8px; overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .card:hover { border-color: #e94560; box-shadow: 0 0 12px #e9456022; }

  .card-img {
    width: 100%; height: 200px; object-fit: contain;
    background: #0a0a1a; cursor: pointer;
    transition: opacity 0.2s;
  }
  .card-img:hover { opacity: 0.85; }

  .card-body { padding: 10px 12px; }
  .card-name {
    font-size: 0.8rem; color: #aaa; margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .card-name span.size { color: #555; font-size: 0.72rem; float: right; }

  /* ── Описание ── */
  .desc-text {
    font-size: 0.8rem; color: #99b; min-height: 32px;
    cursor: pointer; line-height: 1.4;
    border-radius: 4px; padding: 4px 6px;
    border: 1px solid transparent;
    transition: border-color 0.2s;
  }
  .desc-text:hover { border-color: #335; }
  .desc-text.empty { color: #555; font-style: italic; }

  .desc-edit {
    width: 100%; background: #0f1f3a; color: #ccc;
    border: 1px solid #e94560; border-radius: 4px;
    padding: 4px 6px; font-size: 0.8rem; resize: none;
    min-height: 60px; font-family: inherit;
  }
  .desc-edit:focus { outline: none; }

  .card-actions {
    display: flex; gap: 6px; margin-top: 8px; justify-content: flex-end;
  }
  .btn-save {
    background: #0f3460; color: #88aaff;
    border: 1px solid #1a4080; border-radius: 4px;
    padding: 3px 10px; font-size: 0.75rem; cursor: pointer;
  }
  .btn-save:hover { background: #1a4080; }
  .btn-del {
    background: #3a1020; color: #e94560;
    border: 1px solid #5a2030; border-radius: 4px;
    padding: 3px 10px; font-size: 0.75rem; cursor: pointer;
  }
  .btn-del:hover { background: #5a2030; }

  /* ── Лайтбокс ── */
  .lightbox {
    display: none; position: fixed; inset: 0;
    background: #000c; z-index: 1000;
    align-items: center; justify-content: center;
    cursor: zoom-out;
  }
  .lightbox.open { display: flex; }
  .lightbox img {
    max-width: 95vw; max-height: 95vh;
    border: 2px solid #e94560; border-radius: 4px;
  }

  /* ── Пустые состояния ── */
  .empty-state { text-align: center; padding: 60px 20px; color: #555; }
  .empty-state .icon { font-size: 3rem; margin-bottom: 12px; }

  /* ── Статусбар ── */
  #toast {
    position: fixed; bottom: 20px; right: 20px;
    background: #0f3460; color: #88aaff;
    padding: 10px 18px; border-radius: 6px;
    font-size: 0.85rem; opacity: 0; transition: opacity 0.3s;
    pointer-events: none; z-index: 2000;
    border: 1px solid #1a4080;
  }
  #toast.show { opacity: 1; }
</style>
</head>
<body>

<header>
  <h1>📊 PlotViewer</h1>
  <div class="tabs" id="tabs"></div>
</header>

<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-title">Модули</div>
    <div id="modules"></div>
  </aside>
  <main class="content" id="content">
    <div class="empty-state">
      <div class="icon">📁</div>
      <div>Выберите проект и модуль</div>
    </div>
  </main>
</div>

<!-- Лайтбокс -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightbox-img" src="" alt="">
</div>

<div id="toast"></div>

<script>
// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let state = {
  projects: [],
  currentProject: null,
  currentModule: null,
  images: [],
  descs: {}
};

// ─────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────
async function init() {
  const res = await fetch('/api/projects');
  state.projects = await res.json();
  renderTabs();
  if (state.projects.length > 0) selectProject(state.projects[0].name);
}

// ─────────────────────────────────────────────
// Табы проектов
// ─────────────────────────────────────────────
function renderTabs() {
  const el = document.getElementById('tabs');
  el.innerHTML = state.projects.map(p => `
    <div class="tab ${p.name === state.currentProject ? 'active' : ''}"
         onclick="selectProject('${p.name}')">
      ${p.name}
    </div>`).join('');
}

function selectProject(name) {
  state.currentProject = name;
  state.currentModule = null;
  renderTabs();
  renderModules();
  document.getElementById('content').innerHTML = `
    <div class="empty-state"><div class="icon">📂</div><div>Выберите модуль</div></div>`;
}

// ─────────────────────────────────────────────
// Сайдбар — список модулей
// ─────────────────────────────────────────────
function renderModules() {
  const proj = state.projects.find(p => p.name === state.currentProject);
  if (!proj) return;
  const el = document.getElementById('modules');
  if (!proj.modules.length) {
    el.innerHTML = `<div style="padding:12px 14px;color:#555;font-size:.8rem">Нет модулей</div>`;
    return;
  }
  el.innerHTML = proj.modules.map(m => `
    <div class="module-item ${m === state.currentModule ? 'active' : ''}"
         onclick="selectModule('${m}')" title="${m}">
      ${m}
    </div>`).join('');
}

async function selectModule(mod) {
  state.currentModule = mod;
  renderModules();
  await loadImages();
}

// ─────────────────────────────────────────────
// Загрузка и рендер картинок
// ─────────────────────────────────────────────
async function loadImages() {
  const { currentProject: p, currentModule: m } = state;
  if (!p || !m) return;

  const [imgRes, descRes] = await Promise.all([
    fetch(`/api/images/${p}/${m}`),
    fetch(`/api/description/${p}/${m}`)
  ]);
  state.images = await imgRes.json();
  state.descs  = await descRes.json();

  renderGallery();
}

function renderGallery() {
  const { images, descs, currentProject: p, currentModule: m } = state;
  const content = document.getElementById('content');

  if (!images.length) {
    content.innerHTML = `<div class="empty-state"><div class="icon">🖼️</div>
      <div>Нет картинок в модуле <b>${m}</b></div></div>`;
    return;
  }

  // Группируем по subdir
  const groups = {};
  images.forEach(img => {
    const key = img.subdir || '_root_';
    if (!groups[key]) groups[key] = [];
    groups[key].push(img);
  });

  let html = `<div class="content-header">
    <h2>📁 ${p} / ${m}</h2>
    <span class="count">${images.length} файлов</span>
  </div>`;

  const rootImages = groups['_root_'] || [];
  if (rootImages.length) {
    html += renderGrid(rootImages, p, m, descs);
  }

  Object.keys(groups).sort().forEach(key => {
    if (key === '_root_') return;
    html += `<div class="subdir-title">📂 ${key}</div>`;
    html += renderGrid(groups[key], p, m, descs);
  });

  content.innerHTML = html;
}

function renderGrid(images, p, m, descs) {
  const cards = images.map(img => {
    const desc = descs[img.relpath] || '';
    const imgUrl = `/api/image/${p}/${m}?relpath=${encodeURIComponent(img.relpath)}`;
    const safeRel = img.relpath.replace(/'/g, "\\'");
    return `
    <div class="card" id="card-${CSS.escape(img.relpath)}">
      <img class="card-img" src="${imgUrl}" alt="${img.filename}"
           onclick="openLightbox('${imgUrl}')" loading="lazy">
      <div class="card-body">
        <div class="card-name">
          ${img.filename}
          <span class="size">${img.size_kb} KB</span>
        </div>
        <div class="desc-text ${desc ? '' : 'empty'}"
             id="desc-${CSS.escape(img.relpath)}"
             onclick="editDesc('${safeRel}')">
          ${desc || 'Нажми чтобы добавить описание...'}
        </div>
        <div class="card-actions">
          <button class="btn-del" onclick="deleteImage('${safeRel}')">🗑️ Удалить</button>
        </div>
      </div>
    </div>`;
  });
  return `<div class="grid">${cards.join('')}</div>`;
}

// ─────────────────────────────────────────────
// Редактирование описания
// ─────────────────────────────────────────────
function editDesc(relpath) {
  const el = document.getElementById(`desc-${CSS.escape(relpath)}`);
  if (!el || el.tagName === 'TEXTAREA') return;

  const current = state.descs[relpath] || '';
  el.outerHTML = `
    <textarea class="desc-edit" id="desc-${CSS.escape(relpath)}"
      onblur="saveDesc('${relpath.replace(/'/g, "\\'")}')"
      onkeydown="if(event.key==='Escape')cancelEdit('${relpath.replace(/'/g, "\\'")}')">
${current}</textarea>`;

  const ta = document.getElementById(`desc-${CSS.escape(relpath)}`);
  if (ta) { ta.focus(); ta.select(); }
}

async function saveDesc(relpath) {
  const ta = document.getElementById(`desc-${CSS.escape(relpath)}`);
  if (!ta) return;
  const text = ta.value.trim();

  await fetch('/api/description', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      project: state.currentProject,
      module:  state.currentModule,
      relpath, description: text
    })
  });

  state.descs[relpath] = text;

  ta.outerHTML = `
    <div class="desc-text ${text ? '' : 'empty'}"
         id="desc-${CSS.escape(relpath)}"
         onclick="editDesc('${relpath.replace(/'/g, "\\'")}')">
      ${text || 'Нажми чтобы добавить описание...'}
    </div>`;

  toast('💾 Описание сохранено');
}

function cancelEdit(relpath) {
  // Просто уйти из фокуса — сохранит onblur
  const ta = document.getElementById(`desc-${CSS.escape(relpath)}`);
  if (ta) ta.blur();
}

// ─────────────────────────────────────────────
// Удаление
// ─────────────────────────────────────────────
async function deleteImage(relpath) {
  if (!confirm(`Удалить файл?\n${relpath}`)) return;

  const res = await fetch('/api/image', {
    method: 'DELETE',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      project: state.currentProject,
      module:  state.currentModule,
      relpath
    })
  });

  if (res.ok) {
    toast('🗑️ Файл удалён');
    state.images = state.images.filter(i => i.relpath !== relpath);
    delete state.descs[relpath];
    renderGallery();
  } else {
    toast('❌ Ошибка удаления', true);
  }
}

// ─────────────────────────────────────────────
// Лайтбокс
// ─────────────────────────────────────────────
function openLightbox(url) {
  document.getElementById('lightbox-img').src = url;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});

// ─────────────────────────────────────────────
// Toast уведомление
// ─────────────────────────────────────────────
let toastTimer;
function toast(msg, isError = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = isError ? '#3a1020' : '#0f3460';
  el.style.color = isError ? '#e94560' : '#88aaff';
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2500);
}

// ─────────────────────────────────────────────
// Start
// ─────────────────────────────────────────────
init();
</script>
</body>
</html>
```

---

## requirements.txt

```
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
```

Установка:
```bash
pip install -r requirements.txt
```

---

## README.md (кратко)

```markdown
# PlotViewer

Локальная галерея для PNG из тестов GPU-проектов.

## Запуск
```bash
pip install fastapi uvicorn
python app.py
# → http://localhost:8000
```

## Добавить новый проект
В `config.json` добавить секцию:
```json
"МойПроект": {
    "description": "Описание",
    "path": "E:/path/to/Results/Plots"
}
```

## Структура папок
Папки в `path` → вкладки в левом сайдбаре (модули).
PNG файлы → карточки в галерее.
Подпапки → секции внутри модуля.
```

---

## Итоговая структура на экране

```
┌──────────────────────────────────────────────────────────┐
│ 📊 PlotViewer   [GPUWorkLib] [Refactoring]               │
├────────────────┬─────────────────────────────────────────┤
│ Модули         │ 📁 GPUWorkLib / filters  (17 файлов)    │
│                │                                          │
│ • fft_maxima   │ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ ● filters      │ │  [img]   │ │  [img]   │ │  [img]   │ │
│ • heterodyne   │ │          │ │          │ │          │ │
│ • integration  │ ai_fir...  │ ai_iir...  │ report_...│ │
│ • signal_gen.  │ FIR фильтр │ (нажми для│ Moving avg│ │
│ • statistics   │ нижних ч.. │  описания) │ тест 1    │ │
│ • strategies   │ [🗑️ Удал] │ [🗑️ Удал] │ [🗑️ Удал] │ │
│                │ └──────────┘ └──────────┘ └──────────┘ │
│                │                                          │
│                │ 📂 FormSignal  ← subdir секция          │
│                │ ┌──────────┐ ┌──────────┐ ...          │
└────────────────┴─────────────────────────────────────────┘
```

---

## Дополнительные возможности (фаза 2 — потом)

| Фича | Сложность | Польза |
|------|-----------|--------|
| Фильтр/поиск по имени файла | ⭐ | Быстро найти нужный график |
| Экспорт выбранных в PDF/ZIP | ⭐⭐ | Отчёты |
| Автообновление при новых тестах | ⭐⭐ | Запускаем тест → галерея обновилась |
| Сравнение 2 картинок side-by-side | ⭐⭐ | Сравнить до/после оптимизации |
| Теги для картинок | ⭐⭐ | Фильтрация по тегам |
| Bulk delete (выбрать несколько) | ⭐⭐ | Быстрая очистка устаревших |

---

## Таски для реализации (черновик)

- [ ] **TASK-PV-01**: Создать `E:\C++\PlotViewer\` с `requirements.txt` и `config.json`
- [ ] **TASK-PV-02**: Написать `app.py` (FastAPI backend, все endpoints)
- [ ] **TASK-PV-03**: Написать `static/index.html` (галерея, лайтбокс, редактирование описаний)
- [ ] **TASK-PV-04**: Протестировать с GPUWorkLib (58 PNG) + Refactoring
- [ ] **TASK-PV-05**: Проверить удаление, сохранение описаний, вложенные подпапки (signal_generators/)
- [ ] **TASK-PV-06**: Добавить README.md
