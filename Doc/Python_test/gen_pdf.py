#!/usr/bin/env python3
"""Generate PDF from markdown file starting from a specific section."""
import re
import subprocess
import tempfile
import os

INPUT_FILE = "MemoryBank/specs/python_test_refactoring.md"
OUTPUT_PDF  = "MemoryBank/specs/python_test_refactoring.pdf"
START_MARKER = "## 💡 Ключевые решения"

# --- Read & slice -----------------------------------------------------------
with open(INPUT_FILE, encoding="utf-8") as f:
    content = f.read()

idx = content.find(START_MARKER)
if idx == -1:
    raise ValueError(f"Marker '{START_MARKER}' not found in file")

md_slice = content[idx:]

# --- Markdown → HTML --------------------------------------------------------
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension

md = markdown.Markdown(extensions=[
    TableExtension(),
    FencedCodeExtension(),
    "nl2br",
])
body_html = md.convert(md_slice)

# --- Wrap in full HTML with nice CSS ----------------------------------------
HTML = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width"/>
<title>Python Test Refactoring — Ключевые решения</title>
<style>
  @page {{ size: A4; margin: 18mm 18mm 18mm 18mm; }}
  * {{ box-sizing: border-box; }}

  body {{
    font-family: "Noto Sans", "DejaVu Sans", Arial, sans-serif;
    font-size: 11px;
    line-height: 1.55;
    color: #222;
    background: #fff;
  }}

  h1, h2 {{ color: #1a1a6e; border-bottom: 2px solid #4a6fa5; padding-bottom: 4px; margin-top: 20px; }}
  h3 {{ color: #2a4a7f; margin-top: 14px; }}
  h4 {{ color: #3a5a8f; }}

  p {{ margin: 6px 0; }}

  /* Tables */
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 10.5px;
  }}
  th {{
    background: #4a6fa5;
    color: #fff;
    padding: 5px 7px;
    text-align: left;
    font-weight: 600;
  }}
  td {{
    padding: 4px 7px;
    border: 1px solid #ccd;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{ background: #f5f7fc; }}

  /* Code blocks */
  pre {{
    background: #f7f8fa;
    color: #1a1a1a;
    border: 1px solid #d0d4de;
    padding: 10px 12px;
    border-radius: 5px;
    font-family: "Noto Mono", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 9.5px;
    line-height: 1.45;
    white-space: pre;
    overflow-x: auto;
    page-break-inside: avoid;
  }}
  code {{
    background: #eef0f8;
    color: #1a1a1a;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: "Noto Mono", "DejaVu Sans Mono", monospace;
    font-size: 10px;
  }}
  pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
    font-size: 9.5px;
  }}

  /* Inline code in pre blocks (big pipeline diagram) */
  pre {{ font-size: 8.5px; }}

  /* Horizontal rule */
  hr {{ border: none; border-top: 1px solid #ccd; margin: 16px 0; }}

  /* Checkmarks & bullets */
  ul, ol {{ padding-left: 20px; margin: 6px 0; }}
  li {{ margin: 3px 0; }}

  /* Emphasis */
  strong {{ color: #1a1a6e; }}

  /* Avoid page breaks inside tables */
  tr {{ page-break-inside: avoid; }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

# --- Write temp HTML --------------------------------------------------------
tmp_html = os.path.abspath("MemoryBank/specs/_tmp_pdf.html")
with open(tmp_html, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"HTML written: {tmp_html}")

# --- Chrome headless → PDF --------------------------------------------------
out_pdf = os.path.abspath(OUTPUT_PDF)
cmd = [
    "google-chrome",
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    f"--print-to-pdf={out_pdf}",
    "--print-to-pdf-no-header",
    "--no-pdf-header-footer",
    f"file://{tmp_html}",
]

print("Running Chrome headless...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
if result.returncode != 0:
    print("STDERR:", result.stderr[:2000])
    raise RuntimeError("Chrome failed")

print(f"\n✅ PDF ready: {out_pdf}")

# Cleanup temp HTML
os.remove(tmp_html)
