"""
ai_pipeline — разбивка монолита test_ai_filter_pipeline.py
============================================================

Было: 1 файл × 964 строки (AI + scipy + GPU + plot + тесты в куче)
Стало: 4 специализированных модуля + 1 демо-скрипт

Модули:
    llm_parser      — LLMParser (Strategy): Groq / Ollama / Mock
    filter_designer — FilterDesigner: scipy дизайн FIR/IIR
    test_ai_pipeline — тесты (<150 строк, только assert)

Демо-скрипт:
    demo_ai_pipeline.py — запускается руками, рисует графики

SRP применён: каждый модуль отвечает за одну вещь.
"""
