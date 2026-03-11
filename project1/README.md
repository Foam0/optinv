# Проект по оптимальному инвестированию

Тема:

Модель связи валютных курсов и цены на нефть

Критерии оценки:

1. Обзор литературы и постановка задачи
2. Теоретические результаты
3. Программная часть
4. Отчет о проделанной работе
5. Презентация и выступление

Файлы проекта:

- `report.tex` и `report.pdf` — основной отчет
- `slides.tex` и `slides.pdf` — презентация
- `speech_notes.md` — краткий текст выступления
- `src/analysis.py` — воспроизводимый код анализа
- `data/` — загруженные ряды и объединенные выборки
- `results/` — таблицы и графики

Пересборка:

1. `./.venv/bin/python src/analysis.py`
2. `latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex`
3. `latexmk -pdf -interaction=nonstopmode -halt-on-error slides.tex`
