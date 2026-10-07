# Контекст работы: Spotless-Film (форк)

Файл для продолжения работы в терминале (Claude Code или вручную).
Положи его в корень репозитория и скажи агенту: «прочитай CONTEXT.md».

## Репозиторий и ветка

- Репо: `https://github.com/onezerodoubledotsfivethree/Spotless-Film`
- Рабочая ветка: `claude/tkinterdnd2-windows-build-xosfzy` (всё запушено, PR не создавался)
- Приложение: десктоп на CustomTkinter + tkinterdnd2, U-Net ищет пыль на сканах плёнки,
  удаление через OpenCV inpaint (TELEA). Точка входа: `src/spotless_film_modern.py`.

```bash
git fetch origin claude/tkinterdnd2-windows-build-xosfzy
git checkout claude/tkinterdnd2-windows-build-xosfzy
```

## Коммиты в ветке (от старых к новым)

1. `efaa8e9` Install and bundle tkinterdnd2 in builds
2. `6edbab7` Document downloading model weights into src/weights for builds
3. `8c0a848` Remove .DS_Store files
4. `0484c81` Remove .vscode settings and ignore .vscode/
5. `81d1b5d` Allow importing multiple images
6. `ac9cda7` Add batch processing and export of all queued images

## Что сделано

### Сборка и зависимости
- `tkinterdnd2` добавлен в установку пакетов в `WINDOWS_BUILD.md`, `BUILD_INSTRUCTIONS.md`
  и `.github/workflows/build.yml` (workflow отключён через `if: false`).
- `src/build_executable.py`: `check_dependencies()` теперь словарь «pip-имя → import-имя»
  (`pillow→PIL`, `opencv-python→cv2`, `tkinterdnd2`). Раньше pillow/opencv всегда
  считались отсутствующими.
- `src/spotless_film.spec`: `hiddenimports += 'tkinterdnd2'`,
  `datas += collect_data_files('tkinterdnd2')` (нативные библиотеки tkdnd).
- Простая сборка в `WINDOWS_BUILD.md`:
  `pyinstaller --onefile --windowed --collect-all tkinterdnd2 --add-data "weights\*.pth;weights" --name SpotlessFilm spotless_film_modern.py`
- `src/requirements_spotless_film.txt`: исправлено имя пакета `tkinter-dnd2` → `tkinterdnd2`.

### Веса модели
- Ссылка: https://drive.google.com/file/d/1yR4gk2SgU0-p_EOrihckt3gFE8_lYyq7/view?usp=sharing
- Распаковать так, чтобы `.pth` лежали прямо в `src\weights\` (без вложенной папки).
  Приложение сначала ищет `src/weights/v5_bce_unet_epoch30.pth`, потом любой `*.pth`
  (`find_model_files`).
- Без хотя бы одного `.pth` в `src/weights` сборка по `.spec` падает.
- Содержимое архива с Drive не проверялось. Если имя файла другое, поправить схему
  в `WINDOWS_BUILD.md`, шаг 4.

### Чистка репо
- Удалены `.DS_Store` (корень, `dust_remover/`), они уже были в `.gitignore`.
- Удалён `.vscode/settings.json`, в `.gitignore` добавлен `.vscode/`.

### Несколько картинок (очередь), `src/spotless_film_modern.py`
- Состояние в `__init__`: `image_queue` (абсолютные пути), `current_index`,
  `image_results` (path → dict с `raw_prediction_mask`, `dust_mask`, `original_dust_mask`,
  `processed_image`, `preview_processed_image`).
- Импорт: `import_image` → `filedialog.askopenfilenames` + `root.tk.splitlist`
  (на некоторых Tk приходит строка, а не tuple).
- Drag & drop: `drop_target_register(DND_FILES)` на `self.canvas` → `handle_file_drop`
  принимает несколько файлов. Раньше DnD в этом окне не был подключён вообще
  (`SpotlessCanvas` из `professional_canvas.py` не используется).
- `add_images` → `switch_to_image` (сохраняет результаты текущей через
  `save_current_image_results`, грузит новую, восстанавливает её результаты через
  `restore_image_results`) → `refresh_image_queue` (перерисовывает список).
- `load_image` теперь возвращает `bool`; битый файл удаляется из очереди.
- Переключение блокируется во время детекции, удаления и пакетной обработки.
- UI в разделе Import (виден при ≥2 картинках): ◀/▶, «N / M», список (✓ = есть результат),
  Process All / Export All, строка статуса пакета, Clear List.

### Пакетная обработка и экспорт
- `process_all_images`: обрабатывает картинки без `processed_image`. Если маска уже есть
  (например, правленая вручную), детекция не запускается. Порог берётся с текущего
  слайдера. Работает в daemon-потоке, повторяет шаги одиночного пути:
  `predict_dust_mask` → `create_binary_mask` → `dilate_mask` → `perform_cv2_inpainting`
  → `blend_images`.
- Поток не трогает Tk: колбэки кладутся в `self._batch_events` (`queue.Queue`),
  UI-поток разбирает их в `_poll_batch_events` каждые 100 мс.
- Перед стартом потока вызываются `Image.init()` и `gc.collect()`. Без этого на битом файле
  Pillow лениво импортировал плагины в потоке, GC освобождал Tk-шрифт из потока
  и всё зависало (было ~19 с вместо 0.8 с).
- Отмена: кнопка становится «⏹ Cancel» (`_batch_cancel`, `threading.Event`),
  остановка после текущей картинки. Итог: «Processed X of N in Ts; failed: …».
- `export_all_images`: `askdirectory`, файлы `<имя>_dust_removed<исходное расширение>`
  (jpg/jpeg/png/tif/tiff/bmp, иначе `.png`; JPEG сохраняется с quality=95).
  Существующие файлы не перезаписываются (`_2`, `_3`…), необработанные пропускаются.
- Боковая панель стала `CTkScrollableFrame` (width=264): раньше раздел Dust Removal уезжал
  за край окна. Кнопки пакета имеют `width=90`: дефолтные 140px распирали панель.

## Как проверялось (и ограничения)

- Тестов в репо нет (`src/test_removal.py` к UI не относится).
- Смоук-тесты гонялись под `xvfb-run` на `python3.12` с заглушкой `torch` (torch в облаке
  не ставился): 15 проверок очереди + 18 проверок пакета/экспорта, все прошли.
  Детектор подменялся фейком, возвращающим маску; реальные mask/inpaint/blend/export работали.
- **Не проверено:** настоящая модель на реальных сканах, Windows, PyInstaller-сборка `.exe`.

Быстрая ручная проверка локально:
```bash
cd src
pip install -r requirements_spotless_film.txt customtkinter
python spotless_film_modern.py
# выбрать несколько файлов → ◀/▶ → Process All → Export All
```

## Известные ограничения и идеи

- Результаты всех картинок хранятся в RAM. На десятках больших сканов это много,
  Clear List освобождает память.
- Пакет использует один общий порог; поштучной настройки нет.
- ✓ в списке появляется только после переключения с картинки или после пакета
  (результат одиночной обработки сохраняется в `image_results` при уходе с картинки).
- Уже существовавший код одиночных `detect_dust`/`remove_dust` трогает Tk из рабочего
  потока (`status_label.configure` в `completion_callback` детекции). Это не исправлялось.
  Если будут зависания, перевести на тот же паттерн с очередью.
- В `export_image` (одиночный экспорт) на Linux вызывается `open`, который есть только
  на macOS. Это тоже не исправлялось.

## Правила коммитов, которые использовались

Каждый коммит заканчивается строками:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SmAPMUiXwscCtj7AtXJJpv
```
