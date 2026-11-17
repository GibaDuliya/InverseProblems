# Основное обучение BM / ABM

Все команды выполняются из корня репозитория в активированном окружении.

```bash
python -m pip install -e '.[eval,quality,sweep,test]'
```

Устанавливает проект, метрики и зависимости проверки. ArcFace и MUSIQ при первом
создании скачивают pretrained-веса; нужен доступ в интернет или предварительно
заполненный кэш. InsightFace использует `~/.insightface`; путь можно изменить через
`evaluation.metrics.arcface.root`. MUSIQ использует кэш PyTorch (`TORCH_HOME`).
Исходные изображения должны лежать в `data/celeba/img_align_celeba`, partition —
в `data/celeba/list_eval_partition.txt`.

## Конфиги и запуск

`configs/train.yaml` — самостоятельный запуск BM, 26 млн параметров, T=1.

```bash
python -m bridge_restore.train --config configs/train.yaml
```

`configs/celeba_main_grid.yaml` — 12 последовательных запусков, каждый в отдельном
Python-процессе. Сначала medium: BM и ABM × T={1,1.5,2}; затем small.
Список запусков без выделения GPU и загрузки датасета:

```bash
python -m bridge_restore.grid --config configs/celeba_main_grid.yaml --list
```

Один выбранный запуск сетки:

```bash
python -m bridge_restore.grid --config configs/celeba_main_grid.yaml --only medium_bm_T1
```

Первые шесть запусков:

```bash
python -m bridge_restore.grid --config configs/celeba_main_grid.yaml --only medium_bm_T1 medium_bm_T1.5 medium_bm_T2 medium_abm_T1 medium_abm_T1.5 medium_abm_T2
```

Вся сетка:

```bash
python -m bridge_restore.grid --config configs/celeba_main_grid.yaml
```

Одиночный конфиг и medium_bm_T1 в сетке указывают на один output_dir — это один
эксперимент. Не запускайте их одновременно. Существующие результаты не
перезаписываются: для оставшихся комбинаций используйте `--only`.

## Начальные параметры

- A100 80 GB, BF16; фактический пик GPU-памяти локально не измерялся.
- 20 эпох (стартовый бюджет, не результат поиска оптимального числа эпох).
- Batch 32 × accumulation 2 = effective batch 64 для всех моделей.
- AdamW: lr=1e-4, weight_decay=0; EMA=0.996005996001.
- Small: 64 канала, 1 блок — 18.48M; medium: 64 / 2 — 26.00M;
  Числа для BM, ABM отличается лишь входным слоем.
- JPEG quality=10, subsampling=2; center-square crop, bicubic antialias resize 128.
- Время обучения t равномерно в [sigma_min,T]; шум гауссовский.
- SDE: 100 стохастических шагов плюс финальный denoiser, sigma_min=.001.
- seed=42 без принудительных deterministic CUDA алгоритмов и без отдельного RNG оценки.

Валидация после каждой эпохи на фиксированных 512 изображениях (включая 16
визуальных примеров). `subset_size: null` включает весь validation.
`max_steps: null` отключает ограничение числа обновлений. Если задать лимит шагов,
финальная частичная эпоха также оценивается; повторной оценки того же шага нет.
Неполный microbatch и неполная группа accumulation не отбрасываются.

На тесте дополнительно вычисляются FID, KID, observation_mse и observation_mae.
KID требует не менее 1000 изображений в тестовой выборке.

После обучения — фиксированная тестовая подвыборка из 4096 изображений для лучших по LPIPS и ArcFace, с 32 визуальными
примерами. Лучший MUSIQ хранится без автоматического теста. Тест для двух
checkpoint может занимать существенное время; для предварительных запусков можно
задать `test_subset_size`, не изменяя train/validation.

## Критерии и результаты

`evaluation.criteria` — словарь `{name: {metric, mode, test}}`.
Возможны любые включённые метрики с конечным средним. mode=min/max; равные
значения не заменяют checkpoint. Все критерии используют одну генерацию на
валидации. Выбор выполняется по EMA. Периодические checkpoint и last.pt в этом
режиме не пишутся; старый короткий sweep сохранил собственный протокол.

```
runs/celeba_main/<name>/
  config.yaml
  split.json
  metrics.jsonl
  validation/epoch_0001/{summary.json,per_image.jsonl,protocol.json,images/}
  best/lpips/{checkpoint.pt,summary.json,per_image.jsonl,protocol.json,images/}
  best/identity/...
  best/quality/...
  test/lpips/...
  test/identity/...
```

Таблицы по изображениям содержат также JPEG-baseline. Изображения best берутся из
той же генерации, по которой выбирался checkpoint; повторного сэмплирования нет.
Одинаковые checkpoint нескольких критериев тестируются один раз. Тест никогда
не используется для отбора. tqdm показывает train, validation и test.

ArcFace: InsightFace 0.7.3, buffalo_l/w600k_r50, CPUExecutionProvider, aligned112.
Landmarks вычисляются на чистом эталоне, по самому крупному лицу; одно и то же
преобразование применяется к эталону, JPEG и восстановлению. Кэш эталонов живёт
внутри evaluator. Покрытие и missing_sample_ids сохраняются. При покрытии ниже
0.9 запуск останавливается с ID проблемных изображений (порог настраивается).
Отсутствующий score записывается как null, не заменяется нулём.

MUSIQ: pyiqa 0.1.13, KonIQ-10k, RGB [0,1], без внешнего resize. Внутренний батч 1
ограничивает память независимо от batch_size генерации. Метрики работают без
градиентов; JPEG-baseline и restoration используют общие замороженные сети и
раздельные накопители. Метрики разнообразия не добавлены.

## Продолжение и отдельная оценка

Checkpoint содержит raw-веса, EMA, optimizer, RNG, позицию данных и таблицу лучших
критериев. Возобновляется именно сохранённая лучшая точка, не последняя эпоха:

```bash
python -m bridge_restore.train --config configs/train.yaml --resume runs/celeba_main/medium_bm_T1/best/lpips/checkpoint.pt
```

Другие лучшие checkpoint должны соответствовать сохранённой истории. Если они
обновлялись позже выбранной точки, программа сообщает несовпадение вместо
смешивания веток обучения. Для новой папки необходимо перенести соответствующие
исторические best-папки; одного произвольного checkpoint может быть недостаточно.
Лимит max_epochs/max_steps допускается увеличить при возобновлении.

Отдельная оценка (4096 тестовых изображений, все девять метрик):

```bash
python -m bridge_restore.evaluate --checkpoint runs/celeba_main/medium_bm_T1/best/lpips/checkpoint.pt --config configs/evaluation/main.yaml --output runs/manual_test --split test --device cuda
```

## Локальная проверка

Обычные проверки: `python -m pytest -q`.
Проверка evaluator с настоящими ArcFace/MUSIQ и локальным CelebA:

```bash
BRIDGE_QUALITY_SMOKE=1 CELEBA_ROOT=data/celeba INSIGHTFACE_ROOT="$HOME/.insightface" python -m pytest tests/test_quality_metrics.py -q
```

Проверено на CPU: ArcFace(clean,clean)=1.0, средний ArcFace(JPEG,clean)=0.6629
на четырёх фиксированных примерах; MUSIQ(clean)=40.4971, MUSIQ(JPEG)=23.1055.
Это smoke-проверка, не benchmark. Дополнительно проверены повторная оценка,
reset накопителей, совместное использование сетей и совпадение restoration с
baseline при identity-сэмплере. CUDA/BF16 и длительное обучение нужно проверить
на сервере; локально CUDA отсутствует.
