# Bridge Restore

Минимальное обучение VE-диффузионного моста для восстановления изображений.
`model.mode: bm` исключает наблюдение из входа денойзера; `abm` передаёт его
отдельным входом на каждом шаге. Оба режима обучаются на тех же парных данных.
Пакет использует адаптированную ADM U-Net из DDBM; источники и изменения описаны
в [THIRD_PARTY.md](THIRD_PARTY.md). Папка `DDBM/` не нужна для запуска.

## Установка

Python 3.9+, PyTorch 2.4+. Установите подходящую вашей CUDA версию PyTorch,
затем из корня проекта:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
```

Обязательных зависимостей от torchvision, MPI, wandb и flash-attn нет.
Поддерживается одно устройство CPU или CUDA. Для CPU выбирайте `precision: float32`.
На A100 доступен `bfloat16`; скорость и память приведённого конфига нужно измерить
на вашем GPU. Примерные гиперпараметры не являются готовым экспериментальным результатом.

## Первый запуск: CelebA + JPEG

Ожидаемая структура исходных данных:

```text
celeba/
├── img_align_celeba/
└── list_eval_partition.txt
```

В `configs/train.yaml` укажите `data.kwargs.root` и уникальный `output_dir`.
Все относительные пути разрешаются относительно текущей рабочей директории.

```bash
bridge-train --config configs/train.yaml
# Эквивалентно:
python -m bridge_restore.train --config configs/train.yaml
```

Для BM измените `model.mode` на `bm` и выберите новую папку запуска. Manifest
разбиения оставьте тем же. Меняется conditioning, а не парные данные или objective.
Количество параметров печатается при старте; concat добавляет параметры первой свёртки.

Возобновление (можно увеличить `training.max_steps`):

```bash
bridge-train --config configs/train.yaml --resume runs/celeba_abm/checkpoints/last.pt
```

Сохраняются raw/EMA-веса, optimizer, scaler, RNG, позиция обхода данных, конфиг,
checksum разбиения и фактические параметры деградации. Проверяется совместимость
модели, референса, preprocessing и обучения. Допускаются изменение каталога вывода,
бюджета, интервалов логирования/validation/checkpoint, числа workers, устройства и
настроек генерации. При чтении checkpoint используется pickle: загружайте свои доверенные файлы.

`step` — успешное обновление optimizer. Накопление градиентов усредняет по всем
примерам, включая неполные batch. По умолчанию используется AdamW, постоянный LR.
`validation_loss` — взвешенная ошибка денойзера raw-модели на фиксированных временах
и шумах, а не PSNR или оценка восстановлений. Ограничение `validation_batches`
выбирает первые batch стабильного validation-порядка. Test не используется обучением.

## Контракт данных

`data/` создаёт `DataBundle`:

```text
train_loader, val_loader, test_loader, corruption, split_manifest
```

Trainer получает только train/validation-загрузчики. Их batch содержит:

```python
{"x0": ..., "xT": ..., "sample_id": ...}
# x0, xT: float32 [B,3,H,W] в [-1,1]
```

Порядок преобразований: исходник → center crop/resize → чистый float RGB [0,1]
→ деградация → нормировка обоих концов в [-1,1]. JPEG использует округление
`round(255*x)` до uint8 и настоящий Pillow encode/decode. Чистый эталон сохраняет
результат float preprocessing. Настройки/версия кодека влияют на наблюдение; фиксируйте
окружение при сравнении экспериментов. Деградация вычисляется при чтении, пары не кешируются.

`crop_mode: none` выполняет прямой resize до квадратного размера и может менять
соотношение сторон; `center_square` сначала вырезает центральный квадрат.
Встроенные преобразования детерминированы. Все аугментации будущих реализаций должны
сохранять согласованность пары; случайные custom datasets не дают гарантии точного resume.

### Разбиения

- `mode: official`: напрямую использовать официальное разбиение датасета.
- `mode: load`: загрузить существующий manifest.
- `mode: create_if_missing`: один раз распределить группы с заданными fractions/seed.

Для CelebA достаточно `split: {mode: official}`. Источник — `list_eval_partition.txt`;
персональная непересекаемость уже обеспечена его авторами. Identity-аннотации не
загружаются и повторная проверка людей не выполняется. При необходимости анализа
их можно подключить через `data.kwargs.identity_file: identity_CelebA.txt`.
Проверяется только корректность разделения ID изображений. Создание и чтение
отдельного кеша разбиения не нужны; snapshot по-прежнему сохраняется в запуске
для воспроизводимости и checksum checkpoint.

Для своих разбиений (`load` / `create_if_missing`) укажите `manifest_path` и
`group_by`. Существующий manifest имеет приоритет над новым seed/fractions.
Для другого разбиения используйте другой путь. Части должны быть непустыми и
покрывать все ID ровно один раз. При `group_by: identity` группы не пересекаются;
отсутствие identity вызывает ошибку. У пользовательского official-адаптера
гарантия непересечения групп является ответственностью источника разбиения.

Генерация распределяет группы, поэтому fractions относятся приблизительно к числу
групп, а доли изображений могут отличаться. Для данных без групп можно явно выбрать
`group_by: sample`. Для лиц это снимает защиту от пересечения персон и не рекомендуется.

Manifest содержит ID, группы и fingerprint состава датасета. Проверяется состав
ID/групп, а не содержимое каждого файла. Изменённое изображение под прежним ID
автоматически не обнаруживается. Копия manifest сохраняется в каталоге запуска.

### Свой датасет

Укажите импортируемую фабрику `module:function` или `module:Class`:

```yaml
data:
  factory: my_dataset:MyDataset
  kwargs:
    root: /path/to/images
```

Фабрика возвращает индексируемый объект со следующими свойствами:

```python
from bridge_restore.data import CleanSample

class MyDataset:
    def __init__(self, root):
        self.sample_ids = ...  # уникальные стабильные строки
        self.group_ids = ...   # строки или None, тот же порядок

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        image = ...  # CPU float32 tensor [3,H,W], [0,1]
        return CleanSample(image, self.sample_ids[index], self.group_ids[index])
```

Metadata доступны без декодирования всех изображений. Для `mode: official` добавьте
`official_split_assignments = {sample_id: "train" | "val" | "test"}`.
Модуль должен быть импортируем в основном процессе и DataLoader workers.

Есть простой адаптер `bridge_restore.data.datasets:ImageFolderDataset`: `root` —
каталог изображений с вложенными папками; необязательный `groups_file` — JSON
`{"relative/path.jpg": "person-id"}`. ID — относительный путь к файлу.

### Своя деградация

```python
class MyCorruption:
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, image):
        ...  # вернуть CPU float32 RGB [0,1], той же формы

    def to_config(self):
        return {"strength": self.strength}
```

Укажите её фабрику и параметры в `corruption.factory` / `corruption.kwargs`.
Объект не изменяет вход и детерминирован при фиксированных параметрах. Деградация
с другим размером наблюдения пока требует явного восстановления размера в адаптере.
Общий Trainer не читает специфичные атрибуты JPEG; объект доступен через DataBundle.

## Математика и генерация

Используется время DDBM: x0 — чистый конец при t=0, xT — наблюдение при t=T.
T=`sigma_max`. Обучение выбирает t равномерно между `sigma_min` и T и строит

```text
x_t = (1-t²/T²) x0 + (t²/T²) xT + t sqrt(1-t²/T²) epsilon.
```

Денойзер использует DDBM preconditioning и вес `1/c_out²`; параметры масштаба
обоих концов равны `sigma_data`. `cov_xy` влияет на preconditioning, а не на
реальные пары. Изменение T меняет мост: максимальный std промежуточного шума T/2.
Для пилота зафиксированы T=0.5, JPEG QF=10 и center crop с bicubic resize до
128×128 (antialias включён). Одно значение T используется для BM и ABM.

Чистый SDE интегрируется в убывающем t:

```text
b(t,x) = 2/t * (x-D(x,t,[xT]))
g(t) = sqrt(2t)
x_next = x + b*dt + g*sqrt(abs(dt))*epsilon,  dt < 0.
```

Равномерная сетка состоит из `num_steps` интервалов от `T-start_offset` до
`sigma_min`. Начальное состояние приближённо равно наблюдению; после интегрирования
финальный вызов денойзера даёт реконструкцию. Поэтому NFE=`num_steps+1`.
Внутренние состояния не обрезаются. Terminal denoising и возможный clipping —
численные приближения, а не точная реализация непрерывного конечного закона.

```bash
bridge-sample --checkpoint runs/celeba_abm/checkpoints/last.pt \
  --output runs/celeba_abm/samples/val --split val --device cuda \
  --weights ema --seed 123 --steps 100 --limit 100
```

Можно запускать как `python -m bridge_restore.sample`. Чистый эталон не передаётся
в sampler. CLI пока использует парный датасет для получения наблюдений; напрямую
`SDESampler.sample(model, observation)` принимает только наблюдение.

Сохраняются float NPY в модельном диапазоне, PNG для просмотра, `index.jsonl`
с исходными ID и NFE, настройки генерации. При `clip_output: true` float-выдача
также обрезана. Для сохранения необрезанной выдачи отключите этот параметр.
PNG всегда обрезается и округляется. Seed воспроизводит шум для одинакового
порядка и batching; изменение batch size может изменить конкретные реконструкции.

## Структура и проверка

`data/` отвечает за источники, splits и деградации. `models/` — ADM и денойзер.
`bridge.py` содержит VE-формулы и objective. `training.py` управляет optimizer/EMA;
`sampling.py` — численным SDE. `checkpoint.py` сохраняет состояние эксперимента.

Тесты включают отсутствие identity leakage, JPEG-пары, моменты bridge-маргиналей,
conditioning, VJP замороженной сети, дисперсию SDE-шага, NFE, обучение обоих режимов,
точный CPU resume на детерминированных данных и накопление неравных batch.
Проверка на CPU не заменяет profiling/проверку mixed precision на A100.

## Restoration evaluation

Install optional metrics with `pip install -e '.[eval]'`. Training configuration
`evaluation.enabled: true` enables restoration validation and best checkpoint
selection. Metrics are constructed once; only accumulated results are reset.
`unit: step` schedules successful optimizer updates; `unit: epoch` schedules
completed data passes. Accumulation never crosses an epoch boundary. A final
validation runs if the last model version has not yet been evaluated.

`subset_size: null` uses the entire split; otherwise a fixed subset is selected
with `subset_seed`. `weights: ema` uses a persistent separate inference model;
`raw` evaluates the training model without gradients. Generation consumes the
normal global RNG, so changing validation frequency can change training.
`final_test: true` evaluates `checkpoints/best.pt` after training. Resume restores
the best score and evaluation position; keep the run directory including best.pt.

Standalone evaluation:

```bash
python -m bridge_restore.evaluate --checkpoint runs/celeba_abm/checkpoints/best.pt \
  --config configs/evaluation/full.yaml --split test --device cuda --output results/test
```

The standalone config directly contains EvaluationConfig fields. `subset_size`
and `metrics` apply to the selected split. `test_subset_size` and `test_metrics`
are overrides used only by automatic post-training testing.

Supported names: mse, mae, psnr, ssim, ms_ssim, lpips, fid, kid,
observation_mse, observation_mae. PSNR is averaged per image in RGB with range 1.
Inputs are floating-point model tensors, not PNG files. Baseline excludes
observation consistency. FID/KID use TorchMetrics/torch-fidelity Inception, not
Clean-FID; do not mix scores from different backends. KID requires more examples
than subset_size. Default five-scale MS-SSIM requires larger images than 128;
configure fewer `betas` explicitly when using it at this resolution.

Reports contain summary.json, per_image.jsonl and protocol.json. Infinite PSNR
for exact equality is serialized as the string "inf". No generation timing or
CUDA memory measurements are performed. Learned metrics can download pretrained
weights on first use. GPU and pretrained-backend validation must be performed in
the experiment environment.

Fixed visual samples are versioned in `configs/visual_samples/celeba_val.json`
(16 images) and `celeba_test.json` (32 images). Selection uses a seed-42 shuffle
of sorted official split IDs, accepting unseen identities, validation first.
All 48 identities differ. The manifests record identities and metadata hashes.

Set `evaluation.visual_samples_path` for validation and
`evaluation.test_visual_samples_path` for the automatic final test. Standalone
evaluation uses `visual_samples_path` for its selected split: pilot.yaml points
to validation, full.yaml to test. Set the path to null to disable visualization.
Paths are relative to the working directory. Missing IDs, split mismatches,
duplicates and subsets smaller than the visual panel raise errors.

Visual IDs are always included in metric computation. Remaining subset slots
are filled deterministically from other IDs. Each report saves PNG sheets under
`images/panel_XX.png`, eight rows per sheet: clean, observation, reconstruction.
Image pixels retain their native resolution; only labels and margins are added.
Panels use existing reconstructions, with no additional sampler calls. The report
records visual IDs and their order. Visualization does not change metric tensors.

## BM time sweep without validation

```bash
pip install -e '.[eval,sweep]'
python -m bridge_restore.sweep --config configs/celeba_bm_T_sweep.yaml
# Run only one member:
python -m bridge_restore.sweep --config configs/celeba_bm_T_sweep.yaml --T 1.5
```

The dedicated sweep loop uses Trainer.train_step(), not Trainer.fit(). It performs
no intermediate or final validation and does not select a best checkpoint.
Each T starts independently from the same initialization seed. The configuration
uses BM, 18M parameters, FP16, microbatch 4, accumulation 4 (effective batch 16),
one data epoch, then evaluates final EMA weights on exactly the 32 fixed test IDs.
The final update of the epoch may contain fewer images. AMP-skipped updates
consume data but do not count as successful optimizer steps.

Outputs under runs/celeba_bm_T_sweep/T_<value>/ include metrics.jsonl (one record
per attempted update), training_loss.png (raw and moving mean), checkpoints/last.pt,
periodic checkpoints, test reports and visual panels, and result.json.
The sweep entry point starts fresh and refuses nonempty run directories; it does
not implement resume. Do not resume it via the ordinary training entry point,
which has different validation semantics. Set max_updates in the sweep YAML to
a positive number to cap the training of each member for a shorter comparison.
For A100, training.precision can be changed to bfloat16 in experiment.training;
FP16 is configured for compatibility with T4. Data paths are relative to cwd.

Training loss scales differ with T; compare restoration metrics as well as loss
curves. These 32 test examples are a diagnostic tuning set in this experiment,
not an independent final benchmark once used to choose T. The final evaluation
uses 100 stochastic updates and one final denoiser call for every T; larger T
therefore also changes the numerical time-grid spacing.
