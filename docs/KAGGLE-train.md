# Навчання кросенкодерної голови на Kaggle T4

Голова це `BAAI/bge-reranker-v2-m3`, дотренована так, щоб для кожного
варіанта відповіді повертати логіт; softmax по варіантах дає розподіл,
температура з `head.json` робить його каліброваним. Той самий клас моделі
вже працює у проді як реранкер в ONNX int8 на CPU, тому подача голови на
сервер не потребує GPU.

Перший прогін 20.09.2026 упав з CUDA OOM: 559M параметрів у fp32 з AdamW не
вміщуються в T4. Прапорці нижче це наслідок того замiру, не побажання.

## Крок 1. Дані

На сервері, перед вивантаженням, обов'язково знеособити:

```bash
vlc-judge scrub-gold --in attribution.jsonl --out attribution.scrubbed.jsonl
```

Скрипт замінює повні імена, ініціали з прізвищем і коди; номери справ,
дати, суми, «Велика Палата» і ОСОБА_N лишаються, і це перевіряється
лічильниками. Створити Kaggle Dataset `vlc-gold` з файлів
`attribution.scrubbed.jsonl` і `attribution.task.json`.

## Крок 2. Кернел

Прискорювач `NvidiaTeslaT4`. Комірки:

```bash
pip install -q "transformers>=4.40" onnxruntime onnxscript
git clone --branch claude/fervent-shannon-lz9k6j https://github.com/pavlozakharov/vlc-ua.git /kaggle/working/vlc-ua
pip install -q -e /kaggle/working/vlc-ua
```

```bash
cd /kaggle/working && python -m vlc_ua.judge.train.crossencoder \
  --gold /kaggle/input/vlc-gold/attribution.scrubbed.jsonl \
  --task /kaggle/input/vlc-gold/attribution.task.json \
  --base BAAI/bge-reranker-v2-m3 \
  --out /kaggle/working/head-attribution \
  --epochs 2 --lr 2e-5 --batch-rows 4 --dev-share 0.3 \
  --freeze-embeddings --grad-checkpointing --max-hours 10 \
  --export-onnx
```

Що означають прапорці:

- `--max-length` за замовчуванням 512: найдовша пара «запит + фрагмент» в
  еталоні 422 токени, 1024 лише подвоїла б паддинг.
- fp16 увімкнено за замовчуванням (`--no-fp16` вимикає): без тензорних ядер
  T4 не встигає за 12 годин.
- `--freeze-embeddings`: матриця ембеддингів це 256M із 559M параметрів,
  реранкерній голові її перенавчати нема чого; звільняє близько 3,6 ГБ.
- `--grad-checkpointing`: решта пам'яті. Разом із заморозкою скрипт вмикає
  `enable_input_require_grads`, інакше крок проходить, а ваги не рухаються;
  тест `test_server_fixes.py` це перевіряє.
- `--max-hours`: зупинка за бюджетом із калібруванням і записом `head.json`;
  кернел, убитий на стіні 12 годин, не лишає нічого.
- `--limit-rows N` для швидкої проби перед повним прогоном.

Скрипт розбиває кожен рядок на пари «інструкція + значення варіанта» проти
«стан», навчає listwise-крос-ентропією одним forward на пачку, відкладає
30 % випадкових рядків, на них підбирає температуру і пише метрики.
Збагачені рядки навчають, але не калібрують.

## Крок 3. Результат

Тека `head-attribution/`: ваги, токенізатор, `model.onnx`, `tokenizer.json`,
`head.json` із `temperatures`, `dev_metrics` і блоком `train` (рядки, епохи,
lr, fp16, заморозка, години, чи зупинив бюджет). Якщо експорт ONNX упав через
відсутній `onnxscript`, ваги і `head.json` уже на диску; доекспортувати
можна пізніше, у тій самій теці.

Квантування для CPU у тому ж кернелі:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("/kaggle/working/head-attribution/model.onnx",
                 "/kaggle/working/head-attribution/model.onnx", weight_type=QuantType.QInt8)
```

Holdout можна поміряти прямо в кернелі на GPU тим самим харнесом, без
власного циклу оцінки:

```bash
vlc-judge run --backend crossencoder --runtime torch --model-dir /kaggle/working/head-attribution \
  --task /kaggle/input/vlc-gold/attribution.task.json \
  --gold /kaggle/input/vlc-gold/attribution-holdout.scrubbed.jsonl --out runs/ce-holdout.json
```

Зберегти версію кернела; результат забрати CLI, як у проєктному рунбуку
Kaggle: `kaggle kernels output <user>/<kernel> -p /srv/work/judge/heads/`.

## Крок 4. Перевірка на сервері

```bash
~/.edrsr/venv/bin/pip install -q onnxruntime tokenizers
vlc-judge run --backend crossencoder --runtime onnx --model-dir /srv/work/judge/heads/head-attribution \
  --task attribution.task.json --gold attribution-holdout.jsonl --out runs/ce-holdout.json
vlc-judge report runs/ce-holdout.json --task attribution.task.json \
  --gold attribution-holdout.jsonl --baseline runs/kw-holdout.json > reports/ce-holdout.json
```

Holdout будується командою `gold-attribution --skip N` з рішень, яких
навчальний зріз не бачив; це і є чесний замір, dev-половина в `head.json`
лише орієнтир.

## Крок 5. Правило допуску

Голова ставиться поруч із евристикою verify_quote як «ознака, не вердикт»
лише якщо на holdout одночасно:

1. точність вища за keyword-бейзлайн і `vs_baseline.losses` розібрані очима;
   якщо розбір показує помилки еталона, спочатку лексикон, потім перезбірка
   і повторне навчання, так уже було 21.09;
2. `random.ece` не більше 0,05;
3. покриття при порозі на точність 0,95 не нижче за бейзлайн.

Сервінг після допуску:

```bash
vlc-judge serve --backend crossencoder --model-dir /srv/work/judge/heads/head-attribution --port 8009
```

Будь-який клієнт контракту System One, включно з офіційним SDK із
базовою адресою `http://127.0.0.1:8009/v1/systemone`, працює з цією головою.
