# Навчання кросенкодерної голови на Kaggle T4

Голова це `BAAI/bge-reranker-v2-m3`, дотренована так, щоб для кожного
варіанта відповіді повертати логіт; softmax по варіантах дає розподіл,
температура з `head.json` робить його каліброваним. Той самий клас моделі
вже працює у проді як реранкер в ONNX int8 на CPU, тому подача голови на
сервер не потребує GPU.

## Крок 1. Дані

Створити Kaggle Dataset `vlc-gold` з двох файлів, зібраних на сервері за
`docs/HANDOFF-server.md`:

- `attribution.jsonl` з рядками `sample` random і enriched;
- `attribution.task.json`.

Персональних даних у фрагментах постанов ВС уникати: ПІБ фізосіб перед
викладенням вичистити тим самим знеособлювачем, що в проєкті.

## Крок 2. Ядро кернела

Кернел з прискорювачем `NvidiaTeslaT4`. Комірки:

```bash
pip install -q "transformers>=4.40" onnxruntime
git clone --branch claude/fervent-shannon-lz9k6j https://github.com/pavlozakharov/vlc-ua.git /kaggle/working/vlc-ua
pip install -q -e /kaggle/working/vlc-ua
```

```bash
cd /kaggle/working && python -m vlc_ua.judge.train.crossencoder \
  --gold /kaggle/input/vlc-gold/attribution.jsonl \
  --task /kaggle/input/vlc-gold/attribution.task.json \
  --base BAAI/bge-reranker-v2-m3 \
  --out /kaggle/working/head-attribution \
  --epochs 2 --lr 2e-5 --batch-rows 4 --max-length 1024 --dev-share 0.3 \
  --export-onnx
```

Пам'ять T4 на 16 ГБ: пачка з 4 рядків по 5 варіантів при довжині 1024
проходить; при OOM зменшити `--batch-rows` до 2 або `--max-length` до 768.
Довгі фрагменти обрізаються токенізатором, еталон будує їх до 1200 знаків.

Що робить скрипт: розбиває кожен рядок на пари «інструкція + значення
варіанта» проти «стан», навчає listwise-крос-ентропією по варіантах рядка,
відкладає 30 % випадкових рядків, на них підбирає температуру на кожне
питання і пише метрики. Збагачені рядки навчають, але ніколи не калібрують.

## Крок 3. Що в результаті

Тека `head-attribution/`: ваги, токенізатор, `model.onnx`, `tokenizer.json`,
`head.json`:

```json
{"base": "BAAI/bge-reranker-v2-m3", "task": "attribution.task.json",
 "temperatures": {"attribution": 1.2},
 "dev_metrics": {"attribution": {"n_dev": 300, "accuracy": 0.9, "ece": 0.03,
                                  "ece_uncalibrated": 0.08, "temperature": 1.2}},
 "max_length": 1024}
```

Числа тут ілюстративні. Квантування для CPU, у тому ж кернелі:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("/kaggle/working/head-attribution/model.onnx",
                 "/kaggle/working/head-attribution/model.onnx", weight_type=QuantType.QInt8)
```

Зберегти версію кернела; результат забрати CLI, як у проєктному рунбуку
Kaggle: `kaggle kernels output <user>/<kernel> -p /srv/work/judge/heads/`.

## Крок 4. Перевірка на сервері

```bash
~/.edrsr/venv/bin/pip install -q onnxruntime tokenizers
vlc-judge run --backend crossencoder --model-dir /srv/work/judge/heads/head-attribution \
  --task attribution.task.json --gold attribution.jsonl --out runs/ce-attribution.json
vlc-judge report runs/ce-attribution.json --task attribution.task.json \
  --gold attribution.jsonl --baseline runs/kw-attribution.json > reports/ce-attribution.json
```

Звіт рахує метрики на тестовій половині випадкового зрізу з температурою,
підібраною на dev-половині, тобто на рядках, яких навчання не бачило лише
в частині dev. Чесніше мати окремий еталон, зібраний після навчання, з
рішень, яких не було в навчальному наборі; для цього на сервері зібрати
`attribution-holdout.jsonl` з іншого діапазону дат і звітувати саме по ньому.

## Крок 5. Правило допуску

Голова ставиться поруч із евристикою verify_quote як «ознака, не вердикт»
лише якщо на випадковому зрізі одночасно:

1. точність вища за keyword-бейзлайн і `vs_baseline.losses` розібрані очима;
2. `random.ece` не більше 0,05;
3. покриття при порозі на точність 0,95 не нижче за бейзлайн.

Інакше голова не вмикається: більше еталона, перевірка лексикону
заголовків, інші гіперпараметри. Сервінг після допуску:

```bash
vlc-judge serve --backend crossencoder --model-dir /srv/work/judge/heads/head-attribution --port 8009
```

Будь-який клієнт контракту System One, включно з офіційним SDK із
базовою адресою `http://127.0.0.1:8009/v1/systemone`, працює з цією головою.
