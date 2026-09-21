# Передача: сесія Claude Code на сервері edrsr

Скопіювати цей файл цілком у першу репліку сесії `claude` на сервері.
Стан нижче звірено з комітами серверної сесії 20–21.09.2026 у гілці
`claude/fervent-shannon-lz9k6j`. Нічого в прод не вмикати. Жодного запису
в бази проєкту.

## Межі

- Читання: `~/.edrsr/edrsr.db`, `~/.edrsr/positions.db`, `~/laws_src/dep_gold.json`,
  дампи відкинутих речень `~/.edrsr/dep_rejects.jsonl` і `dep_g_rejects.jsonl`,
  `~/dag_pilot_state/slice.jsonl`, `/srv/work/sweep/*-classified.jsonl`.
  Усі бази відкривати лише `mode=ro`.
- Запис: тільки в `/srv/work/judge/` і в клон репозиторію `vlc-ua`.
- Ключі провайдерів живуть у `~/.secrets/<провайдер>.env`; експортувати у
  змінну оточення на час команди, ніколи не копіювати в репозиторій чи звіт.
- Файл, що залишає сервер (Kaggle), проходить `vlc-judge scrub-gold`; у звіт
  іде кількість замін і контрольні лічильники, що не змінилися.
- Важкі прогони по прод-базі не в робочі години Павла.

## Що вже зроблено (20–21.09.2026, серверна сесія)

| Крок | Стан | Де лежить |
|---|---|---|
| Пакет встановлено, тести | зроблено | `/srv/work/judge/vlc-ua` |
| Схема edrsr.db | `documents(full_text, judgment_code→judgment_forms, adjudication_date)`; добір за `rowid DESC` через `idx_doc_filter`, бо індексу на дату немає | `gold/attribution.py:read_sqlite_texts` |
| Еталон атрибуції | v5: 1 909 постанов, 10 156 рядків; п'ять ітерацій лексикону заголовків, контроль читанням 27/30 | `/srv/work/judge/attribution.jsonl` |
| Знеособлення перед вивантаженням | `scrub-gold`: імена, ініціали з прізвищем, коди; номери справ, дати, суми, ОСОБА_N не змінені | `gold/scrub.py` |
| Голова атрибуції | навчена на Kaggle T4 (fp16, заморожені ембеддинги, listwise), ONNX-експорт | `/srv/work/judge/heads/…` |
| Holdout | окремий зріз через `--skip`, 2 894 рядки, оцінка через torch на GPU | `reports/…` |
| Розбір програшів голови | 76 рядків проти keyword; шість найупевненіших виявились помилками еталона, лексикон виправлено | коміти c78e63e, 6bab067 |
| Еталон відступів | будується з `--evidence`: мітка з самого речення; 23/26 проти 14/26 у вихідних міток | `gold/departures.py:evidence_label` |
| Канали logprobs | Groq і Cohere відмовляють параметр, працюють як one-hot учителі; проба це показує | `probe.py`, `backends/logprob.py` |

Числа точності, ECE, порогів і покриття брати лише з файлів у
`/srv/work/judge/reports/`, не з пам'яті сесії.

## Крок 1. Оновити пакет

```bash
cd /srv/work/judge && git -C vlc-ua pull
~/.edrsr/venv/bin/pip install -q -e ./vlc-ua
~/.edrsr/venv/bin/python -m pytest -q vlc-ua/tests
```

Далі `vlc-judge` означає `~/.edrsr/venv/bin/vlc-judge`, робоча тека `/srv/work/judge`.

## Крок 2. Перезбірка еталона після правок лексикону

Після кожної правки `HEADER_RULES` еталон і holdout перезбираються тим самим
кодом, інакше голова навчається на одному, а міряється на іншому:

```bash
vlc-judge gold-attribution --texts ~/.edrsr/edrsr.db --limit 2000 --per-doc 6 \
  --out attribution.jsonl
vlc-judge gold-attribution --texts ~/.edrsr/edrsr.db --skip 3000 --limit 600 --per-doc 6 \
  --out attribution-holdout.jsonl
vlc-judge scrub-gold --in attribution.jsonl --out attribution.scrubbed.jsonl
vlc-judge scrub-gold --in attribution-holdout.jsonl --out attribution-holdout.scrubbed.jsonl
```

Версію еталона фіксувати у назві теки звіту (`reports/v6/…`), щоб числа
різних версій не змішувались. Контроль читанням: 30 випадкових рядків нової
версії, розбіжність ярлика із заголовком це дефект лексикону.

## Крок 3. Бейзлайн і голова на holdout

```bash
vlc-judge run --backend keyword --task attribution.task.json \
  --gold attribution-holdout.jsonl --out runs/v6/kw-holdout.json
vlc-judge run --backend crossencoder --runtime onnx --model-dir heads/<остання голова> \
  --task attribution.task.json --gold attribution-holdout.jsonl --out runs/v6/ce-holdout.json
vlc-judge report runs/v6/ce-holdout.json --task attribution.task.json \
  --gold attribution-holdout.jsonl --baseline runs/v6/kw-holdout.json > reports/v6/ce-holdout.json
```

На CPU через ONNX holdout на 2 894 рядки займає години; якщо є GPU-сесія,
`--runtime torch` там, де стоїть torch. Правило допуску голови незмінне:
точність вища за keyword, ECE не більше 0,05, покриття при порозі 0,95 не
нижче за бейзлайн, програші прочитані очима.

## Крок 4. Еталон відступів і бейзлайн

```bash
vlc-judge gold-departures --evidence \
  --dep-gold ~/laws_src/dep_gold.json \
  --positions-db ~/.edrsr/positions.db --limit 3000 \
  --rejects ~/.edrsr/dep_rejects.jsonl \
  --out departures.jsonl
vlc-judge run --backend keyword --task departure_pair.task.json \
  --gold departures.jsonl --out runs/v6/kw-departures.json
vlc-judge report runs/v6/kw-departures.json --task departure_pair.task.json \
  --gold departures.jsonl > reports/v6/kw-departures.json
```

Без `--evidence` мітки успадковують три відомі дефекти джерел, вони описані
в докстрінгу `evidence_label`. Адьюдикація 30 випадкових пар читанням, файл
`departures.adjudication.jsonl` з полем `draw: random`, потім перезбірка з
`--adjudication`.

## Крок 5. jev як точка порівняння і учитель

Ключ у `~/.secrets/typesafe.env`; якщо його ще нема, `bash vlc-ua/docs/wiki/apply-typesafe.sh`
спитає його з термінала і заодно поставить нотатку у вікі.

```bash
set -a; . ~/.secrets/typesafe.env; set +a
vlc-judge run --backend typesafe --model jev-1.13.0 --task attribution.task.json \
  --gold attribution-holdout.jsonl --limit 5 --out runs/v6/jev-probe.json --cache .judge-cache-jev
head -c 800 runs/v6/jev-probe.json
vlc-judge run --backend typesafe --model jev-1.13.0 --task attribution.task.json \
  --gold attribution-holdout.jsonl --limit 500 --out runs/v6/jev-holdout.json --cache .judge-cache-jev
vlc-judge report runs/v6/jev-holdout.json --task attribution.task.json \
  --gold attribution-holdout.jsonl --baseline runs/v6/kw-holdout.json > reports/v6/jev-holdout.json
```

Проба на п'яти рядках показує, чи прийнято ім'я `jev-1.13.0` і чи є
`confidence` у відповіді; заголовок `X-Zero-Data-Retention` бекенд шле, назву
в документації не звірено, тож ZDR вмикати ще й у консолі. Лише публічні
тексти ЄДРСР. Той самий прогін на `departures.jsonl` із задачею
`departure_pair`. Після прогону ключ перевипустити: він побував у чаті.

## Крок 6. Звіт

Файл `/srv/work/judge/REPORT.md`, українською, без оцінок, лише числа з
посиланнями на файли звітів:

1. Версія еталона, рядків за видами, контроль читанням.
2. Holdout: keyword, голова, jev, кожен з `random.accuracy`, `random.ece`,
   `random.threshold`, `latency_s`, `n_failures`, `vs_baseline`.
3. Відступи: рядків за `source` і ярликом, бейзлайн, jev.
4. Проби каналів logprobs.
5. Що не вдалося і чому, окремим списком.

Еталони в git не додавати, вони лишаються у `/srv/work/judge/`; у репозиторій
комітити лише правки коду з поясненням у повідомленні коміту.
