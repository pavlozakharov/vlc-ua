# Передача: сесія Claude Code на сервері edrsr

Скопіювати цей файл цілком у першу репліку сесії `claude` на сервері.
Мета сесії: побудувати еталони для голів `attribution` і `departure_pair`
з реальних даних, прогнати keyword-бейзлайн, з'ясувати, який із наявних
LLM-каналів повертає logprobs, і віддати письмовий звіт із числами.
Нічого в прод не вмикати. Жодного запису в бази проєкту.

## Межі

- Читання: `~/.edrsr/edrsr.db`, `~/.edrsr/positions.db`, `~/laws_src/dep_gold.json`,
  дамп відкинутих речень екстрактора, якщо він є, `~/dag_pilot_state/slice.jsonl`,
  `/srv/work/sweep/*-classified.jsonl`. Усі бази відкривати лише `mode=ro`.
- Запис: тільки в `/srv/work/judge/` і в клон репозиторію `vlc-ua`.
- Ключі провайдерів живуть у `~/.secrets/<провайдер>.env`; експортувати у
  змінну оточення на час команди, ніколи не копіювати в репозиторій чи звіт.
- Матеріали клієнтських справ у цю роботу не входять: еталон атрибуції
  будується лише з публічних текстів ЄДРСР.
- Важкі прогони по прод-базі не в робочі години Павла.

## Крок 1. Репозиторій і пакет

```bash
mkdir -p /srv/work/judge && cd /srv/work/judge
git clone --branch claude/fervent-shannon-lz9k6j https://github.com/pavlozakharov/vlc-ua.git \
  || git -C vlc-ua pull
~/.edrsr/venv/bin/pip install -e ./vlc-ua
~/.edrsr/venv/bin/vlc-judge task attribution > attribution.task.json
~/.edrsr/venv/bin/vlc-judge task departure_pair > departure_pair.task.json
~/.edrsr/venv/bin/python -m pytest -q vlc-ua/tests
```

Далі `vlc-judge` означає `~/.edrsr/venv/bin/vlc-judge`.

## Крок 2. Схема edrsr.db

`gold/attribution.py:read_sqlite_texts` припускає таблицю з колонками
`doc_id`, `text`, `judgment`, `date`. Спершу перевірити:

```bash
sqlite3 "file:$HOME/.edrsr/edrsr.db?mode=ro" ".tables"
sqlite3 "file:$HOME/.edrsr/edrsr.db?mode=ro" ".schema documents"
```

Якщо таблиця або колонки називаються інакше, змінити SQL у
`read_sqlite_texts` у клоні і записати у звіт, що саме змінено. Повний
текст постанови може лежати в іншій таблиці, ніж картка; брати той стовпець,
який містить повний текст рішення ВС, а не конспект.

## Крок 3. Еталон атрибуції

```bash
vlc-judge gold-attribution --texts ~/.edrsr/edrsr.db --limit 2000 --per-doc 6 \
  --out attribution.jsonl
```

Команда друкує кількість фрагментів за видами: court, party, lower, facts,
procedural. Очікується кілька тисяч рядків. Якщо якогось виду менше сотні,
перевірити на десяти випадкових рішеннях, чи впізнає лексикон заголовків
у `gold/attribution.py:HEADER_RULES` реальні заголовки, і доповнити його,
записавши у звіт кожен доданий шаблон.

Контроль якості ярликів, обов'язковий: 30 випадкових рядків прочитати
очима і записати в `attribution.adjudication.jsonl` рядки виду
`{"id": "...", "gold": "court", "draw": "random"}`. Розбіжність ярлика із
заголовком означає дефект лексикону, а не дефект моделі.

## Крок 4. Еталон відступів

```bash
vlc-judge gold-departures \
  --dep-gold ~/laws_src/dep_gold.json \
  --positions-db ~/.edrsr/positions.db --limit 3000 \
  --rejects <шлях до дампу DUMP_REJECTS, якщо є> \
  --out departures.jsonl
```

Джерела і їхня довіра записані в кожному рядку в полі `source`: `lpd` це
офіційна розмітка ВС, `grammar` це регекс-шар із заміряною точністю 87–88 %,
`grammar-reject:<кошик>` це негативи з кошиків «заперечення» і «генерика».
Питання ставиться на пару «речення + одна ціль»: одне речення законно
перелічує кілька справ, і верифікатор на ціле речення якорить на першій.

## Крок 5. Keyword-бейзлайн

```bash
vlc-judge run --backend keyword --task attribution.task.json \
  --gold attribution.jsonl --out runs/kw-attribution.json
vlc-judge report runs/kw-attribution.json --task attribution.task.json \
  --gold attribution.jsonl > reports/kw-attribution.json

vlc-judge run --backend keyword --task departure_pair.task.json \
  --gold departures.jsonl --out runs/kw-departures.json
vlc-judge report runs/kw-departures.json --task departure_pair.task.json \
  --gold departures.jsonl > reports/kw-departures.json
```

Бейзлайн існує, щоб було з чим порівнювати: голова, яка не б'є регекс на
випадковому зрізі, у прод не йде.

## Крок 6. Проба logprobs на наявних каналах

Проба одним запитом показує, чи віддає канал справжній розподіл, чи лише
текст. Канали і ключі за вікі проєкту:

```bash
set -a; . ~/.secrets/groq.env; set +a
LLM_API_KEY="$GROQ_API_KEY" vlc-judge probe-logprobs \
  --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile

set -a; . ~/.secrets/openrouter.env; set +a
LLM_API_KEY="$OPENROUTER_API_KEY" vlc-judge probe-logprobs \
  --base-url https://openrouter.ai/api/v1 --model nvidia/nemotron-3-super-120b:free

set -a; . ~/.secrets/mistral.env; set +a
LLM_API_KEY="$MISTRAL_API_KEY" vlc-judge probe-logprobs \
  --base-url https://api.mistral.ai/v1 --model mistral-small-latest
```

Так само спробувати GLM і Cloudflare Workers AI через їхні
OpenAI-сумісні адреси, якщо ключі є. У звіт заносити рядок `verdict`
кожної проби. Канал із `"logprobs": true` придатний для logprob-бекенду;
канал з `degraded` годиться лише як одноточковий учитель для розмітки.

Якщо хоч один канал повертає logprobs, прогнати його на еталоні атрибуції
з обмеженням, щоб не з'їсти добову квоту:

```bash
LLM_API_KEY=... vlc-judge run --backend logprob --base-url <url> --model <model> \
  --task attribution.task.json --gold attribution.jsonl --limit 500 \
  --out runs/logprob-attribution.json
vlc-judge report runs/logprob-attribution.json --task attribution.task.json \
  --gold attribution.jsonl --baseline runs/kw-attribution.json > reports/logprob-attribution.json
```

Вичерпана квота або 429 з'являються у звіті як `failures`, це окремий клас
результату; документ із таким статусом не отримує вердикту.

## Крок 6-а. Необов'язково: jev як точка порівняння і учитель

Якщо є ключ TypeSafe у `~/.secrets/typesafe.env` (реєстрація самостійна на
console.typesafe.ai; ключ ніколи не копіювати в репозиторій чи звіт):

```bash
set -a; . ~/.secrets/typesafe.env; set +a
vlc-judge run --backend typesafe --model jev-1.13.0 --task attribution.task.json \
  --gold attribution.jsonl --limit 500 --out runs/jev-attribution.json
vlc-judge report runs/jev-attribution.json --task attribution.task.json \
  --gold attribution.jsonl --baseline runs/kw-attribution.json > reports/jev-attribution.json
```

Лише публічні тексти ЄДРСР; версію моделі пінити, не використовувати
алiас `jev-latest`. У звіт іде точність, ECE, поріг і wins/losses проти
keyword-бейзлайну: це перший замір jev на українському юридичному тексті.

## Крок 7. Звіт

Файл `/srv/work/judge/REPORT.md`, українською, без оцінок «добре/погано»,
лише числа з посиланнями на файли звітів:

1. Схема edrsr.db і що змінено в SQL.
2. Еталон атрибуції: рядків за видами; скільки з 30 прочитаних очима
   збіглися з ярликом заголовка.
3. Еталон відступів: рядків за `source` і за ярликом.
4. Для кожного прогону: `random.accuracy`, `random.ece`, `random.threshold`
   (поріг, покриття, досягнута точність), `latency_s`, `n_failures`,
   для logprob ще `vs_baseline.wins/losses`.
5. Результати проб logprobs по каналах.
6. Що не вдалося і чому, окремим списком.

Після звіту: `git add` еталонів не робити, вони лишаються у `/srv/work/judge/`;
у репозиторій комітити лише правки коду з поясненням у повідомленні коміту.
