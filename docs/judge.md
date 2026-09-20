Пакет judge: типізовані рішення та оцінка
==========================================

Пакет judge реалізує "типізовані голови рішень": функції для класифікації, які приймають стан та набір іменованих питань, повертаючи ймовірнісні розподіли за замкнутим набором опцій. Розподіл, а не текст — це базова одиниця. Самооцінка впевненості моделлю не використовується: впевненість виходить із форми розподілу (нормована ентропія).

Три типи питань:

* Noul: так/ні. Поле `p` — імовірність "так".
* Choice: один-з-N. Поля `choice` (найімовірніша опція), `confidence` (з ентропії розподілу).
* Score: впорядкована шкала. Поля `choice`, `confidence`, `score` (середнє значення індексу рівня).

Правила дизайну:

* Бекенд не вирішує нічого: пороги, прийняття/відхилення, ескалація на повний прочит — у коді викликавача.
* Температура за-питання, підібрана на утримуваному зрізі, робить розподіли відкаліброваними.
* Кожен ряд золота зберігає `source` — рядок, звідки взялась мітка (хедер, мовна розмітка, людина).
* Збагачений зріз лише для чутливості; пороги завжди підбираються на випадковому зрізі.

Формат gold (JSONL, один об'єкт на рядок):

```json
{"id": "doc1:0:600", "state": {"fragment": "текст..."}, 
 "question": "attribution", "gold": "court", 
 "sample": "random", "source": "header:Позиція Верховного Суду"}
```

Поля:
- `id`: унікальний ідентифікатор елемента
- `state`: рядок, словник або список — контекст для питання
- `question`: ім'я питання (має бути у файлі task)
- `gold`: правильна опція (ключ з options питання)
- `sample`: "random" або "enriched"; випадковий зріз для порогів та звітів
- `source`: звідки мітка (для трасування поганих міток назад до джерела)

Формат task (JSON):

```json
{
  "attribution": {
    "type": "choice",
    "instructions": "Чиїм голосом цей фрагмент?",
    "criteria": {
      "court": "суд розмірковує",
      "party": "аргументи сторін",
      "lower": "рішення нижчої інстанції",
      "facts": "фактичні обставини",
      "procedural": "процесуальні дії"
    }
  }
}
```

Команди CLI (в порядку):

```bash
vlc-judge task attribution > gold/attribution.task.json
vlc-judge gold-attribution --texts edrsr.db --out gold/attribution.jsonl --limit 2000
vlc-judge run --backend keyword --task gold/attribution.task.json \
  --gold gold/attribution.jsonl --out runs/keyword.json
vlc-judge run --backend logprob --base-url http://127.0.0.1:8000/v1 \
  --model qwen3 --task gold/attribution.task.json \
  --gold gold/attribution.jsonl --out runs/logprob.json
vlc-judge run --backend crossencoder --model-dir heads/attribution \
  --task gold/attribution.task.json --gold gold/attribution.jsonl \
  --out runs/crossencoder.json
vlc-judge report runs/crossencoder.json --task gold/attribution.task.json \
  --gold gold/attribution.jsonl --baseline runs/keyword.json
vlc-judge serve --backend crossencoder --model-dir heads/attribution --port 8009
```

Звіт (JSON) розділяє, що правила не дозволяють змішувати:

**random (точність, калібрування, черга):**
- `n_random`: кількість рядків зі sample="random"
- `accuracy`: точність на тестовій половині випадкового зрізу
- `ece`: очікувана помилка калібрування (0.02 добре, 0.15 погано)
- `brier`: середньоквадратична помилка розподілу
- `temperature`: скаляр, підібраний на dev половині
- `threshold`: об'єкт з threshold, coverage (доля прийнятих), achieved_precision для цільової точності (за замовчуванням 0.95)
- `confusion`: матриця gold → predicted → count

**enriched (чутливість):**
- `sensitivity_accuracy`, `ece`, `confusion`
- Примітка у звіті: пороги, підібрані на збагаченому зрізі, не переносяться на реальний потік

**vs_baseline (порівняння):**
- `wins`, `losses`, `both_right`, `both_wrong`
- ID перших 50 перемог/поразок для дебагу

**failures:**
- `n_failures`: помилки провайдера (quota, network)
- Мають обробляти як окремий клас, не змішувати з неправильною класифікацією

Таблиця бекендів:

| Бекенд | Потребує | Обмеження |
|--------|----------|-----------|
| keyword | список питань в коді | Лише те, що у правилах; не вивчає |
| logprob | LLM_API_KEY, --base-url, --model | Ніколи верифікацію цитат; деградує до one-hot, якщо канал не віддає logprobs |
| crossencoder | model-dir з head.json, model.onnx, tokenizer.json | Лише питання, на яких навчали; CPU-only |
| typesafe | TYPESAFE_API_KEY | учитель для розмітки і точка порівняння; не робочий шлях |
| cloudflare | CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN | учитель; контекст на цьому маршруті 32K |
| systemone-http | --base-url | Будь-який локальний сервіс System One |

Ніколи не використовувати для: верифікації цитат (розподіл не доводить наявності символів у файлі), синтезу правової позиції, розрахунку дедлайнів (функція часу).

Для departures та screening: золото вимагає файлів, що живуть лише на сервері (positions.db, dep_gold.json, rejects dump, DAG slice, sweep_classify outputs).
