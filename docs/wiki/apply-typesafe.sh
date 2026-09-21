#!/usr/bin/env bash
# Ставить канал TypeSafe jev на сервер: ключ у ~/.secrets, нотатка у вікі,
# рядки в переліках дешевих моделей, синхронізація вікі.
#
#   bash docs/wiki/apply-typesafe.sh            # ключ спитає з терміналу, не з аргументів
#
# Ключ ніколи не передається аргументом і не пишеться в журнал.
set -euo pipefail

WIKI="${WIKI:-$HOME/Wiki}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SECRETS="$HOME/.secrets"
ENVF="$SECRETS/typesafe.env"

# 1. ключ
mkdir -p "$SECRETS" && chmod 700 "$SECRETS"
if [ -s "$ENVF" ]; then
  echo "ключ уже є: $ENVF (не перезаписую)"
else
  read -r -s -p "TYPESAFE_API_KEY: " KEY; echo
  [ -n "$KEY" ] || { echo "порожній ключ, вихід"; exit 1; }
  umask 077
  printf 'TYPESAFE_API_KEY=%s\n' "$KEY" > "$ENVF"
  chmod 600 "$ENVF"
  unset KEY
  echo "записано $ENVF"
fi

# 2. нотатка у вікі
NOTE_SRC="$HERE/01_Projects/TypeSafe-jev-канал.md"
NOTE_DST="$WIKI/01_Projects/TypeSafe-jev-канал.md"
if [ -e "$NOTE_DST" ]; then
  echo "нотатка вже є: $NOTE_DST (не перезаписую; порівняйте з $NOTE_SRC)"
else
  cp "$NOTE_SRC" "$NOTE_DST" && echo "нотатку додано: $NOTE_DST"
fi

# 3. рядок у таблиці «Ролі дешевих моделей»
PLAN="$WIKI/01_Projects/План-підвищення-якості-пошуку-2026-09-19.md"
ROW='| TypeSafe jev | Учитель слабкої розмітки для голів judge і точка порівняння на еталоні: типізовані рішення, не текст | Лише через vlc-judge на випадковому зрізі; звіт із wins/losses проти keyword-бейзлайну; у робочий пошук не вводити |'
if [ -f "$PLAN" ] && ! grep -qF 'TypeSafe jev' "$PLAN"; then
  python3 - "$PLAN" "$ROW" <<'PY'
import sys, pathlib
p, row = pathlib.Path(sys.argv[1]), sys.argv[2]
s = p.read_text(encoding="utf-8")
anchor = "| Cloudflare |"
i = s.find(anchor)
if i < 0:
    sys.exit("рядок Cloudflare у таблиці не знайдено, рядок jev не додано")
j = s.find("\n", i)
p.write_text(s[:j + 1] + row + "\n" + s[j + 1:], encoding="utf-8")
print("рядок додано у таблицю «Ролі дешевих моделей»")
PY
else
  echo "таблиця «Ролі дешевих моделей»: рядок уже є або файл відсутній"
fi

# 4. рядок у таблиці провайдерів «Пул безкоштовних агентів»
POOL="$WIKI/01_Projects/Безкоштовні-агенти.md"
ROW2='| **TypeSafe jev** (платний, $0,042/1M вхідних) | не заміряно на укр. | заявлено 70–500 мс, незалежно ~1,4 с | ключ з 21.09.2026, див. [[TypeSafe-jev-канал]] |'
if [ -f "$POOL" ] && ! grep -qF 'TypeSafe jev' "$POOL"; then
  python3 - "$POOL" "$ROW2" <<'PY'
import sys, pathlib
p, row = pathlib.Path(sys.argv[1]), sys.argv[2]
s = p.read_text(encoding="utf-8")
anchor = "| Gemini 2.0 Flash |"
i = s.find(anchor)
if i < 0:
    sys.exit("рядок Gemini у таблиці провайдерів не знайдено, рядок jev не додано")
j = s.find("\n", i)
p.write_text(s[:j + 1] + row + "\n" + s[j + 1:], encoding="utf-8")
print("рядок додано у таблицю провайдерів")
PY
else
  echo "таблиця провайдерів: рядок уже є або файл відсутній"
fi

# 5. лінт і синхронізація вікі, як у проєкті
if [ -x "$HOME/power-venv/bin/python" ] && [ -f "$HOME/laws_src/wiki_power_lint.py" ]; then
  "$HOME/power-venv/bin/python" "$HOME/laws_src/wiki_power_lint.py" || true
fi
if [ -x "$HOME/bin/wiki-sync" ]; then
  "$HOME/bin/wiki-sync"
else
  echo "wiki-sync не знайдено: закомітьте зміни у $WIKI вручну"
fi
