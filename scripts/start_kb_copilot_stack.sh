#!/bin/sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-5055}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

echo "== KB Copilot stack =="
echo "Root: $ROOT"
echo "Port: $PORT"
echo "Ollama: $OLLAMA_HOST"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "Missing venv. Run:"
  echo "  cd \"$ROOT\" && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo
echo "== Ollama model checklist =="
if curl -fsS "$OLLAMA_HOST/api/tags" >/tmp/kb_copilot_ollama_tags.json 2>/dev/null; then
  for m in llama3.1:8b llama3.2:3b qwen2.5:7b qwen2.5:3b mistral; do
    if rg -q "\"name\"\\s*:\\s*\"$m\"" /tmp/kb_copilot_ollama_tags.json; then
      echo "  [ok] $m"
    else
      echo "  [missing] $m   -> ollama pull $m"
    fi
  done
else
  echo "  Ollama not reachable at $OLLAMA_HOST"
  echo "  Start Ollama first: ollama serve"
fi

echo
echo "== Starting KB Copilot =="
exec "$ROOT/.venv/bin/python" app.py
