#!/bin/bash
# One-time setup: create venv, install deps, scaffold .env

set -e
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "Created .env — add your Alpaca paper-trading API keys:"
  echo "  ALPACA_API_KEY=..."
  echo "  ALPACA_SECRET_KEY=..."
  echo ""
  echo "Get keys at: https://app.alpaca.markets (paper trading is free)"
fi

echo "Setup complete. To run:"
echo "  source .venv/bin/activate"
echo "  python main.py"
