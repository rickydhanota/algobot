#!/bin/bash
# Double-click this file from Finder to launch the bot.
# A Terminal window opens showing live log output.

cd "$(dirname "$0")"

if pgrep -f "python.*main.py" > /dev/null; then
  echo "AlgoBot is already running."
  echo "PID: $(pgrep -f 'python.*main.py')"
  echo ""
  echo "Press Ctrl+C to stop tailing the log, or close this window."
  echo ""
  tail -f bot.log
  exit 0
fi

echo "Starting AlgoBot..."
./launch.sh &
BOT_PID=$!
echo "Bot launched (PID $BOT_PID). Tailing log..."
echo ""
sleep 2
tail -f bot.log
