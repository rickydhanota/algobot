#!/bin/bash
# Auto-launched by cron at 6:25 AM PT (9:25 AM ET) on weekdays.
# Waits for market open, then starts the bot. Logs to bot.log.

BOT_DIR="$HOME/Desktop/trading_bot"
LOG="$BOT_DIR/bot.log"
PYTHON="$BOT_DIR/.venv/bin/python"

echo "=== $(date) | Launch triggered ===" >> "$LOG"

# Skip if already running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "Bot already running, exiting." >> "$LOG"
    exit 0
fi

# Skip weekends (belt-and-suspenders in case cron fires on a holiday edge case)
DOW=$(date +%u)
if [ "$DOW" -ge 6 ]; then
    echo "Weekend — skipping." >> "$LOG"
    exit 0
fi

cd "$BOT_DIR" || exit 1
"$PYTHON" main.py >> "$LOG" 2>&1
echo "=== $(date) | Bot exited ===" >> "$LOG"
