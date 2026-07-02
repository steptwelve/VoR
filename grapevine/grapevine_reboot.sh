#!/bin/bash
# grapevine_reboot.sh
# Runs at @reboot — checks if today's Grapevine quote has already been posted.
# If not, waits for network then runs the poster.
# This handles the case where lsm was down at 9 AM cron time.
#
# Version: 1.0
# Created: 2026-07-02
# Author:  Jackson Shaw with Claude (Anthropic)

GUARD_DIR="/home/jackson/grapevine"
TODAY=$(date +%Y-%m-%d)
GUARD_FILE="$GUARD_DIR/.posted_$TODAY"
LOG="$GUARD_DIR/grapevine.log"

echo "[@reboot] Grapevine reboot check — $TODAY" >> "$LOG"

# If already posted today, do nothing
if [ -f "$GUARD_FILE" ]; then
    echo "[@reboot] Already posted today, skipping." >> "$LOG"
    exit 0
fi

# Wait for network (up to 60 seconds)
for i in $(seq 1 12); do
    if ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
        echo "[@reboot] Network up after ${i}x5s" >> "$LOG"
        break
    fi
    sleep 5
done

# Wait an extra 10 seconds for services to settle
sleep 10

echo "[@reboot] Quote not posted yet — running grapevine_poster.py" >> "$LOG"
/usr/bin/python3 /home/jackson/grapevine/grapevine_poster.py >> "$LOG" 2>&1
