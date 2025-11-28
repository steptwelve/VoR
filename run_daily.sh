#!/bin/bash
cd /Users/jackson/Documents/GitHub/daily
source venv/bin/activate
python3 daily_poster.py "$@"
deactivate