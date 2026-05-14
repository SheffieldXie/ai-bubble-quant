#!/bin/bash
# install_cron.sh - Install daily market check as a cron job
# Usage: ./install_cron.sh [time]
# Default time: 9:00 AM (market open)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"
LOG_DIR="$PROJECT_DIR/logs"

# Default: 9:00 AM Monday-Friday (market days)
CRON_TIME="${1:-0 9 * * 1-5}"

echo "========================================="
echo "📊 AI Bubble Quant — Cron Job Setup"
echo "========================================="
echo ""
echo "Project: $PROJECT_DIR"
echo "Schedule: $CRON_TIME (Mon-Fri)"
echo "Python: $PYTHON"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Remove existing cron job if present
crontab -l 2>/dev/null | grep -v "ai-bubble-quant/daily_check" > /tmp/crontab_old
crontab /tmp/crontab_old
rm /tmp/crontab_old

# Add new cron job
(crontab -l 2>/dev/null; echo "# AI Bubble Quant Daily Check") >> /tmp/crontab_new
(crontab -l 2>/dev/null | grep -v "ai-bubble-quant") >> /tmp/crontab_new
echo "$CRON_TIME cd $PROJECT_DIR && $PYTHON daily_check.py >> $LOG_DIR/cron.log 2>&1" >> /tmp/crontab_new
crontab /tmp/crontab_new
rm /tmp/crontab_new

echo ""
echo "✅ Cron job installed successfully!"
echo ""
echo "To verify:"
echo "  crontab -l | grep ai-bubble-quant"
echo ""
echo "To manually run:"
echo "  cd $PROJECT_DIR && python3 daily_check.py"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/daily_check_*.log"
echo ""
echo "To remove:"
echo "  crontab -e  # and delete the ai-bubble-quant line"
