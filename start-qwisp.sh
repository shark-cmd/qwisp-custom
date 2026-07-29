#!/bin/bash
# Start qwisp server directly (no Nginx proxy needed)
# Accessible on both localhost and Tailscale network

set -e

QWISP_PORT=9080
TAILSCALE_IP=$(tailscale ip 2>/dev/null | head -1)

echo "=== Qwisp Server ==="
echo ""

# Kill Nginx if running
if pgrep -x nginx > /dev/null; then
    echo "Stopping Nginx..."
    brew services stop nginx 2>/dev/null || pkill -9 nginx 2>/dev/null || true
    sleep 1
fi

# Kill any existing qwisp process
if pgrep -f "qwisp serve" > /dev/null; then
    echo "Stopping existing qwisp..."
    pkill -f "qwisp serve" 2>/dev/null || true
    sleep 2
fi

echo "Starting qwisp server..."
QWISP_PREFIX_MAX=81920 QWISP_PORT=$QWISP_PORT qwisp serve &
QWISP_PID=$!
echo "Qwisp started (PID: $QWISP_PID)"
sleep 3

echo ""
echo "=== Ready ==="
echo "Local access:      http://127.0.0.1:$QWISP_PORT"
if [ -n "$TAILSCALE_IP" ]; then
    echo "Tailscale access:  http://$TAILSCALE_IP:$QWISP_PORT"
fi
echo "OpenAI base URL:   http://${TAILSCALE_IP:-127.0.0.1}:$QWISP_PORT/v1"
echo ""
echo "Press Ctrl+C to stop qwisp"
echo ""

wait "$QWISP_PID"
