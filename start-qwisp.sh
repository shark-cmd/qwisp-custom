#!/bin/bash
# Start qwisp server directly (no Nginx proxy needed).
# Accessible on both localhost and Tailscale network.
#
# Environment variables (all optional):
#
#   QWISP_PORT          Server port (default: 9080)
#   QWISP_PREFIX_MAX    Max context tokens (default: 81920 = 80K)
#   QWISP_SLIDING_WINDOW  Enable sliding window compacting.
#                         Set to target window size in tokens (0 = disabled).
#                         Fires when context >= QWISP_SLIDING_WINDOW - QWISP_WINDOW_HEADROOM.
#                         On trigger: drops oldest tokens, resets KV cache, re-prefills
#                         truncated context from scratch (correct RoPE positions).
#   QWISP_WINDOW_HEADROOM  Tokens to preserve after compacting (default: 4096).
#                           Effective window kept = QWISP_SLIDING_WINDOW - headroom.
#
# Production example (fires only near 80K limit):
#   QWISP_SLIDING_WINDOW=75000 QWISP_WINDOW_HEADROOM=6000
#
# Recommended for agentic workloads with large system prompts (e.g. 50K tokens):
#   QWISP_SLIDING_WINDOW=32768 QWISP_WINDOW_HEADROOM=4096
#   → compacts to ~28K, keeps SDPA decode at full speed (~75-85 tok/s)
#   → without this, 50K+ KV length slows decode to 24-27 tok/s

set -e

QWISP_PORT=${QWISP_PORT:-9080}
TAILSCALE_IP=$(tailscale ip 2>/dev/null | head -1)

echo "=== Qwisp Server ==="
echo ""

# Kill Nginx if running (no longer needed — qwisp binds 0.0.0.0 directly)
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
QWISP_PREFIX_MAX=81920 \
QWISP_SLIDING_WINDOW=32768 \
QWISP_WINDOW_HEADROOM=4096 \
QWISP_PORT=$QWISP_PORT \
qwisp serve &
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
