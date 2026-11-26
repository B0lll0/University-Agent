#!/usr/bin/env bash
# Launch 7-inch Touch Display in full-screen Chromium Kiosk Mode on Raspberry Pi
set -euo pipefail

URL="${1:-http://127.0.0.1:5050/edge}"
BROWSER="$(command -v chromium-browser || command -v chromium || true)"

if [ -z "$BROWSER" ]; then
  echo "Chromium not found. Open $URL manually in your browser."
  exit 1
fi

exec "$BROWSER" \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --autoplay-policy=no-user-gesture-required \
  "$URL"
