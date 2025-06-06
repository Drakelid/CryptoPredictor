#!/bin/sh

# Create a better API URL that works from browser
# If VITE_API_BASE_URL contains 'backend:8000', replace with localhost:8000 for browser compatibility
BROWSER_API_URL="$VITE_API_BASE_URL"
echo "Original VITE_API_BASE_URL: $VITE_API_BASE_URL"

if echo "$BROWSER_API_URL" | grep -q "backend:8000"; then
  # When accessed from a browser, 'backend' hostname won't resolve
  # Replace it with localhost for browser access
  BROWSER_API_URL="http://localhost:8000"
  echo "Modified API URL for browser compatibility: $BROWSER_API_URL"
fi

# Generate runtime config with actual environment variables
echo "// This file exposes environment variables to the frontend at runtime" > /app/dist/env-config.js
echo "window.__ENV = window.__ENV || {};" >> /app/dist/env-config.js
echo "window.__ENV.VITE_API_BASE_URL = \"$BROWSER_API_URL\";" >> /app/dist/env-config.js
echo "window.__ENV.ORIGINAL_API_URL = \"$VITE_API_BASE_URL\";" >> /app/dist/env-config.js
echo "window.__ENV.DEBUG_MODE = true;" >> /app/dist/env-config.js

# Show the content of the created file for debugging
echo "Generated env-config.js content:"
cat /app/dist/env-config.js

echo "Starting server at $(date)"

# Start the serve command
exec "$@"
