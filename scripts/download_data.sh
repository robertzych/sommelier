#!/usr/bin/env bash
# Download pre-built Qdrant data from the latest GitHub release.
# Run from the sommelier repo root: bash scripts/download_data.sh
set -euo pipefail

RELEASE_URL="https://github.com/robertzych/sommelier/releases/latest/download/qdrant_storage.zip"
DEST="qdrant_storage"

if [ -d "$DEST" ]; then
    echo "qdrant_storage/ already exists. Delete it first to re-download."
    exit 1
fi

echo "Downloading pre-built Qdrant data..."
curl -L --progress-bar -o qdrant_storage.zip "$RELEASE_URL"
echo "Extracting..."
unzip -q qdrant_storage.zip
rm qdrant_storage.zip
echo "Done. Run: uv run sommelier query \"What is the default broker port?\""
