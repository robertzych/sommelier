#!/usr/bin/env bash
# Create qdrant_storage.zip for upload to GitHub Releases.
# Run from the sommelier repo root after ingestion: bash scripts/create_data_zip.sh
set -euo pipefail

if [ ! -d "qdrant_storage" ]; then
    echo "qdrant_storage/ not found. Run ingestion first:"
    echo "  uv run sommelier ingest --docs-path /path/to/pinot-docs --version latest"
    exit 1
fi

zip -r qdrant_storage.zip qdrant_storage/
echo "Created qdrant_storage.zip ($(du -sh qdrant_storage.zip | cut -f1))."
echo "Upload this file to the GitHub release: https://github.com/robertzych/sommelier/releases"
