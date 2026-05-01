#!/bin/bash

# Project Migrator for Google Colab
# Packaging only the essentials for cloud training

PROJECT_ROOT="/Users/shriyakotala/Documents/MS_proj/prj"
ZIP_NAME="project_migration.zip"

echo "📦 Packaging project for Google Colab..."

# Move to project root
cd "$PROJECT_ROOT"

# Clean up any remaining pycache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Create the zip archive
# -r: recursive
# -q: quiet
zip -r "$ZIP_NAME" src data checkpoints outputs pyproject.toml setup.sh README.md uv.lock -x "*.DS_Store" "*__pycache__*"

echo "✅ Done! Migration archive created: $PROJECT_ROOT/$ZIP_NAME"
echo "👉 Upload this file to your Google Drive to continue on Colab."
