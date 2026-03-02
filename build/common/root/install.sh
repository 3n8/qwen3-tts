#!/bin/bash

set -e

echo "[info] Setting up qwen3-tts..."

echo "[info] Setting locale..."
export LC_ALL=en_GB.UTF-8
export LANG=en_GB.UTF-8

echo "[info] Cleaning up..."
rm -rf /tmp/*

echo "[info] qwen3-tts setup complete"
