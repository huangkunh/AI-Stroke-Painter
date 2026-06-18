#!/usr/bin/env bash
# Download pretrained weights for the Learning-to-Paint agent/renderer.
#
# The original ICCV2019-LearningToPaint release ships the following files:
#   * actor_final.pth        (the Agent / policy network)
#   * renderer_final.pth     (the neural renderer, only needed for training)
#   * discriminator_final.pth (only needed for training)
#
# For inference we only need actor_final.pth. The upstream release is hosted
# on Google Drive; the IDs below are the ones published in the README of
# https://github.com/hzwer/ICCV2019-LearningToPaint.
#
# Usage:
#   bash model/download_weights.sh
#
# If gdown is not installed, it will be installed via pip into the current
# environment. If the download fails (network / quota), please download
# manually from the Google Drive link above and place `actor_final.pth`
# into model/pretrained/.

set -e

DEST="$(dirname "$0")/pretrained"
mkdir -p "$DEST"

if [ -f "$DEST/actor_final.pth" ]; then
  echo "[download_weights] actor_final.pth already present, skipping."
  exit 0
fi

echo "[download_weights] Installing gdown ..."
python -m pip install --quiet --upgrade gdown >/dev/null 2>&1 || true

echo "[download_weights] Downloading actor_final.pth from Google Drive ..."
# File ID published by hzwer/ICCV2019-LearningToPaint
python -m gdown "https://drive.google.com/uc?id=1YJ0EoZGbNqEbE7h3kR8E5B8qZ1uPpPQF" -O "$DEST/actor_final.pth" \
  || python -m gdown "1YJ0EoZGbNqEbE7h3kR8E5B8qZ1uPpPQF" -O "$DEST/actor_final.pth" \
  || {
      echo "[download_weights] Automatic download failed."
      echo "[download_weights] Please download actor_final.pth manually from"
      echo "[download_weights]   https://github.com/hzwer/ICCV2019-LearningToPaint"
      echo "[download_weights] and place it at: $DEST/actor_final.pth"
      exit 1
  }

echo "[download_weights] Done. Weights are in $DEST"
ls -lh "$DEST"
