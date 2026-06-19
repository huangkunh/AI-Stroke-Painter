#!/usr/bin/env bash
# =============================================================================
# Download pretrained weights for the Learning-to-Paint agent.
#
# The original ICCV2019-LearningToPaint release (https://github.com/hzwer/ICCV2019-LearningToPaint)
# ships the following files on Google Drive:
#   * actor_final.pth         (the Agent / policy network)  <-- needed for inference
#   * renderer_final.pth      (the neural renderer)          <-- only for training
#   * discriminator_final.pth (the discriminator)            <-- only for training
#
# This script downloads actor_final.pth into model/pretrained/.
# It tries multiple known file IDs / mirror URLs in order, and verifies the
# downloaded file is non-trivial in size. If all automatic methods fail it
# prints a clear manual-download guide.
#
# Usage:
#   bash model/download_weights.sh
#
# Exit codes:
#   0  - weights ready (downloaded or already present)
#   1  - all automatic downloads failed (see manual instructions in stderr)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/pretrained"
TARGET="${DEST}/actor_final.pth"
MIN_SIZE_BYTES=500000   # 500 KB – sanity check; real weights are ~MBs

mkdir -p "$DEST"

# ---------------------------------------------------------------------------
# 0. Already present?
# ---------------------------------------------------------------------------
if [ -f "$TARGET" ]; then
    SIZE=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET" 2>/dev/null || echo 0)
    if [ "$SIZE" -ge "$MIN_SIZE_BYTES" ]; then
        echo "[download_weights] ✓ actor_final.pth already present ($SIZE bytes), skipping."
        exit 0
    else
        echo "[download_weights] existing file too small ($SIZE bytes), re-downloading..."
        rm -f "$TARGET"
    fi
fi

# ---------------------------------------------------------------------------
# 1. Ensure gdown is available
# ---------------------------------------------------------------------------
echo "[download_weights] ensuring gdown is installed..."
PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -m gdown --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --quiet --upgrade gdown >/dev/null 2>&1 || {
        echo "[download_weights] WARNING: failed to install gdown." >&2
    }
fi

# ---------------------------------------------------------------------------
# 2. Try each known source in order
# ---------------------------------------------------------------------------
# File IDs published by hzwer/ICCV2019-LearningToPaint (Google Drive).
# These may rotate; we keep several candidates and try them sequentially.
GDRIVE_IDS=(
    "1YJ0EoZGbNqEbE7h3kR8E5B8qZ1uPpPQF"   # actor_final.pth (primary)
    "1rWVOcK5JUXa7gH5MnPYbf7FjT1XeDS0b"   # alternate mirror
)

download_via_gdown() {
    local id="$1"
    echo "[download_weights] trying gdown with file ID: $id"
    # --fuzzy lets gdown handle share-link or direct-ID forms.
    if "$PYTHON_BIN" -m gdown --fuzzy "https://drive.google.com/uc?id=${id}" -O "$TARGET" 2>&1; then
        if [ -f "$TARGET" ]; then
            local sz
            sz=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET" 2>/dev/null || echo 0)
            if [ "$sz" -ge "$MIN_SIZE_BYTES" ]; then
                echo "[download_weights] ✓ downloaded via gdown ($sz bytes)"
                return 0
            fi
            echo "[download_weights] gdown output too small ($sz bytes), discarding." >&2
            rm -f "$TARGET"
        fi
    fi
    return 1
}

download_via_curl() {
    # Direct curl fallback (works only if the file is publicly downloadable).
    local id="$1"
    echo "[download_weights] trying direct curl for file ID: $id"
    if curl -fSL "https://drive.google.com/uc?export=download&id=${id}" -o "$TARGET" 2>&1; then
        if [ -f "$TARGET" ]; then
            local sz
            sz=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET" 2>/dev/null || echo 0)
            if [ "$sz" -ge "$MIN_SIZE_BYTES" ]; then
                echo "[download_weights] ✓ downloaded via curl ($sz bytes)"
                return 0
            fi
            rm -f "$TARGET"
        fi
    fi
    return 1
}

SUCCESS=0
for id in "${GDRIVE_IDS[@]}"; do
    if download_via_gdown "$id"; then SUCCESS=1; break; fi
    if download_via_curl "$id";  then SUCCESS=1; break; fi
done

# ---------------------------------------------------------------------------
# 3. Report result
# ---------------------------------------------------------------------------
if [ "$SUCCESS" -eq 1 ]; then
    echo "[download_weights] Done. Weights saved to: $TARGET"
    ls -lh "$DEST"
    exit 0
fi

cat >&2 <<EOF
[download_weights] ✗ All automatic download methods failed.

This is usually caused by:
  - Google Drive quota / rate-limiting for shared files
  - Network restrictions in this environment
  - The file ID having been rotated upstream

Please download actor_final.pth MANUALLY:
  1. Visit https://github.com/hzwer/ICCV2019-LearningToPaint
  2. Find the Google Drive link for actor_final.pth in the README
  3. Download it and place it at:
       $TARGET

Once the file is in place, re-run inference with --mode rl (or --mode auto).
If you just want a quick demo, run with --mode lite instead — no weights needed.
EOF
exit 1
