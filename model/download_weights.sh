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
# It tries multiple known file IDs / mirror URLs in order, verifies the
# downloaded file is non-trivial in size AND (optionally) matches a known
# SHA256 checksum. If all automatic methods fail it prints a clear
# manual-download guide.
#
# Usage:
#   bash model/download_weights.sh              # download + size check
#   bash model/download_weights.sh --check      # only verify existing file
#   EXPECTED_SHA256=<hex> bash model/download_weights.sh   # enforce hash
#
# Exit codes:
#   0  - weights ready (downloaded or already present) and verified
#   1  - all automatic downloads failed (see manual instructions in stderr)
#   2  - file present but failed integrity check
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/pretrained"
TARGET="${DEST}/actor_final.pth"
MIN_SIZE_BYTES=500000   # 500 KB – sanity check; real weights are ~MBs

# Optional: if EXPECTED_SHA256 is set in the environment, enforce it.
# (Left empty by default because the upstream file may be re-encoded.)
EXPECTED_SHA256="${EXPECTED_SHA256:-}"

# ---------------------------------------------------------------------------
# 0. Already present?
# ---------------------------------------------------------------------------
mkdir -p "$DEST"

if [ "${1:-}" = "--check" ]; then
    if [ -f "$TARGET" ]; then
        verify_file
        echo "[download_weights] ✓ existing file verified."
        exit 0
    else
        echo "[download_weights] ✗ no file to check at $TARGET" >&2
        exit 2
    fi
fi

if [ -f "$TARGET" ]; then
    SIZE=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET" 2>/dev/null || echo 0)
    if [ "$SIZE" -ge "$MIN_SIZE_BYTES" ]; then
        echo "[download_weights] actor_final.pth already present ($SIZE bytes)."
        if verify_file; then
            echo "[download_weights] ✓ integrity check passed."
            exit 0
        else
            echo "[download_weights] ✗ integrity check failed; re-downloading." >&2
            rm -f "$TARGET"
        fi
    else
        echo "[download_weights] existing file too small ($SIZE bytes); re-downloading."
        rm -f "$TARGET"
    fi
fi

# ---------------------------------------------------------------------------
# 1. Candidate Google Drive file IDs (tried in order)
# ---------------------------------------------------------------------------
# The canonical ID published in hzwer/ICCV2019-LearningToPaint README.
# Additional IDs can be appended here if mirrors become available.
GDRIVE_IDS=(
    "1YJ0EoZGbNqEbE7h3kR8E5B8qZ1uPpPQF"
)

# ---------------------------------------------------------------------------
# 2. Download helpers
# ---------------------------------------------------------------------------
verify_file() {
    # Returns 0 (success) if the file passes size + optional hash checks.
    local f="$TARGET"
    if [ ! -f "$f" ]; then return 1; fi
    local sz
    sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    if [ "$sz" -lt "$MIN_SIZE_BYTES" ]; then
        echo "[download_weights] ✗ file too small ($sz < $MIN_SIZE_BYTES bytes)" >&2
        return 1
    fi
    if [ -n "$EXPECTED_SHA256" ]; then
        local actual
        actual=$(sha256sum "$f" 2>/dev/null | awk '{print $1}' || \
                 shasum -a 256 "$f" 2>/dev/null | awk '{print $1}')
        if [ "$actual" != "$EXPECTED_SHA256" ]; then
            echo "[download_weights] ✗ SHA256 mismatch" >&2
            echo "[download_weights]   expected: $EXPECTED_SHA256" >&2
            echo "[download_weights]   actual:   $actual" >&2
            return 1
        fi
        echo "[download_weights] ✓ SHA256 verified: $actual"
    fi
    return 0
}

download_via_gdown() {
    local id="$1"
    echo "[download_weights] trying gdown (id=$id) ..."
    python -m pip install --quiet --upgrade gdown >/dev/null 2>&1 || true
    if python -m gdown "https://drive.google.com/uc?id=${id}" -O "$TARGET" 2>&1 \
        | grep -v "Downloading\|100%\|^\s*$"; then
        if verify_file; then
            echo "[download_weights] ✓ downloaded via gdown"
            return 0
        fi
    fi
    rm -f "$TARGET"
    return 1
}

download_via_curl() {
    local id="$1"
    echo "[download_weights] trying curl (id=$id) ..."
    if curl -fsSL "https://drive.google.com/uc?export=download&id=${id}" \
        -o "$TARGET" 2>/dev/null; then
        if verify_file; then
            echo "[download_weights] ✓ downloaded via curl"
            return 0
        fi
    fi
    rm -f "$TARGET"
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
    # Final integrity report
    ACTUAL_SHA=$(sha256sum "$TARGET" 2>/dev/null | awk '{print $1}' || \
                 shasum -a 256 "$TARGET" 2>/dev/null | awk '{print $1}')
    echo "[download_weights] SHA256: ${ACTUAL_SHA:-unknown}"
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
  4. (Optional) Verify integrity:
       bash model/download_weights.sh --check

Once the file is in place, re-run inference with --mode rl (or --mode auto).
If you just want a quick demo, run with --mode lite instead — no weights needed.
EOF
exit 1
