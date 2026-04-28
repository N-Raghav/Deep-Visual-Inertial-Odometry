#!/usr/bin/env bash
# Build the P4Ph2 Apptainer image.
# On HPC login nodes you typically have fakeroot; locally you may need sudo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEF="$SCRIPT_DIR/p4ph2.def"
SIF="${1:-$SCRIPT_DIR/p4ph2.sif}"

if apptainer build --fakeroot "$SIF" "$DEF"; then
    echo "built: $SIF"
else
    echo "fakeroot build failed; retrying with sudo (will prompt)..." >&2
    sudo apptainer build "$SIF" "$DEF"
    echo "built: $SIF"
fi
