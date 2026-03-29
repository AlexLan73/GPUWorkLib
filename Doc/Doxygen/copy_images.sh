#!/bin/bash
# ============================================================
#  GPUWorkLib - Copy plot images to Doc/Modules/{module}/images/
#  Run from: Doc/Doxygen/  (or project root)
#  Purpose: stable image storage for Doxygen docs
# ============================================================
set -e

# Detect project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/Doxyfile" ]; then
    ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    ROOT="$(pwd)"
fi

echo "============================================================"
echo " Copy plot images -> Doc/Modules/{module}/images/"
echo " Project root: $ROOT"
echo "============================================================"

# --- Source directories (check both Results/Plots and build/**/Results/Plots) ---
RESULTS_PLOTS="$ROOT/Results/Plots"
BUILD_PLOTS=""

# Search for plots in build/ directories
for d in "$ROOT"/build/*/Results/Plots "$ROOT"/build/Results/Plots; do
    if [ -d "$d" ]; then
        BUILD_PLOTS="$d"
        echo "Found build plots: $BUILD_PLOTS"
        break
    fi
done

DOC_MODULES="$ROOT/Doc/Modules"

# --- Copy function: source_dir -> doc_module_name ---
copy_module() {
    local src_name="$1"    # name in Results/Plots/
    local dst_name="$2"    # name in Doc/Modules/
    local dst="$DOC_MODULES/$dst_name/images"

    mkdir -p "$dst"

    local count=0

    # 1. Copy from Results/Plots/
    if [ -d "$RESULTS_PLOTS/$src_name" ]; then
        cp -ru "$RESULTS_PLOTS/$src_name/"* "$dst/" 2>/dev/null && \
            count=$(find "$RESULTS_PLOTS/$src_name" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" \) | wc -l)
        echo "    $src_name (Results/Plots): $count files"
    fi

    # 2. Copy from build/.../Results/Plots/ (if exists, newer files win)
    if [ -n "$BUILD_PLOTS" ] && [ -d "$BUILD_PLOTS/$src_name" ]; then
        local build_count=$(find "$BUILD_PLOTS/$src_name" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" \) | wc -l)
        cp -ru "$BUILD_PLOTS/$src_name/"* "$dst/" 2>/dev/null
        echo "    $src_name (build):         $build_count files"
    fi

    # Show total in destination
    local total=$(find "$dst" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" \) 2>/dev/null | wc -l)
    echo "    => $dst_name/images/: $total files total"
}

echo ""
echo "=== Copying images ==="

# Module mapping: Results/Plots name -> Doc/Modules name
copy_module "fft_maxima"         "fft_func"
copy_module "filters"            "filters"
copy_module "heterodyne"         "heterodyne"
copy_module "signal_generators"  "signal_generators"
copy_module "statistics"         "statistics"
copy_module "strategies"         "strategies"
copy_module "vector_algebra"     "vector_algebra"
copy_module "capon"              "capon"
copy_module "range_angle"        "range_angle"
copy_module "fm_correlator"      "fm_correlator"
copy_module "lch_farrow"         "lch_farrow"
copy_module "integration"        "integration"

echo ""
echo "=== Summary ==="
total=$(find "$DOC_MODULES"/*/images -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.svg" \) 2>/dev/null | wc -l)
echo "Total images in Doc/Modules/*/images/: $total"
echo "============================================================"
echo " Done! Now run: ./build_docs.sh"
echo "============================================================"
