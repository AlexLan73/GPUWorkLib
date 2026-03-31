#!/bin/bash
# ============================================================
#  GPUWorkLib - Doxygen Build Script
#  Order: clean -> DrvGPU -> modules pass1 (.tag) ->
#         modules pass2 (cross-links) -> main (TAGFILES)
# ============================================================
set -e
DOXYGEN="doxygen"

MODULES="signal_generators fft_func filters statistics heterodyne vector_algebra capon range_angle fm_correlator strategies lch_farrow"

echo "============================================================"
echo " GPUWorkLib Doxygen Build"
echo "============================================================"

# --- Step 0: Clean ---
echo ""
echo "=== Step 0/5: Clean ==="

rm -rf html
rm -rf DrvGPU/html DrvGPU/drvgpu.tag

for m in $MODULES; do
    rm -rf "modules/$m/html" "modules/$m/$m.tag"
done

echo "    Clean OK"

# --- Step 1: DrvGPU ---
echo ""
echo "=== Step 1/5: DrvGPU ==="
(cd DrvGPU && $DOXYGEN Doxyfile)
echo "    DrvGPU OK"

# --- Step 2: Modules pass 1 — generate .tag files ---
echo ""
echo "=== Step 2/5: Modules pass 1 (generate .tag) ==="
for m in $MODULES; do
    echo "--- $m ---"
    (cd "modules/$m" && $DOXYGEN Doxyfile)
    echo "    $m OK"
done

# --- Step 3: Modules pass 2 — rebuild with cross-module TAGFILES ---
echo ""
echo "=== Step 3/5: Modules pass 2 (cross-links) ==="
for m in $MODULES; do
    echo "--- $m ---"
    # Build TAGFILES string: DrvGPU + all OTHER modules
    TAGS="../../DrvGPU/drvgpu.tag=../../../DrvGPU/html"
    for other in $MODULES; do
        if [ "$other" != "$m" ]; then
            TAGS="$TAGS ../../modules/$other/$other.tag=../../../modules/$other/html"
        fi
    done
    (cd "modules/$m" && ( TAGFILES="$TAGS" $DOXYGEN <(cat Doxyfile; echo "TAGFILES = $TAGS") ))
    echo "    $m OK"
done

# --- Step 4: Main ---
echo ""
echo "=== Step 4/5: Main Doxyfile (TAGFILES) ==="
$DOXYGEN Doxyfile
echo "    Main OK"

echo ""
echo "============================================================"
echo " Done!"
echo "============================================================"
