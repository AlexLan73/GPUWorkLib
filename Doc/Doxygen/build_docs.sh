#!/bin/bash
# ============================================================
#  GPUWorkLib - Doxygen Build Script
#  Order: clean -> DrvGPU (base) -> modules -> main (TAGFILES)
# ============================================================
set -e
DOXYGEN="doxygen"

MODULES="signal_generators fft_func filters statistics heterodyne vector_algebra capon range_angle fm_correlator strategies lch_farrow"

echo "============================================================"
echo " GPUWorkLib Doxygen Build"
echo "============================================================"

# --- Step 0: Clean ---
echo ""
echo "=== Step 0/4: Clean ==="

rm -rf html
rm -rf DrvGPU/html DrvGPU/drvgpu.tag

for m in $MODULES; do
    rm -rf "modules/$m/html" "modules/$m/$m.tag"
done

echo "    Clean OK"

# --- Step 1: DrvGPU ---
echo ""
echo "=== Step 1/4: DrvGPU ==="
(cd DrvGPU && $DOXYGEN Doxyfile)
echo "    DrvGPU OK"

# --- Step 2: Modules ---
echo ""
echo "=== Step 2/4: Modules ==="
for m in $MODULES; do
    echo "--- $m ---"
    (cd "modules/$m" && $DOXYGEN Doxyfile)
    echo "    $m OK"
done

# --- Step 3: Main ---
echo ""
echo "=== Step 3/4: Main Doxyfile (TAGFILES) ==="
$DOXYGEN Doxyfile
echo "    Main OK"

echo ""
echo "============================================================"
echo " Done!"
echo "============================================================"
