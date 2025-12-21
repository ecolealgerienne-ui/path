#!/bin/bash
# Manual download helper for CoNSeP dataset
#
# CoNSeP dataset is available from:
# 1. Official Warwick site (may require authentication)
# 2. Google Drive (from HoVer-Net paper supplementary materials)
# 3. Direct contact with authors

set -e

OUTPUT_DIR="${1:-data/evaluation/consep}"

echo "=================================================================="
echo "CoNSeP Dataset Manual Download Helper"
echo "=================================================================="
echo ""
echo "The CoNSeP dataset (70 MB) must be downloaded manually."
echo ""
echo "📥 Download options:"
echo ""
echo "1. Official Warwick site (recommended):"
echo "   https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/"
echo ""
echo "2. HoVer-Net GitHub (check Releases):"
echo "   https://github.com/vqdang/hover_net"
echo ""
echo "3. Google Drive (from paper supplementary):"
echo "   Search for 'CoNSeP dataset HoVer-Net' on Google Scholar"
echo "   Download from paper's supplementary materials"
echo ""
echo "=================================================================="
echo ""
echo "📁 After download, place the file here:"
echo "   $OUTPUT_DIR/consep_dataset.zip"
echo ""
echo "Then extract:"
echo "   unzip $OUTPUT_DIR/consep_dataset.zip -d $OUTPUT_DIR/"
echo ""
echo "Or use the automated conversion:"
echo "   python scripts/evaluation/download_evaluation_datasets.py --dataset consep"
echo ""
echo "=================================================================="
echo ""

# Check if file already exists
if [ -f "$OUTPUT_DIR/consep_dataset.zip" ]; then
    SIZE=$(stat -f%z "$OUTPUT_DIR/consep_dataset.zip" 2>/dev/null || stat -c%s "$OUTPUT_DIR/consep_dataset.zip" 2>/dev/null)
    SIZE_MB=$((SIZE / 1024 / 1024))

    if [ $SIZE_MB -gt 50 ]; then
        echo "✅ Found existing file: $OUTPUT_DIR/consep_dataset.zip ($SIZE_MB MB)"
        echo ""
        read -p "Extract now? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mkdir -p "$OUTPUT_DIR"
            unzip -o "$OUTPUT_DIR/consep_dataset.zip" -d "$OUTPUT_DIR/"
            echo ""
            echo "✅ Extraction complete!"
            echo "   Train: $(ls "$OUTPUT_DIR/Train" 2>/dev/null | wc -l) files"
            echo "   Test:  $(ls "$OUTPUT_DIR/Test" 2>/dev/null | wc -l) files"
        fi
    else
        echo "⚠️  File exists but is too small ($SIZE_MB MB < 50 MB)"
        echo "    This may be an incomplete download or HTML redirect page."
        echo "    Please re-download the dataset."
    fi
else
    echo "❌ File not found: $OUTPUT_DIR/consep_dataset.zip"
    echo ""
    echo "Please download manually from the URLs above."
fi

echo ""
echo "=================================================================="
