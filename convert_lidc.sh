#!/usr/bin/env bash
set -euo pipefail

# ensure dcmdump & MITK-Phenotyping in PATH
if ! command -v dcmdump &>/dev/null; then
  echo "install dcmtk (e.g. sudo apt install dcmtk)" >&2; exit 1
fi
export PATH=/opt/MITK-Phenotyping/bin:$PATH
export LD_LIBRARY_PATH=/opt/MITK-Phenotyping/bin:~/CS_598/mitk_libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RAW_ROOT="$HOME/CS_598/LIDC-Dataset-Sample"
OUT_ROOT="$HOME/CS_598/LIDC-Dataset-Sample-Converted"
mkdir -p "$OUT_ROOT/planars"

for PAT_DIR in "$RAW_ROOT"/LIDC-IDRI-*; do
  pid=$(basename "$PAT_DIR")
  echo "=== Patient $pid ==="

  # find every folder that contains DICOMs
  mapfile -t SCAN_DIRS < <(
    find "$PAT_DIR" -type f -name '*.dcm' -printf '%h\n' | sort -u
  )

  # convert each CT folder
  for SCAN_DIR in "${SCAN_DIRS[@]}"; do
    scan_id=$(basename "$SCAN_DIR")
    OUT_VOL="$OUT_ROOT/${pid}_${scan_id}_ct_scan.nrrd"
    echo "-> [${pid}/${scan_id}] DICOM→NRRD → $OUT_VOL"
    if ! MitkCLDicom2Nrrd -i "$SCAN_DIR" -o "$OUT_VOL"; then
      echo "   ⚠️ Conversion failed for $scan_id" >&2
    fi
  done

  # pick up the annotation XML
  XML_FILE=$(find "$PAT_DIR" -type f -name '*.xml' | head -n1 || true)
  if [[ -z "$XML_FILE" ]]; then
    echo "⚠️ No XML for $pid, skipping mask"
    continue
  fi

  # generate a mask via the PlanarFigure→NRRD tool
  echo "-> [${pid}] Generating mask via PlanarFigure→NRRD"
  VOL_IN=$(ls "$OUT_ROOT/${pid}"*_ct_scan.nrrd | head -n1)
  OUT_MASK="$OUT_ROOT/planars/${pid}_mask.nrrd"

  if ! MitkCLPlanarFigureToNrrd \
        --planar             "$XML_FILE" \
        --image              "$VOL_IN" \
        --output             "$OUT_MASK"
  then
    echo "   ⚠️ Planar→NRRD failed for $pid" >&2
  else
    echo "   ✅ Wrote mask: $OUT_MASK"
  fi

  echo "✅ Done with $pid"
done

echo "All done! Volumes in $OUT_ROOT, masks in $OUT_ROOT/planars"
