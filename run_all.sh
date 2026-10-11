#!/bin/sh
# Runs the checks and every step in order. About 3 minutes on a laptop, no graphics card, no download.
set -e
cd "$(dirname "$0")"
python3 tests.py
for f in step1_club step2_alone step3_round step4_train step5_names step6_rounds step7_twins; do
  echo; echo "== $f"; python3 $f.py
done
