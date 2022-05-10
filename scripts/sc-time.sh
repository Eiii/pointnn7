#!/usr/bin/env bash

FILTERS="TPC Int GC"
DATA=cluster

for f in $FILTERS; do
  python -m pointnn.eval.sc2.loss plot $DATA/sc2*.pkl --filter $f --out sc2pred_${f}.png
done
