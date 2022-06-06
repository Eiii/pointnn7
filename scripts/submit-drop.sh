#!/usr/bin/env bash

NUMS=$(seq 2)
NETS="tpc int gc"

for c in $NUMS; do
  for n in $NETS; do
    NAME=we-${n}-med-drop2
    echo $NAME $c
    ./batch/submit.sh $NAME $c
  done
done
