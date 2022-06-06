#!/usr/bin/env bash

NUMS="1 2 3"
NETS="int tpc gc"
POST="onepred twopred"

for c in $NUMS; do
  for n in $NETS; do
    for p in $POST; do
      NAME=sc-${n}-med-${p}-lrtest
      echo $NAME $c
      ./batch/submit.sh $NAME $c
    done
  done
done
