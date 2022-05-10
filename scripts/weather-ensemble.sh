#!/usr/bin/env bash

DATA=/nfs/stak/users/merriler/cluster/hpc-share/data/weather
NETS=/nfs/stak/users/merriler/cluster/hpc-share/output/t-we-final/TPC-Med\:*
OUT=/nfs/stak/users/merriler/cluster/hpc-share/output/

for d in 0.1 0.2 0.5 ; do
  echo $d
  python -m pointnn.eval.weather.loss --data $DATA --bs 32 --net $NETS --out $OUT/weather-drop-${d/./}.pkl --drop $d
done
