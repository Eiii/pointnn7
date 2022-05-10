#!/usr/bin/env bash

DATA=/nfs/stak/users/merriler/cluster/hpc-share/data/sc2scene/test
NETS=/nfs/stak/users/merriler/cluster/hpc-share/output/t-sc-final
OUT=/nfs/stak/users/merriler/cluster/hpc-share/output

TPCS=$NETS/TPC-Med*
TS=$(seq -s , 10)

python -m pointnn.eval.sc2.loss --data $DATA --bs 256 --net $TPCS --out $OUT/dist-sc2-tpc.pkl --ts $TS
