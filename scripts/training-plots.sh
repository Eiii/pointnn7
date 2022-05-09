#!/usr/bin/env bash

SC=cluster/t-sc-final
TR=cluster/t-tr-final
WE=cluster/t-we-final
SIZES="Small Med Large"

for size in $SIZES ; do 
  python -m pointnn.eval.training $SC --filter-all $size --filter-ignore OnePred,TwoPred --out sc-${size}.png
done

for size in $SIZES ; do
  python -m pointnn.eval.training $TR --filter-all $size --filter-ignore NoSpace --out tr-${size}.png --smooth 20
done

for size in $SIZES ; do
  python -m pointnn.eval.training $WE --filter-all $size --out we-${size}.png --smooth 10
done
