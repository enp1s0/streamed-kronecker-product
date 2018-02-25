#!/bin/sh


for i in `seq 100 100 3000`;do
	b_time=$(./batched-k $i | grep 計算時間 | sed -e 's/計算時間 : //' | sed -e 's/\s\[ms\]//')
	s_time=$(./seq-k $i | grep 計算時間 | sed -e 's/計算時間 : //' | sed -e 's/\s\[ms\]//')
	echo "scale=3; $s_time / $b_time" | bc
done
