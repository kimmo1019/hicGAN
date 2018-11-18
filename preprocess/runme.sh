#!/bin/bash
chromes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 "X" "Y")
#for idx in {1..18}
do
	i=`printf "%02d" $idx`
	idx2=`expr $idx + 49`
	j=`printf "%02d" $idx2`
	echo $i, $j
	sbatch submit.sh $i $j
done
