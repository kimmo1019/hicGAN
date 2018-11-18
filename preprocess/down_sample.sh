#!/bin/bash
num=$(cat total_merged_nodups.txt |wc -l)
echo $num
ratio=16
num_downsample=`expr $(($num/$ratio))`
echo $num_downsample
shuf -n $num_downsample total_merged_nodups.txt | sort -k3,3d -k7,7d  > total_merged_nodups_downsample_ratio_$ratio.txt

