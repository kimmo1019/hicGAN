#!/bin/bash

#Cell type indexes are as follows:
#GM12878 idx: 1-18
#K562 idx:69-74
#IMR90 idx:50-56
#NHEK idx:65-67
#we take GM12878 cell type for an example

for idx in {1..18}
do
	number=$(expr $idx + 549)
	folder_idx=`printf "%03d" $number`
	#echo $folder_idx
	wget ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551${folder_idx}/suppl/GSM1551${folder_idx}_HIC$1_merged_nodups.txt.gz
done
