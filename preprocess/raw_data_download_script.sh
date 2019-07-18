#!/bin/bash
#Usage: bash raw_data_download_script.sh GM12878

#Cell type indexes are listed as follows (refer to GSE63525):
#GM12878 idx: 1-18
#K562 idx:69-74
#IMR90 idx:50-56
#NHEK idx:65-67
#we take GM12878 cell type for an example
declare -i start_idx
declare -i end_idx
declare -i idx

PPATH=$(dirname $(readlink -f "$0"))
DPATH=${PPATH/preprocess/data}
mkdir -p "$DPATH/$1" 
	
if [ $1 == 'GM12878' ]
then
	start_idx=1
	end_idx=18
elif [ $1 == 'K562' ]
then
	start_idx=69
	end_idx=74
elif [ $1 == 'IMR90' ]
then
	start_idx=50
	end_idx=56
elif [ $1 == 'NHEK' ]
then
	start_idx=65
	end_idx=67
else
	echo "The input cell type is not identified, please refer to GEO database with accession number GSE63525"
	exit
fi

for ((idx=$start_idx; idx<=$end_idx; idx++))
do
	number=$(expr $idx + 549)
	wget -P "$DPATH/$1" ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551`printf "%03d" $number`/suppl/GSM1551`printf "%03d" $number`_HIC`printf "%03d" $idx`_merged_nodups.txt.gz
done





