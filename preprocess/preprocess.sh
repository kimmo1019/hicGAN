#!/bin/bash
DPATH=$1
CELL=$2
mkdir -p "$DPATH/$CELL"
#merge raw data 
find "$DPATH/$CELL" -name "*_merged_nodups.txt.gz"|xargs zcat | sort -k3,3d -k7,7d > "$DPATH/$CELL/total_merged_nodups.txt"
#downsample
ratio=16
num=$(cat $DPATH/$CELL/total_merged_nodups.txt |wc -l)
num_downsample=`expr $(($num/$ratio))`
shuf -n $num_downsample $DPATH/$CELL/total_merged_nodups.txt | sort -k3,3d -k7,7d  > $DPATH/$CELL/total_merged_nodups_downsample_ratio_$ratio.txt
echo "merge data done!"

mkdir -p "$DPATH/$CELL/intra_NONE"
mkdir -p "$DPATH/$CELL/intra_VC"
mkdir -p "$DPATH/$CELL/intra_KR"

juicer_tool="/home/liuqiao/software/juicer/scripts/juicer_tools.jar"
#generate .HIC file using juicer tool
java -Xmx50g  -jar $juicer_tool pre $DPATH/$CELL/total_merged_nodups.txt $DPATH/$CELL/total_merged.hic hg19
java -Xmx50g  -jar $juicer_tool pre $DPATH/$CELL/total_merged_nodups_downsample_ratio_$ratio.txt $DPATH/$CELL/total_merged_downsample_ratio_$ratio.hic hg19

chromes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 "X" "Y")

#generate Hi-C raw contacts using iuicer tool
for chrom in ${chromes[@]}
do
	sbatch sbatch_juicer_script.sh $chrom $DPATH $CELL
	#directly run "bash sbatch_juicer_script.sh $chrom $DPATH $CELL" if slurm is not installed
done
